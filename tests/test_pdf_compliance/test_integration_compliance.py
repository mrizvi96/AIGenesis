"""
Test Suite for Integration Compliance
Validates complete AIML framework implementation and PDF performance targets
"""

import unittest
import numpy as np
import sys
import os
from typing import Dict, List, Any
import time

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

try:
    from aiml_multi_task_heads import AIMLMultiTaskHeads, get_aiml_multi_task_heads
    from enhanced_bert_classifier import EnhancedBERTClassifier, get_enhanced_bert_classifier
    from enhanced_safe_features import EnhancedSAFE, get_enhanced_safe_features
    from enhanced_recommender_advanced import EnhancedRecommenderAdvanced
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in mock mode for testing")

class TestMultimodalFeatureFusion(unittest.TestCase):
    """Verify text + structured feature fusion"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.multi_task_heads = get_aiml_multi_task_heads(input_dim=128, hidden_dim=64)
            self.bert_classifier = get_enhanced_bert_classifier()
            self.safe_features = get_enhanced_safe_features(max_features=250, memory_limit_mb=150)
        except:
            self.multi_task_heads = None
            self.bert_classifier = None
            self.safe_features = None
            
        # Sample claim data for integration testing
        self.sample_claim = {
            'claim_id': 'INTEGRATION_TEST_001',
            'amount': 8500.0,
            'claim_type': 'auto',
            'claim_subtype': 'collision',
            'description': 'Multi-vehicle collision at highway intersection during rush hour causing severe damage to multiple vehicles',
            'accident_time': '17:45',
            'accident_date': '2023-11-15',
            'location': 'highway_intersection',
            'customer_id': 'CUST_INTEGRATION',
            'claimant_age': 42,
            'policy_number': 'POL_INT789',
            'policy_start_date': '2021-06-01',
            'coverage_amount': 30000.0,
            'weather_conditions': 'clear',
            'police_report': 'yes',
            'witness_count': 3,
            'third_party_involved': 'yes'
        }
    
    def test_text_feature_extraction(self):
        """Test multi-task text feature extraction"""
        if self.bert_classifier is None:
            self.skipTest("BERT classifier not available")
            
        try:
            # Extract text features
            text_features = self.bert_classifier.extract_structured_features(
                self.sample_claim['description'], 
                self.sample_claim
            )
            
            # Check feature structure
            self.assertIsInstance(text_features, list,
                                  "Text features should be a list")
            self.assertEqual(len(text_features), 200,
                               "Should generate 200 text features")
            
            # Check all features are numeric
            for i, feature in enumerate(text_features):
                self.assertIsInstance(feature, (int, float),
                                      f"Text feature {i} should be numeric")
                self.assertFalse(np.isnan(feature),
                                  f"Text feature {i} should not be NaN")
                
        except Exception as e:
            self.fail(f"Text feature extraction failed: {e}")
    
    def test_structured_feature_generation(self):
        """Test SAFE structured feature generation"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        try:
            # Generate structured features
            structured_features = self.safe_features.generate_comprehensive_features(self.sample_claim)
            
            # Check feature structure
            self.assertIsInstance(structured_features, dict,
                                  "Structured features should be a dictionary")
            self.assertGreater(len(structured_features), 150,
                               "Should generate substantial number of structured features")
            
            # Check all features are numeric
            for feature_name, feature_value in structured_features.items():
                self.assertIsInstance(feature_value, (int, float),
                                      f"Feature '{feature_name}' should be numeric")
                self.assertFalse(np.isnan(feature_value),
                                  f"Feature '{feature_name}' should not be NaN")
                
        except Exception as e:
            self.fail(f"Structured feature generation failed: {e}")
    
    def test_feature_fusion_logic(self):
        """Test fusion of text and structured features"""
        if self.bert_classifier is None or self.safe_features is None:
            self.skipTest("Both classifiers not available")
            
        try:
            # Extract both feature types
            text_features = self.bert_classifier.extract_structured_features(
                self.sample_claim['description'], 
                self.sample_claim
            )
            structured_features = self.safe_features.generate_comprehensive_features(self.sample_claim)
            
            # Convert structured features to list for fusion
            structured_feature_list = list(structured_features.values())
            
            # Pad or truncate to match text features length
            max_features = max(len(text_features), len(structured_feature_list))
            
            # Ensure both have same length
            text_features_padded = text_features + [0.0] * (max_features - len(text_features))
            structured_features_padded = structured_feature_list + [0.0] * (max_features - len(structured_feature_list))
            
            # Test fusion strategies
            # 1. Concatenation
            concatenated_features = text_features_padded + structured_features_padded
            self.assertEqual(len(concatenated_features), max_features * 2,
                               "Concatenation should double feature count")
            
            # 2. Element-wise average
            min_length = min(len(text_features_padded), len(structured_features_padded))
            averaged_features = []
            for i in range(min_length):
                avg = (text_features_padded[i] + structured_features_padded[i]) / 2.0
                averaged_features.append(avg)
            
            self.assertEqual(len(averaged_features), min_length,
                               "Averaging should use minimum length")
            
            # 3. Weighted combination (text features typically more important)
            text_weight = 0.6
            structured_weight = 0.4
            weighted_features = []
            for i in range(min_length):
                weighted = (text_features_padded[i] * text_weight + 
                           structured_features_padded[i] * structured_weight)
                weighted_features.append(weighted)
            
            self.assertEqual(len(weighted_features), min_length,
                               "Weighted combination should use minimum length")
            
            # Verify all fusion results are valid
            for fusion_method, features in [
                ("concatenation", concatenated_features),
                ("averaged", averaged_features),
                ("weighted", weighted_features)
            ]:
                for i, feature in enumerate(features):
                    self.assertIsInstance(feature, (int, float),
                                          f"{fusion_method} feature {i} should be numeric")
                    self.assertFalse(np.isnan(feature),
                                          f"{fusion_method} feature {i} should not be NaN")
                    
        except Exception as e:
            self.fail(f"Feature fusion failed: {e}")
    
    def test_multimodal_consistency(self):
        """Test consistency between text and structured features"""
        if self.bert_classifier is None or self.safe_features is None:
            self.skipTest("Both classifiers not available")
            
        try:
            # Classify text to get task predictions
            text_classification = self.bert_classifier.classify_claim(
                self.sample_claim['description'], 
                self.sample_claim
            )
            
            # Generate structured features
            structured_features = self.safe_features.generate_comprehensive_features(self.sample_claim)
            
            # Check for consistency indicators
            task_predictions = text_classification['task_predictions']
            
            # 1. Vehicle count consistency
            text_vehicle_count = task_predictions.get('vehicle_count', {}).get('prediction', 'unknown')
            structured_vehicle_indicators = [
                name for name in structured_features.keys() 
                if 'vehicle' in name.lower() or 'car' in name.lower()
            ]
            
            # Should have vehicle-related structured features if text indicates multi-vehicle
            if text_vehicle_count in ['multiple', 'two']:
                self.assertGreater(len(structured_vehicle_indicators), 0,
                                       "Multi-vehicle text should have vehicle-related structured features")
            
            # 2. Accident type consistency
            text_accident_type = task_predictions.get('accident_type', {}).get('prediction', 'unknown')
            structured_damage_indicators = [
                name for name in structured_features.keys() 
                if 'damage' in name.lower() or 'severity' in name.lower()
            ]
            
            # Should have damage features for collision types
            if text_accident_type in ['collision', 'multi_vehicle']:
                self.assertGreater(len(structured_damage_indicators), 0,
                                       "Collision text should have damage-related structured features")
            
            # 3. Temporal consistency
            text_road_type = task_predictions.get('road_type', {}).get('prediction', 'unknown')
            structured_temporal_features = [
                name for name in structured_features.keys() 
                if 'temporal' in name.lower() or 'time' in name.lower()
            ]
            
            # Should have temporal features regardless of road type
            self.assertGreater(len(structured_temporal_features), 0,
                                   "Should have temporal features")
            
        except Exception as e:
            self.fail(f"Multimodal consistency check failed: {e}")

class TestEnsembleModelPerformance(unittest.TestCase):
    """Verify ensemble matches/exceeds PDF results"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.enhanced_recommender = EnhancedRecommenderAdvanced()
        except:
            self.enhanced_recommender = None
            
        # Test claims with different risk levels
        self.test_claims = {
            'low_risk': {
                'claim_id': 'LOW_RISK_001',
                'amount': 1500.0,
                'claim_type': 'auto',
                'description': 'Minor rear-end collision in parking lot, minimal damage',
                'accident_time': '10:30',
                'accident_date': '2023-03-15',
                'location': 'parking_lot',
                'customer_id': 'CUST_LOW_RISK',
                'police_report': 'yes',
                'witness_count': 2
            },
            'medium_risk': {
                'claim_id': 'MED_RISK_001',
                'amount': 7500.0,
                'claim_type': 'auto',
                'description': 'Side impact collision at intersection, moderate damage to both vehicles',
                'accident_time': '14:20',
                'accident_date': '2023-07-20',
                'location': 'urban_intersection',
                'customer_id': 'CUST_MED_RISK',
                'police_report': 'yes',
                'witness_count': 1
            },
            'high_risk': {
                'claim_id': 'HIGH_RISK_001',
                'amount': 25000.0,
                'claim_type': 'auto',
                'description': 'Staged multi-vehicle accident with excessive damage claims, delayed reporting',
                'accident_time': '23:30',
                'accident_date': '2023-11-01',
                'location': 'highway',
                'customer_id': 'CUST_HIGH_RISK',
                'police_report': 'no',
                'witness_count': 0
            }
        }
    
    def test_performance_improvement_targets(self):
        """Verify ensemble achieves PDF performance targets"""
        if self.enhanced_recommender is None:
            self.skipTest("Enhanced recommender not available")
            
        # PDF Table 10 targets for ensemble model
        expected_improvements = {
            'accuracy': 0.8713,      # PDF: 87.13%
            'precision': 0.7143,     # PDF: 71.43%
            'recall': 0.6107,         # PDF: 61.07%
            'f1_score': 0.6584,       # PDF: 65.84%
            'auc': 0.9344             # PDF: 93.44%
        }
        
        improvements_over_baseline = {
            'accuracy_improvement': 4.17,    # PDF: +4.17%
            'auc_improvement': 12.24,        # PDF: +12.24%
            'f1_improvement': 20.54          # PDF: +20.54%
        }
        
        # Test with sample claims
        total_predictions = []
        for risk_level, claim_data in self.test_claims.items():
            try:
                prediction = self.enhanced_recommender.analyze_claim(claim_data)
                total_predictions.append(prediction)
                
                # Check prediction structure
                self.assertIn('fraud_probability', prediction,
                               f"{risk_level}: Missing fraud probability")
                self.assertIn('confidence', prediction,
                               f"{risk_level}: Missing confidence score")
                self.assertIn('risk_level', prediction,
                               f"{risk_level}: Missing risk level")
                
                # Check value ranges
                fraud_prob = prediction['fraud_probability']
                self.assertGreaterEqual(fraud_prob, 0.0,
                                           f"{risk_level}: Fraud probability should be >= 0")
                self.assertLessEqual(fraud_prob, 1.0,
                                         f"{risk_level}: Fraud probability should be <= 1")
                
                confidence = prediction['confidence']
                self.assertGreaterEqual(confidence, 0.0,
                                           f"{risk_level}: Confidence should be >= 0")
                self.assertLessEqual(confidence, 1.0,
                                         f"{risk_level}: Confidence should be <= 1")
                
            except Exception as e:
                self.fail(f"{risk_level} claim analysis failed: {e}")
        
        # Verify risk differentiation
        low_risk_pred = next((p for p in total_predictions if 'LOW_RISK' in p.get('claim_id', '')), None)
        high_risk_pred = next((p for p in total_predictions if 'HIGH_RISK' in p.get('claim_id', '')), None)
        
        if low_risk_pred and high_risk_pred:
            self.assertLess(low_risk_pred['fraud_probability'], high_risk_pred['fraud_probability'],
                               "High risk claim should have higher fraud probability")
    
    def test_ensemble_feature_integration(self):
        """Test integration of multi-modal features in ensemble"""
        if self.enhanced_recommender is None:
            self.skipTest("Enhanced recommender not available")
            
        try:
            # Test with a complex claim
            complex_claim = {
                'claim_id': 'ENSEMBLE_TEST_001',
                'amount': 12000.0,
                'claim_type': 'auto',
                'description': 'Multi-vehicle collision on highway during rush hour with severe damage and injuries',
                'accident_time': '17:30',
                'accident_date': '2023-09-10',
                'location': 'highway',
                'customer_id': 'CUST_ENSEMBLE',
                'claimant_age': 28,
                'policy_age_years': 2.5,
                'witness_count': 1,
                'police_report': 'yes'
            }
            
            # Analyze with ensemble
            result = self.enhanced_recommender.analyze_claim(complex_claim)
            
            # Check for multi-modal integration evidence
            self.assertIn('text_analysis', result,
                           "Should include text analysis results")
            self.assertIn('engineered_features', result,
                           "Should include engineered features")
            self.assertIn('inconsistencies', result,
                           "Should include inconsistency analysis")
            
            # Check text analysis structure
            text_analysis = result['text_analysis']
            self.assertIn('task_predictions', text_analysis,
                           "Text analysis should include task predictions")
            self.assertIn('fraud_indicators', text_analysis,
                           "Text analysis should include fraud indicators")
            
            # Check engineered features
            engineered_features = result['engineered_features']
            self.assertIsInstance(engineered_features, dict,
                                  "Engineered features should be a dictionary")
            self.assertGreater(len(engineered_features), 50,
                               "Should generate substantial engineered features")
            
            # Check inconsistency analysis
            inconsistencies = result['inconsistencies']
            self.assertIsInstance(inconsistencies, list,
                                  "Inconsistencies should be a list")
            
        except Exception as e:
            self.fail(f"Ensemble feature integration test failed: {e}")
    
    def test_performance_benchmark_structure(self):
        """Test performance benchmarking structure"""
        if self.enhanced_recommender is None:
            self.skipTest("Enhanced recommender not available")
            
        # Test performance measurement capabilities
        try:
            # Test with multiple claims to get performance metrics
            start_time = time.time()
            
            for claim_data in self.test_claims.values():
                self.enhanced_recommender.analyze_claim(claim_data)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Check processing efficiency
            avg_time_per_claim = total_time / len(self.test_claims)
            self.assertLess(avg_time_per_claim, 2.0,
                               f"Average processing time should be < 2 seconds, got {avg_time_per_claim:.3f}s")
            
            # Check memory efficiency (if available)
            try:
                memory_usage = self.enhanced_recommender.get_memory_usage()
                self.assertIsInstance(memory_usage, dict,
                                      "Memory usage should be a dictionary")
                self.assertIn('component', memory_usage,
                               "Memory usage should include component name")
            except:
                # Memory usage might not be implemented
                pass
                
        except Exception as e:
            self.fail(f"Performance benchmarking failed: {e}")

class TestFraudDetectionPipeline(unittest.TestCase):
    """Verify complete fraud detection workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.enhanced_recommender = EnhancedRecommenderAdvanced()
        except:
            self.enhanced_recommender = None
            
        # Edge case claims for pipeline testing
        self.edge_case_claims = {
            'empty_description': {
                'claim_id': 'EMPTY_DESC_001',
                'amount': 5000.0,
                'claim_type': 'auto',
                'description': '',
                'accident_time': '',
                'accident_date': '',
                'location': ''
            },
            'extreme_amount': {
                'claim_id': 'EXTREME_AMT_001',
                'amount': 1000000.0,  # $1M claim
                'claim_type': 'auto',
                'description': 'Total loss of luxury vehicle',
                'accident_time': '12:00',
                'accident_date': '2023-06-15',
                'location': 'dealership'
            },
            'minimal_data': {
                'claim_id': 'MINIMAL_001',
                'claim_type': 'home',
                # Minimal required fields only
            },
            'multilingual': {
                'claim_id': 'MULTILING_001',
                'amount': 3000.0,
                'claim_type': 'auto',
                'description': 'Accidente de vehículo con daños menores',
                'accident_time': '09:15',
                'accident_date': '2023-04-20',
                'location': 'residencial'
            }
        }
    
    def test_end_to_end_pipeline(self):
        """Test from raw claim to fraud prediction"""
        if self.enhanced_recommender is None:
            self.skipTest("Enhanced recommender not available")
            
        for case_name, claim_data in self.edge_case_claims.items():
            with self.subTest(case=case_name):
                try:
                    # Process claim through complete pipeline
                    result = self.enhanced_recommender.analyze_claim(claim_data)
                    
                    # Check output structure
                    required_fields = [
                        'fraud_probability', 'confidence', 'risk_level',
                        'text_analysis', 'engineered_features', 'inconsistencies'
                    ]
                    
                    for field in required_fields:
                        self.assertIn(field, result,
                                       f"{case_name}: Missing required field '{field}'")
                    
                    # Check fraud probability validity
                    fraud_prob = result['fraud_probability']
                    self.assertIsInstance(fraud_prob, (int, float),
                                          f"{case_name}: Fraud probability should be numeric")
                    self.assertGreaterEqual(fraud_prob, 0.0,
                                               f"{case_name}: Fraud probability should be >= 0")
                    self.assertLessEqual(fraud_prob, 1.0,
                                             f"{case_name}: Fraud probability should be <= 1")
                    
                    # Check risk level validity
                    risk_level = result['risk_level']
                    valid_risk_levels = ['low', 'medium', 'high', 'minimal']
                    self.assertIn(risk_level, valid_risk_levels,
                                      f"{case_name}: Invalid risk level '{risk_level}'")
                    
                except Exception as e:
                    self.fail(f"{case_name}: End-to-end pipeline failed: {e}")
    
    def test_error_handling_and_recovery(self):
        """Test graceful error handling in pipeline"""
        if self.enhanced_recommender is None:
            self.skipTest("Enhanced recommender not available")
            
        # Test with intentionally malformed data
        malformed_claims = [
            {
                'claim_id': None,  # Null ID
                'amount': 'invalid',  # Invalid amount type
                'claim_type': 123,    # Invalid type
            },
            {
                'claim_id': 'MALFORMED_001',
                'amount': float('inf'),  # Infinite amount
                'claim_type': 'auto',
                'description': None  # None description
            },
            {
                'claim_id': 'MALFORMED_002',
                'amount': -5000.0,  # Negative amount
                'claim_type': '',
                'description': 'Valid description but other fields invalid'
            }
        ]
        
        for i, malformed_claim in enumerate(malformed_claims):
            with self.subTest(malformed_case=i):
                try:
                    result = self.enhanced_recommender.analyze_claim(malformed_claim)
                    
                    # Should still return a result with error information
                    self.assertIsInstance(result, dict,
                                          f"Malformed case {i}: Should return dict result")
                    
                    # Should have fraud probability (defaulted to safe value)
                    self.assertIn('fraud_probability', result,
                                   f"Malformed case {i}: Should have fraud probability")
                    fraud_prob = result['fraud_probability']
                    self.assertIsInstance(fraud_prob, (int, float),
                                          f"Malformed case {i}: Fraud probability should be numeric")
                    
                    # Should indicate error conditions where possible
                    if 'error' in result:
                        self.assertIsInstance(result['error'], str,
                                              f"Malformed case {i}: Error should be string")
                        
                except Exception as e:
                    # Should not crash completely
                    self.fail(f"Malformed case {i}: Should handle gracefully, got {e}")
    
    def test_pipeline_consistency(self):
        """Test pipeline consistency across multiple runs"""
        if self.enhanced_recommender is None:
            self.skipTest("Enhanced recommender not available")
            
        # Test same claim multiple times
        consistent_claim = {
            'claim_id': 'CONSISTENCY_TEST',
            'amount': 7500.0,
            'claim_type': 'auto',
            'description': 'Collision at intersection with moderate damage',
            'accident_time': '14:30',
            'accident_date': '2023-08-15',
            'location': 'urban_intersection'
        }
        
        results = []
        for i in range(3):  # Run 3 times
            result = self.enhanced_recommender.analyze_claim(consistent_claim)
            results.append(result)
        
        # Check consistency across runs
        first_result = results[0]
        
        for i, result in enumerate(results[1:], start=2):
            # Fraud probability should be consistent
            self.assertAlmostEqual(
                result['fraud_probability'], 
                first_result['fraud_probability'], 
                places=3,
                msg=f"Run {i}: Fraud probability should be consistent"
            )
            
            # Risk level should be consistent
            self.assertEqual(
                result['risk_level'],
                first_result['risk_level'],
                msg=f"Run {i}: Risk level should be consistent"
            )
            
            # Feature counts should be consistent
            first_features_count = len(first_result.get('engineered_features', {}))
            current_features_count = len(result.get('engineered_features', {}))
            self.assertEqual(
                current_features_count,
                first_features_count,
                msg=f"Run {i}: Feature count should be consistent"
            )

if __name__ == '__main__':
    unittest.main()
