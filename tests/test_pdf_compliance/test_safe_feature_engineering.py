"""
Test Suite for SAFE Feature Engineering PDF Compliance
Validates Smart Feature Engineering against AIML paper specifications
"""

import unittest
import numpy as np
import sys
import os
from typing import Dict, List, Any
import math

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

try:
    from enhanced_safe_features import EnhancedSAFE, get_enhanced_safe_features
    from safe_features import SAFE
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in mock mode for testing")

class TestSAFEFeatureGenerationCompliance(unittest.TestCase):
    """Validate SAFE implementation meets/exceeds PDF"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.safe_features = get_enhanced_safe_features(max_features=250, memory_limit_mb=150)
        except:
            self.safe_features = None
            
        # Sample claim data for testing
        self.sample_claim = {
            'claim_id': 'TEST_001',
            'amount': 7500.0,
            'claim_type': 'auto',
            'claim_subtype': 'collision',
            'description': 'Vehicle collision at intersection causing severe damage to front bumper',
            'accident_time': '14:30',
            'accident_date': '2023-06-15',
            'location': 'highway_intersection',
            'customer_id': 'CUST_001',
            'claimant_age': 35,
            'policy_number': 'POL123456',
            'policy_start_date': '2022-01-01',
            'coverage_amount': 25000.0,
            'weather_conditions': 'clear',
            'police_report': 'yes',
            'witness_count': 2,
            'third_party_involved': 'yes'
        }
    
    def test_feature_scaling_capability(self):
        """Verify 33 → 200-300 feature scaling"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        try:
            features = self.safe_features.generate_comprehensive_features(self.sample_claim)
            
            # Check feature count is in target range
            self.assertGreaterEqual(len(features), 200,
                                     f"Should generate at least 200 features, got {len(features)}")
            self.assertLessEqual(len(features), 300,
                                  f"Should generate at most 300 features, got {len(features)}")
            
            # Check all features are numeric
            for feature_name, feature_value in features.items():
                self.assertIsInstance(feature_value, (int, float),
                                      f"Feature '{feature_name}' should be numeric")
                self.assertFalse(np.isnan(feature_value),
                                  f"Feature '{feature_name}' should not be NaN")
                self.assertFalse(np.isinf(feature_value),
                                  f"Feature '{feature_name}' should not be infinite")
                
        except Exception as e:
            self.fail(f"Feature generation failed: {e}")
    
    def test_enhanced_base_features(self):
        """Verify enhanced base features vs PDF"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        expected_base_categories = {
            'temporal': 15,      # Enhanced from PDF
            'amount': 12,         # Enhanced from PDF  
            'frequency': 10,       # Enhanced from PDF
            'geographic': 8,       # Enhanced from PDF
            'policy': 8,          # Enhanced from PDF
            'claimant': 6,        # Enhanced from PDF
            'behavioral': 8,       # NEW - beyond PDF
            'consistency': 6,      # Enhanced from PDF
            'external_factors': 5    # NEW - beyond PDF
        }
        
        # Test each category
        features = self.safe_features.generate_comprehensive_features(self.sample_claim)
        
        for category, expected_count in expected_base_categories.items():
            # Map category names to actual feature prefixes
            category_mapping = {
                'geographic': ['geo', 'location', 'state'],
                'temporal': ['temporal', 'time', 'hour', 'day'],
                'amount': ['amount', 'payment', 'charge'],
                'frequency': ['frequency', 'count', 'rate'],
                'policy': ['policy', 'coverage'],
                'claimant': ['claimant', 'age', 'driver'],
                'behavioral': ['behavioral', 'pattern'],
                'consistency': ['consistency', 'diff'],
                'external_factors': ['external', 'weather']
            }

            # Count features in this category
            keywords = category_mapping.get(category, [category.split('_')[0]])
            category_features = [name for name in features.keys()
                             if any(keyword in name.lower() for keyword in keywords)]
            
            # Should have at least some features from each category
            self.assertGreater(len(category_features), 0,
                                   f"Should have features for category '{category}'")
            
            # Check specific feature patterns
            if category == 'temporal':
                temporal_keywords = ['temporal_', 'time_', 'hour_', 'day_', 'season_']
                temporal_features = [name for name in features.keys() 
                                 if any(kw in name.lower() for kw in temporal_keywords)]
                self.assertGreaterEqual(len(temporal_features), 10,
                                          f"Should have at least 10 temporal features")
                
            elif category == 'amount':
                amount_keywords = ['amount_', 'cost_', 'value_']
                amount_features = [name for name in features.keys() 
                                if any(kw in name.lower() for kw in amount_keywords)]
                self.assertGreaterEqual(len(amount_features), 8,
                                          f"Should have at least 8 amount features")
                
            elif category == 'behavioral':
                behavioral_keywords = ['behavioral_', 'urgency_', 'emotional_', 'legal_']
                behavioral_features = [name for name in features.keys() 
                                    if any(kw in name.lower() for kw in behavioral_keywords)]
                self.assertGreaterEqual(len(behavioral_features), 5,
                                          f"Should have at least 5 behavioral features")
    
    def test_smart_interactions(self):
        """Verify smart feature interactions implementation"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        # Test interaction generator
        interaction_generator = self.safe_features.interaction_generator
        
        # Check domain-important pairs exist
        self.assertIsNotNone(interaction_generator.domain_important_pairs,
                              "Domain-important pairs should be defined")
        self.assertGreater(len(interaction_generator.domain_important_pairs), 10,
                               "Should have at least 10 domain-important pairs")
        
        # Check interaction types are defined
        self.assertIn('multiplication', interaction_generator.interaction_types,
                       "Should have multiplication interaction type")
        self.assertIn('addition', interaction_generator.interaction_types,
                       "Should have addition interaction type")
        self.assertIn('division', interaction_generator.interaction_types,
                       "Should have division interaction type")
        
        # Test interaction generation
        base_features = {
            'amount': 5000.0,
            'risk_score': 0.7,
            'frequency_count': 2.0,
            'location_risk': 0.8
        }
        
        interactions = interaction_generator.generate_interactions(base_features)
        
        # Check interactions are generated
        self.assertGreater(len(interactions), 0,
                               "Should generate at least some interactions")
        
        # Check interaction naming convention
        for interaction_name in interactions.keys():
            self.assertIn('_x_', interaction_name,
                               f"Interaction '{interaction_name}' should follow naming convention")
            
        # Check interaction values are numeric
        for interaction_value in interactions.values():
            self.assertIsInstance(interaction_value, (int, float),
                                  "Interaction values should be numeric")
            self.assertFalse(np.isnan(interaction_value),
                                  "Interaction values should not be NaN")
    
    def test_mathematical_transformations(self):
        """Verify mathematical feature transformations"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        # Test mathematical transformer
        math_transformer = self.safe_features.math_transformer
        
        # Test with sample features
        test_features = {
            'amount': 1000.0,
            'risk_score': 0.5,
            'count': 10.0,
            'rate': 2.5
        }
        
        transformations = math_transformer.transform_features(test_features)
        
        # Check transformations are generated
        self.assertGreater(len(transformations), 0,
                               "Should generate mathematical transformations")
        
        # Check specific transformations
        expected_transforms = ['log_amount', 'sqrt_count', 'squared_risk_score']
        for expected_transform in expected_transforms:
            self.assertIn(expected_transform, transformations,
                           f"Should generate '{expected_transform}' transformation")
        
        # Verify transformation values
        if 'log_amount' in transformations:
            expected_log = math.log1p(1000.0)  # log(1001)
            actual_log = transformations['log_amount']
            self.assertAlmostEqual(actual_log, expected_log, places=5,
                                   msg="Log transformation incorrect")
        
        if 'sqrt_count' in transformations:
            expected_sqrt = math.sqrt(10.0)
            actual_sqrt = transformations['sqrt_count']
            self.assertAlmostEqual(actual_sqrt, expected_sqrt, places=5,
                                   msg="Square root transformation incorrect")
    
    def test_feature_importance_scoring(self):
        """Verify domain-knowledge based importance scoring"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        # Test importance calculation
        test_features = {
            'amount': 5000.0,        # High importance
            'risk_score': 0.8,        # High importance
            'frequency': 2.0,          # Medium importance
            'normalized_feature': 0.5,   # Low importance
            'amount_x_risk_score': 4000.0,  # Interaction - high importance
            'log_amount': 8.52,          # Transformation - medium importance
        }
        
        for feature_name, feature_value in test_features.items():
            importance = self.safe_features._calculate_feature_importance(feature_name, feature_value)
            
            # Check importance range
            self.assertGreaterEqual(importance, 0.0,
                                       f"Importance for '{feature_name}' should be >= 0")
            self.assertLessEqual(importance, 1.0,
                                  f"Importance for '{feature_name}' should be <= 1")
            
            # Check domain knowledge weighting
            if 'amount' in feature_name.lower() or 'risk' in feature_name.lower():
                self.assertGreaterEqual(importance, 0.7,
                                           f"Amount/risk features should have high importance")
            
            if '_x_' in feature_name:  # Interaction feature
                self.assertGreaterEqual(importance, 0.8,
                                           f"Interaction features should have high importance")
            
            if 'normalized' in feature_name.lower():
                self.assertLessEqual(importance, 0.5,
                                      f"Normalized features should have lower importance")
    
    def test_optimal_feature_selection(self):
        """Verify memory-constrained feature selection"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        # Create many features to test selection
        many_features = {}
        for i in range(500):  # More than max_features
            many_features[f'feature_{i}'] = np.random.random()
        
        # Add some high-importance features
        many_features['amount'] = 10000.0
        many_features['risk_score'] = 0.9
        many_features['amount_x_risk_score'] = 9000.0
        
        selected_features = self.safe_features._select_optimal_features(many_features)
        
        # Check selection respects max_features limit
        self.assertLessEqual(len(selected_features), self.safe_features.max_features,
                                  f"Should select at most {self.safe_features.max_features} features")
        
        # Check high-importance features are selected
        self.assertIn('amount', selected_features,
                       "High-importance 'amount' feature should be selected")
        self.assertIn('risk_score', selected_features,
                       "High-importance 'risk_score' feature should be selected")
        self.assertIn('amount_x_risk_score', selected_features,
                       "High-importance interaction feature should be selected")
        
        # Check feature diversity
        feature_categories = set()
        for feature_name in selected_features.keys():
            if 'amount' in feature_name:
                feature_categories.add('amount')
            elif 'risk' in feature_name:
                feature_categories.add('risk')
            elif 'temporal' in feature_name:
                feature_categories.add('temporal')
            elif 'geo' in feature_name:
                feature_categories.add('geographic')
        
        self.assertGreater(len(feature_categories), 1,
                               "Should select features from multiple categories")

class TestFeatureQuality(unittest.TestCase):
    """Validate feature quality and diversity"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            self.safe_features = get_enhanced_safe_features(max_features=250, memory_limit_mb=150)

            # Sample claim data for testing
            self.sample_claim = {
                'Claim_Report_Number': 'TEST-001',
                'Date_Of_Loss': '2023-01-15',
                'Time_Of_Loss': '14:30',
                'Claim_Type': 'Auto',
                'Claim_Total_Payments': 5000.0,
                'Estimate_Total_Charges': 4500.0,
                'Length_Of_Loss': '2 days',
                'Days_Between_Loss_And_Report': 3,
                'Was_Police_Report_Filed': True,
                'Witness_Count': 2,
                'Claimant_Age': 35,
                'Annual_Miles': 12000,
                'Vehicle_Make': 'Toyota',
                'Vehicle_Model': 'Camry',
                'Vehicle_Year': 2020,
                'Policy_State': 'CA',
                'Incident_State': 'CA',
                'Channel_Type': 'Agent',
                'Loss_Description': 'Car rear-ended at stop light',
                'High_Prime_Indicator': True
            }
        except:
            self.safe_features = None
            self.sample_claim = {}
    
    def test_feature_name_normalization(self):
        """Verify ML-friendly feature naming"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        # Test with problematic feature names
        test_features = {
            'amount!@#': 1000.0,
            'risk-score': 0.8,
            'time of day': 14.5,
            'location risk': 0.7,
            '123feature': 42.0,
            'feature with spaces and symbols!@#$%': 1.0
        }
        
        normalized = self.safe_features._normalize_feature_names(test_features)
        
        # Check all names are ML-friendly
        for feature_name in normalized.keys():
            # Should only contain alphanumeric and underscores
            self.assertTrue(all(c.isalnum() or c == '_' for c in feature_name),
                               f"Feature '{feature_name}' should only contain alphanumeric and underscores")
            
            # Should not start with number
            self.assertTrue(feature_name[0].isalpha() or feature_name[0] == '_',
                               f"Feature '{feature_name}' should start with letter or underscore")
            
            # Should not have consecutive underscores
            self.assertNotIn('__', feature_name,
                                f"Feature '{feature_name}' should not have consecutive underscores")
        
        # Check values are preserved
        self.assertEqual(normalized['feature_with_spaces_and_symbols'], 1.0,
                          "Values should be preserved during normalization")
    
    def test_feature_diversity_analysis(self):
        """Analyze feature diversity against PDF standards"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        sample_claim = {
            'claim_id': 'DIVERSITY_TEST',
            'amount': 12000.0,
            'claim_type': 'auto',
            'description': 'Multi-vehicle accident on highway during rush hour with severe damage',
            'accident_time': '18:45',
            'accident_date': '2023-12-20',
            'location': 'urban_intersection',
            'customer_id': 'CUST_DIVERSE'
        }
        
        features = self.safe_features.generate_comprehensive_features(sample_claim)
        
        # Analyze feature diversity
        feature_patterns = {
            'temporal': ['temporal_', 'time_', 'hour_', 'day_', 'season_', 'month_'],
            'amount': ['amount_', 'cost_', 'value_', 'log_', 'sqrt_'],
            'interaction': ['_x_'],
            'geographic': ['geo_', 'location_', 'urban_', 'rural_'],
            'behavioral': ['behavioral_', 'urgency_', 'emotional_', 'legal_'],
            'policy': ['policy_', 'coverage_', 'age_years_'],
            'consistency': ['consistency_', 'delay_', 'reporting_'],
            'external': ['external_', 'weather_', 'police_', 'witness_']
        }
        
        diversity_results = {}
        for category, patterns in feature_patterns.items():
            matching_features = [name for name in features.keys() 
                             if any(pattern in name.lower() for pattern in patterns)]
            diversity_results[category] = len(matching_features)
        
        # Should have features from multiple categories
        non_empty_categories = [cat for cat, count in diversity_results.items() if count > 0]
        self.assertGreater(len(non_empty_categories), 5,
                               f"Should have features from at least 5 categories, got {len(non_empty_categories)}")
        
        # Should have sufficient features from key categories
        self.assertGreaterEqual(diversity_results.get('temporal', 0), 8,
                                      "Should have at least 8 temporal features")
        self.assertGreaterEqual(diversity_results.get('amount', 0), 6,
                                      "Should have at least 6 amount features")
        self.assertGreaterEqual(diversity_results.get('interaction', 0), 20,
                                      "Should have at least 20 interaction features")
    
    def test_memory_efficiency_compliance(self):
        """Verify memory constraint compliance"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        # Test memory limit enforcement
        original_limit = self.safe_features.memory_limit_mb
        
        # Generate features and check memory usage
        features = self.safe_features.generate_comprehensive_features(self.sample_claim)
        
        # Check memory usage report
        memory_info = self.safe_features.get_memory_usage()
        
        self.assertIn('memory_limit_mb', memory_info,
                       "Memory info should include limit")
        self.assertIn('features_generated', memory_info,
                       "Memory info should include feature count")
        self.assertEqual(memory_info['memory_limit_mb'], original_limit,
                          "Memory limit should be preserved")
        
        # Check system memory usage is reasonable
        system_memory = memory_info.get('system_memory', {})
        if 'allocated_mb' in system_memory:
            self.assertLessEqual(system_memory['allocated_mb'], original_limit,
                                    "Should not exceed memory limit")

class TestSAFEPerformance(unittest.TestCase):
    """Benchmark SAFE against PDF performance improvements"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.safe_features = get_enhanced_safe_features(max_features=250, memory_limit_mb=150)
        except:
            self.safe_features = None
    
    def test_generation_efficiency(self):
        """Verify efficient feature generation"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        import time
        
        sample_claim = {
            'claim_id': 'EFFICIENCY_TEST',
            'amount': 5000.0,
            'claim_type': 'auto',
            'description': 'Test claim for efficiency testing'
        }
        
        # Time feature generation
        start_time = time.time()
        features = self.safe_features.generate_comprehensive_features(sample_claim)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should generate features efficiently (under 1 second)
        self.assertLess(generation_time, 1.0,
                              f"Feature generation should take < 1 second, took {generation_time:.3f}s")
        
        # Should generate reasonable number of features
        self.assertGreater(len(features), 100,
                               "Should generate substantial number of features")
        self.assertLess(len(features), 300,
                              "Should not exceed maximum feature limit")
    
    def test_robustness_to_missing_data(self):
        """Test handling of incomplete claim data"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        # Test cases with missing data
        incomplete_claims = [
            # Missing amount
            {
                'claim_id': 'MISSING_AMOUNT',
                'description': 'Collision with missing amount data',
                'claim_type': 'auto'
            },
            # Missing description
            {
                'claim_id': 'MISSING_DESC',
                'amount': 3000.0,
                'claim_type': 'auto'
            },
            # Minimal data
            {
                'claim_id': 'MINIMAL_DATA',
                'claim_type': 'home'
            },
            # Empty values
            {
                'claim_id': 'EMPTY_VALUES',
                'amount': 0.0,
                'description': '',
                'accident_time': '',
                'location': ''
            }
        ]
        
        for i, claim_data in enumerate(incomplete_claims):
            with self.subTest(case=i):
                try:
                    features = self.safe_features.generate_comprehensive_features(claim_data)
                    
                    # Should still generate some features
                    self.assertGreater(len(features), 0,
                                           f"Case {i}: Should generate features even with missing data")
                    
                    # All features should be valid numbers
                    for feature_name, feature_value in features.items():
                        self.assertIsInstance(feature_value, (int, float),
                                              f"Case {i}: Feature '{feature_name}' should be numeric")
                        self.assertFalse(np.isnan(feature_value),
                                              f"Case {i}: Feature '{feature_name}' should not be NaN")
                        
                except Exception as e:
                    self.fail(f"Case {i}: Should handle missing data gracefully: {e}")
    
    def test_scalability_with_feature_limits(self):
        """Test behavior at different feature limits"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        test_limits = [50, 100, 200, 300]
        
        for limit in test_limits:
            with self.subTest(limit=limit):
                # Create SAFE instance with specific limit
                safe_limited = EnhancedSAFE(max_features=limit, memory_limit_mb=100)
                
                sample_claim = {
                    'claim_id': f'SCALE_TEST_{limit}',
                    'amount': 7500.0,
                    'description': 'Test claim for scalability testing'
                }
                
                features = safe_limited.generate_comprehensive_features(sample_claim)
                
                # Should respect the limit
                self.assertLessEqual(len(features), limit,
                                      f"Limit {limit}: Should not exceed feature limit")
                
                # Should generate meaningful features even at low limits
                if limit >= 50:
                    self.assertGreater(len(features), limit * 0.5,
                                          f"Limit {limit}: Should generate at least 50% of limit")
    
    def test_feature_importance_report(self):
        """Test feature importance reporting functionality"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        report = self.safe_features.get_feature_importance_report()
        
        # Check report structure
        self.assertIn('total_features_generated', report,
                       "Report should include total features generated")
        self.assertIn('max_features_allowed', report,
                       "Report should include max features allowed")
        self.assertIn('feature_categories', report,
                       "Report should include feature categories")
        
        # Check feature categories
        categories = report['feature_categories']
        expected_categories = [
            'temporal', 'amount', 'frequency', 'geographic', 
            'policy', 'claimant', 'behavioral', 'consistency', 'external_factors'
        ]
        
        for category in expected_categories:
            self.assertIn(category, categories,
                           f"Report should include {category} category")
        
        # Check reasonable values
        self.assertGreaterEqual(report['max_features_allowed'], 100,
                                "Max features should be reasonable")
        self.assertLessEqual(report['max_features_allowed'], 500,
                                "Max features should be reasonable")

class TestPDFComplianceComparison(unittest.TestCase):
    """Compare SAFE implementation against PDF specifications"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.safe_features = get_enhanced_safe_features(max_features=250, memory_limit_mb=150)
        except:
            self.safe_features = None
    
    def test_pdf_feature_count_compliance(self):
        """Verify feature count meets PDF improvement targets"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        # PDF: 216 original variables → 1,155 features with SAFE
        # Our implementation should achieve similar or better ratios
        
        sample_claim = {
            'claim_id': 'PDF_COMPARISON',
            'amount': 8500.0,
            'claim_type': 'auto',
            'description': 'Multi-vehicle collision with comprehensive damage assessment',
            'accident_time': '16:20',
            'accident_date': '2023-08-10',
            'location': 'highway',
            'customer_id': 'CUST_PDF'
        }
        
        features = self.safe_features.generate_comprehensive_features(sample_claim)
        
        # Should achieve significant feature expansion
        feature_expansion_ratio = len(features) / 10  # Assume ~10 base fields
        
        # PDF achieved 5.35x expansion (1155/216)
        # We should achieve similar or better
        self.assertGreater(feature_expansion_ratio, 3.0,
                               f"Should achieve at least 3x feature expansion, got {feature_expansion_ratio:.2f}x")
    
    def test_enhanced_capabilities_beyond_pdf(self):
        """Verify enhancements beyond PDF specifications"""
        if self.safe_features is None:
            self.skipTest("SAFE features not available")
            
        features = self.safe_features.generate_comprehensive_features(self.sample_claim)
        
        # Check for PDF+ capabilities
        pdf_plus_features = {
            'behavioral_features': any('behavioral_' in f for f in features.keys()),
            'enhanced_consistency': any('consistency_' in f for f in features.keys()),
            'external_factors': any('external_' in f for f in features.keys()),
            'smart_interactions': any('_x_' in f for f in features.keys()),
            'mathematical_transforms': any(f.startswith(('log_', 'sqrt_', 'squared_')) 
                                         for f in features.keys())
        }
        
        for capability, present in pdf_plus_features.items():
            self.assertTrue(present,
                              f"Should implement PDF+ capability: {capability}")
    
    def setUp(self):
        """Set up test fixtures for PDF comparison"""
        try:
            self.safe_features = get_enhanced_safe_features(max_features=250, memory_limit_mb=150)
        except:
            self.safe_features = None
            
        # Standard sample claim for comparison tests
        self.sample_claim = {
            'claim_id': 'PDF_COMPARISON_001',
            'amount': 7500.0,
            'claim_type': 'auto',
            'claim_subtype': 'collision',
            'description': 'Vehicle collision at intersection causing severe damage to front bumper',
            'accident_time': '14:30',
            'accident_date': '2023-06-15',
            'location': 'highway_intersection',
            'customer_id': 'CUST_001',
            'claimant_age': 35,
            'policy_number': 'POL123456',
            'policy_start_date': '2022-01-01',
            'coverage_amount': 25000.0,
            'weather_conditions': 'clear',
            'police_report': 'yes',
            'witness_count': 2,
            'third_party_involved': 'yes'
        }

if __name__ == '__main__':
    unittest.main()
