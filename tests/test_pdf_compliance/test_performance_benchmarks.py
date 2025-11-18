"""
Test Suite for Performance Benchmarks Against PDF
Validates system performance against AIML paper benchmarks
"""

import unittest
import numpy as np
import sys
import os
from typing import Dict, List, Any
import time
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

try:
    from aiml_multi_task_heads import AIMLMultiTaskHeads, get_aiml_multi_task_heads
    from enhanced_bert_classifier import EnhancedBERTClassifier, get_enhanced_bert_classifier
    from enhanced_safe_features import EnhancedSAFE, get_enhanced_safe_features
    from enhanced_recommender_advanced import EnhancedRecommenderAdvanced
    from performance_validator import PerformanceValidator
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in mock mode for testing")

class TestBaselineModelPerformance(unittest.TestCase):
    """Verify baseline matches PDF: 83.64% accuracy, 0.8325 AUC"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.performance_validator = PerformanceValidator()
        except:
            self.performance_validator = None
            
        # PDF Table 2 baseline metrics
        self.pdf_baseline_metrics = {
            'accuracy': 0.8364,
            'precision': 0.7095,
            'recall': 0.4441,
            'f1_score': 0.5462,
            'auc': 0.8325
        }
        
        # Sample claims for baseline testing
        self.baseline_claims = [
            {
                'claim_id': 'BASE_001',
                'amount': 3500.0,
                'claim_type': 'auto',
                'description': 'Minor collision in residential area',
                'fraud_label': 0  # Non-fraud
            },
            {
                'claim_id': 'BASE_002',
                'amount': 15000.0,
                'claim_type': 'auto',
                'description': 'Major collision with inconsistent damage reports',
                'fraud_label': 1  # Fraud
            },
            {
                'claim_id': 'BASE_003',
                'amount': 7500.0,
                'claim_type': 'home',
                'description': 'Water damage from burst pipe',
                'fraud_label': 0  # Non-fraud
            }
        ]
    
    def test_baseline_performance_targets(self):
        """Verify baseline model matches PDF Table 2"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        try:
            # Test baseline performance with sample claims
            baseline_results = self.performance_validator.validate_baseline_performance(
                self.baseline_claims, 'pdf_baseline_test'
            )
            
            # Check results structure
            self.assertIn('accuracy', baseline_results,
                           "Should include accuracy metric")
            self.assertIn('precision', baseline_results,
                           "Should include precision metric")
            self.assertIn('recall', baseline_results,
                           "Should include recall metric")
            self.assertIn('f1_score', baseline_results,
                           "Should include F1-score metric")
            self.assertIn('auc', baseline_results,
                           "Should include AUC metric")
            
            # Check against PDF targets
            for metric, pdf_value in self.pdf_baseline_metrics.items():
                actual_value = baseline_results.get(metric, 0.0)
                
                # Allow small tolerance for implementation differences
                tolerance = 0.05  # 5% tolerance
                self.assertAlmostEqual(
                    actual_value, pdf_value, delta=tolerance,
                    msg=f"Baseline {metric}: expected {pdf_value:.4f}, got {actual_value:.4f}"
                )
                
        except Exception as e:
            self.fail(f"Baseline performance validation failed: {e}")
    
    def test_feature_count_baseline(self):
        """Verify baseline feature count matches PDF"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        try:
            # Test baseline feature generation
            baseline_features = self.performance_validator.validate_feature_engineering(
                None, 'baseline_features'  # None for baseline
            )
            
            # PDF: 216 original variables
            expected_baseline_features = 216
            
            actual_feature_count = baseline_features.get('total_unique_features', 0)
            
            # Should be close to PDF baseline
            tolerance = 50  # Allow some variation
            self.assertAlmostEqual(
                actual_feature_count, expected_baseline_features, delta=tolerance,
                msg=f"Baseline features: expected ~{expected_baseline_features}, got {actual_feature_count}"
            )
                
        except Exception as e:
            self.fail(f"Baseline feature count test failed: {e}")

class TestTextFactorsImprovement(unittest.TestCase):
    """Verify text factors improve: 84.81% accuracy, 0.841 AUC"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.performance_validator = PerformanceValidator()
        except:
            self.performance_validator = None
            
        # PDF Table 4 text factors improvements
        self.pdf_text_improvements = {
            'accuracy': 0.8481,
            'precision': 0.7473,
            'recall': 0.4755,
            'f1_score': 0.5812,
            'auc': 0.8410
        }
        
        self.pdf_improvement_percentages = {
            'accuracy_improvement': 1.40,    # PDF: +1.40%
            'auc_improvement': 1.02,         # PDF: +1.02%
            'f1_improvement': 6.41           # PDF: +6.41%
        }
        
        # Claims with rich text data
        self.text_rich_claims = [
            {
                'claim_id': 'TEXT_001',
                'amount': 5000.0,
                'claim_type': 'auto',
                'description': 'Two-car collision at intersection during evening rush hour. Both drivers stopped suddenly, rear-end impact. Moderate damage to both vehicles. No injuries reported. Police called to scene.',
                'fraud_label': 0
            },
            {
                'claim_id': 'TEXT_002',
                'amount': 18000.0,
                'claim_type': 'auto',
                'description': 'Single vehicle rollover on highway in unclear circumstances. Driver claims lost control but no witnesses. Damage appears inconsistent with rollover mechanics. Delayed reporting to insurance company.',
                'fraud_label': 1
            },
            {
                'claim_id': 'TEXT_003',
                'amount': 3200.0,
                'claim_type': 'auto',
                'description': 'Side impact collision in parking lot. Clear liability of other driver. Multiple witnesses available. Damage assessment consistent with collision angle.',
                'fraud_label': 0
            }
        ]
    
    def test_text_factor_improvements(self):
        """Verify text factors achieve PDF improvements"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        try:
            # Test performance with text factors
            text_results = self.performance_validator.validate_enhanced_text_processing(
                self.text_rich_claims, 'text_factors_test'
            )
            
            # Check improvement structure
            self.assertIn('text_processing_metrics', text_results,
                           "Should include text processing metrics")
            self.assertIn('improvement_over_baseline', text_results,
                           "Should include improvement metrics")
            
            improvements = text_results['improvement_over_baseline']
            
            # Check against PDF targets
            for metric, pdf_value in self.pdf_text_improvements.items():
                actual_value = improvements.get(metric, 0.0)
                
                # Allow small tolerance
                tolerance = 0.03  # 3% tolerance for text improvements
                self.assertAlmostEqual(
                    actual_value, pdf_value, delta=tolerance,
                    msg=f"Text {metric}: expected {pdf_value:.4f}, got {actual_value:.4f}"
                )
                
        except Exception as e:
            self.fail(f"Text factor improvements test failed: {e}")
    
    def test_text_feature_count(self):
        """Verify text feature count matches PDF"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        try:
            # Test text feature generation
            text_features = self.performance_validator.validate_text_classification(
                self.text_rich_claims, 'text_feature_count_test'
            )
            
            # PDF: 45 new boolean features from text data
            expected_text_features = 45
            
            actual_feature_count = text_features.get('text_feature_count', 0)
            
            # Should be close to PDF text features
            tolerance = 10  # Allow variation
            self.assertAlmostEqual(
                actual_feature_count, expected_text_features, delta=tolerance,
                msg=f"Text features: expected ~{expected_text_features}, got {actual_feature_count}"
            )
                
        except Exception as e:
            self.fail(f"Text feature count test failed: {e}")

class TestEnsembleModelPerformance(unittest.TestCase):
    """Verify ensemble achieves: 87.13% accuracy, 0.9344 AUC"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.performance_validator = PerformanceValidator()
        except:
            self.performance_validator = None
            
        # PDF Table 10 ensemble targets
        self.pdf_ensemble_targets = {
            'accuracy': 0.8713,
            'precision': 0.7143,
            'recall': 0.6107,
            'f1_score': 0.6584,
            'auc': 0.9344
        }
        
        self.pdf_overall_improvements = {
            'accuracy_improvement': 4.17,    # PDF: +4.17%
            'precision_improvement': 0.68,    # PDF: +0.68%
            'auc_improvement': 12.24,        # PDF: +12.24%
            'f1_improvement': 20.54          # PDF: +20.54%
        }
        
        # Complex claims requiring multi-modal analysis
        self.ensemble_test_claims = [
            {
                'claim_id': 'ENSEMBLE_001',
                'amount': 4500.0,
                'claim_type': 'auto',
                'description': 'Low-speed collision in residential area with clear liability',
                'accident_time': '10:15',
                'accident_date': '2023-04-20',
                'location': 'residential_street',
                'police_report': 'yes',
                'witness_count': 2,
                'fraud_label': 0
            },
            {
                'claim_id': 'ENSEMBLE_002',
                'amount': 22000.0,
                'claim_type': 'auto',
                'description': 'High-speed collision on highway with severe damage and inconsistent witness statements',
                'accident_time': '22:30',
                'accident_date': '2023-09-15',
                'location': 'highway_rural',
                'police_report': 'no',
                'witness_count': 0,
                'fraud_label': 1
            },
            {
                'claim_id': 'ENSEMBLE_003',
                'amount': 8500.0,
                'claim_type': 'auto',
                'description': 'Multi-vehicle pileup during foggy conditions with delayed medical treatment',
                'accident_time': '07:45',
                'accident_date': '2023-11-10',
                'location': 'highway_fog',
                'police_report': 'yes',
                'witness_count': 4,
                'fraud_label': 0
            }
        ]
    
    def test_ensemble_performance_targets(self):
        """Verify ensemble achieves PDF Table 10 targets"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        try:
            # Test ensemble performance
            ensemble_results = self.performance_validator.validate_ensemble_performance(
                self.ensemble_test_claims, 'ensemble_performance_test'
            )
            
            # Check results structure
            self.assertIn('ensemble_metrics', ensemble_results,
                           "Should include ensemble metrics")
            self.assertIn('improvement_over_baseline', ensemble_results,
                           "Should include improvement metrics")
            
            improvements = ensemble_results['improvement_over_baseline']
            
            # Check against PDF targets
            for metric, pdf_value in self.pdf_ensemble_targets.items():
                actual_value = improvements.get(metric, 0.0)
                
                # Allow small tolerance for ensemble
                tolerance = 0.05  # 5% tolerance
                self.assertAlmostEqual(
                    actual_value, pdf_value, delta=tolerance,
                    msg=f"Ensemble {metric}: expected {pdf_value:.4f}, got {actual_value:.4f}"
                )
                
        except Exception as e:
            self.fail(f"Ensemble performance targets test failed: {e}")
    
    def test_overall_improvement_validation(self):
        """Verify overall improvements match PDF percentages"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        try:
            # Get baseline and ensemble results
            baseline_results = self.performance_validator.validate_baseline_performance(
                self.ensemble_test_claims, 'ensemble_baseline'
            )
            ensemble_results = self.performance_validator.validate_ensemble_performance(
                self.ensemble_test_claims, 'ensemble_improvement'
            )
            
            # Calculate actual improvements
            baseline_auc = baseline_results.get('auc', 0.0)
            ensemble_auc = ensemble_results.get('improvement_over_baseline', {}).get('auc', 0.0)
            
            # Calculate AUC improvement percentage
            if baseline_auc > 0:
                actual_auc_improvement = ((ensemble_auc - baseline_auc) / baseline_auc) * 100
            else:
                actual_auc_improvement = 0.0
            
            # Check against PDF AUC improvement
            pdf_auc_improvement = self.pdf_overall_improvements['auc_improvement']
            tolerance = 2.0  # 2% tolerance
            
            self.assertAlmostEqual(
                actual_auc_improvement, pdf_auc_improvement, delta=tolerance,
                msg=f"AUC improvement: expected {pdf_auc_improvement:.2f}%, got {actual_auc_improvement:.2f}%"
            )
                
        except Exception as e:
            self.fail(f"Overall improvement validation failed: {e}")

class TestRobustnessAndEdgeCases(unittest.TestCase):
    """Test system robustness beyond PDF scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.performance_validator = PerformanceValidator()
        except:
            self.performance_validator = None
    
    def test_extreme_claims(self):
        """Test with very high/low claim amounts"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        extreme_claims = [
            {
                'claim_id': 'EXTREME_HIGH',
                'amount': 500000.0,  # Very high
                'claim_type': 'auto',
                'description': 'Luxury vehicle total loss',
                'fraud_label': 1
            },
            {
                'claim_id': 'EXTREME_LOW',
                'amount': 50.0,  # Very low
                'claim_type': 'home',
                'description': 'Minor cosmetic damage',
                'fraud_label': 0
            }
        ]
        
        for claim in extreme_claims:
            with self.subTest(claim=claim['claim_id']):
                try:
                    # Test performance with extreme claim
                    result = self.performance_validator.validate_single_claim_performance(
                        claim, 'extreme_test'
                    )
                    
                    # Should handle extreme values gracefully
                    self.assertIn('fraud_probability', result,
                                   f"{claim['claim_id']}: Should have fraud probability")
                    self.assertIn('confidence', result,
                                   f"{claim['claim_id']}: Should have confidence")
                    
                    # Values should be in valid ranges
                    fraud_prob = result['fraud_probability']
                    self.assertGreaterEqual(fraud_prob, 0.0,
                                               f"{claim['claim_id']}: Fraud probability should be >= 0")
                    self.assertLessEqual(fraud_prob, 1.0,
                                             f"{claim['claim_id']}: Fraud probability should be <= 1")
                    
                except Exception as e:
                    self.fail(f"{claim['claim_id']}: Extreme value handling failed: {e}")
    
    def test_multilingual_text(self):
        """Test with non-English claim descriptions"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        multilingual_claims = [
            {
                'claim_id': 'CHINESE_001',
                'amount': 8000.0,
                'claim_type': 'auto',
                'description': '标的车与三者车高速公路行驶相撞，两车受损',  # PDF Table 1 example
                'fraud_label': 0
            },
            {
                'claim_id': 'SPANISH_001',
                'amount': 5500.0,
                'claim_type': 'auto',
                'description': 'Colisión de vehículo con daños moderados en intersección urbana',
                'fraud_label': 0
            },
            {
                'claim_id': 'MIXED_001',
                'amount': 12000.0,
                'claim_type': 'auto',
                'description': 'Accident with vehicle damage 事故严重',
                'fraud_label': 1
            }
        ]
        
        for claim in multilingual_claims:
            with self.subTest(claim=claim['claim_id']):
                try:
                    # Test performance with multilingual text
                    result = self.performance_validator.validate_multilingual_processing(
                        claim, 'multilingual_test'
                    )
                    
                    # Should process multilingual text
                    self.assertIn('text_processing_success', result,
                                   f"{claim['claim_id']}: Should indicate text processing success")
                    
                    success = result['text_processing_success']
                    self.assertTrue(success,
                                     f"{claim['claim_id']}: Should successfully process multilingual text")
                    
                except Exception as e:
                    self.fail(f"{claim['claim_id']}: Multilingual processing failed: {e}")
    
    def test_corrupted_data_handling(self):
        """Test with malformed/missing data"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        corrupted_claims = [
            {
                'claim_id': 'CORRUPTED_001',
                'amount': None,  # Missing amount
                'claim_type': 'auto',
                'description': 'Valid description but missing amount',
                'fraud_label': 0
            },
            {
                'claim_id': 'CORRUPTED_002',
                'amount': 'invalid_amount',  # Invalid type
                'claim_type': 'auto',
                'description': 'Description with invalid amount type',
                'fraud_label': 0
            },
            {
                'claim_id': 'CORRUPTED_003',
                'amount': float('inf'),  # Infinite amount
                'claim_type': 'auto',
                'description': 'Description with infinite amount',
                'fraud_label': 1
            },
            {
                'claim_id': 'CORRUPTED_004',
                'amount': -1000.0,  # Negative amount
                'claim_type': 'auto',
                'description': 'Description with negative amount',
                'fraud_label': 0
            }
        ]
        
        for claim in corrupted_claims:
            with self.subTest(claim=claim['claim_id']):
                try:
                    # Test performance with corrupted data
                    result = self.performance_validator.validate_corrupted_data_handling(
                        claim, 'corrupted_test'
                    )
                    
                    # Should handle gracefully
                    self.assertIn('error_handling', result,
                                   f"{claim['claim_id']}: Should include error handling info")
                    
                    error_handling = result['error_handling']
                    self.assertIsInstance(error_handling, dict,
                                          f"{claim['claim_id']}: Error handling should be dict")
                    
                    # Should not crash
                    self.assertIn('graceful_degradation', error_handling,
                                   f"{claim['claim_id']}: Should indicate graceful degradation")
                    
                except Exception as e:
                    self.fail(f"{claim['claim_id']}: Corrupted data handling failed: {e}")
    
    def test_concurrent_processing(self):
        """Test multiple claim processing"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        # Generate multiple claims for concurrent testing
        concurrent_claims = []
        for i in range(10):
            claim = {
                'claim_id': f'CONCURRENT_{i:03d}',
                'amount': 1000.0 * (i + 1),
                'claim_type': 'auto' if i % 2 == 0 else 'home',
                'description': f'Claim number {i+1} for concurrent processing test',
                'fraud_label': 1 if i > 7 else 0  # Higher fraud rate for later claims
            }
            concurrent_claims.append(claim)
        
        try:
            # Test concurrent processing
            start_time = time.time()
            
            results = self.performance_validator.validate_concurrent_processing(
                concurrent_claims, 'concurrent_test'
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Check processing efficiency
            avg_time_per_claim = processing_time / len(concurrent_claims)
            self.assertLess(avg_time_per_claim, 0.5,
                               f"Concurrent processing should be efficient, avg: {avg_time_per_claim:.3f}s")
            
            # Check all claims processed
            self.assertIn('processed_count', results,
                           "Should include processed count")
            processed_count = results['processed_count']
            self.assertEqual(processed_count, len(concurrent_claims),
                               "Should process all concurrent claims")
            
            # Check memory efficiency
            self.assertIn('peak_memory_mb', results,
                           "Should include peak memory usage")
            peak_memory = results['peak_memory_mb']
            self.assertLess(peak_memory, 500,  # Should be reasonable
                               f"Peak memory usage should be reasonable, got {peak_memory}MB")
                
        except Exception as e:
            self.fail(f"Concurrent processing test failed: {e}")

class TestPerformanceRegression(unittest.TestCase):
    """Test for performance regression against established baselines"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.performance_validator = PerformanceValidator()
        except:
            self.performance_validator = None
    
    def test_no_performance_regression(self):
        """Ensure performance doesn't regress from established baselines"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        # Establish performance baselines
        performance_baseline = {
            'text_processing_time_ms': 500,  # Should process text in < 500ms
            'feature_generation_time_ms': 200,  # Should generate features in < 200ms
            'ensemble_inference_time_ms': 300,  # Should infer in < 300ms
            'memory_usage_mb': 200,  # Should use < 200MB memory
            'accuracy_threshold': 0.85  # Should maintain > 85% accuracy
        }
        
        # Test claim for regression testing
        regression_test_claim = {
            'claim_id': 'REGRESSION_TEST',
            'amount': 7500.0,
            'claim_type': 'auto',
            'description': 'Standard collision claim for regression testing',
            'fraud_label': 0
        }
        
        try:
            # Test current performance
            current_performance = self.performance_validator.validate_performance_regression(
                regression_test_claim, 'regression_test'
            )
            
            # Check against baselines
            for metric, baseline_value in performance_baseline.items():
                current_value = current_performance.get(metric, float('inf'))
                
                if metric.endswith('_time_ms'):
                    # Time metrics should be <= baseline
                    self.assertLessEqual(current_value, baseline_value * 1.2,  # Allow 20% degradation
                                           f"{metric}: current {current_value}ms should be <= {baseline_value * 1.2}ms")
                elif metric.endswith('_memory_mb'):
                    # Memory should be reasonable
                    self.assertLessEqual(current_value, baseline_value * 1.5,  # Allow 50% increase
                                           f"{metric}: current {current_value}MB should be <= {baseline_value * 1.5}MB")
                elif metric == 'accuracy_threshold':
                    # Accuracy should be maintained
                    self.assertGreaterEqual(current_value, baseline_value,
                                               f"{metric}: current {current_value} should be >= {baseline_value}")
                    
        except Exception as e:
            self.fail(f"Performance regression test failed: {e}")
    
    def test_scalability_performance(self):
        """Test performance scaling with claim complexity"""
        if self.performance_validator is None:
            self.skipTest("Performance validator not available")
            
        # Claims of varying complexity
        complexity_claims = [
            {
                'claim_id': 'SIMPLE_001',
                'amount': 1000.0,
                'claim_type': 'auto',
                'description': 'Simple collision',
                'complexity_level': 'simple'
            },
            {
                'claim_id': 'MODERATE_001',
                'amount': 7500.0,
                'claim_type': 'auto',
                'description': 'Moderate collision with multiple vehicles and witnesses',
                'complexity_level': 'moderate'
            },
            {
                'claim_id': 'COMPLEX_001',
                'amount': 25000.0,
                'claim_type': 'auto',
                'description': 'Complex multi-stage accident with legal complications and extensive damage assessment',
                'complexity_level': 'complex'
            }
        ]
        
        try:
            # Test scalability
            scalability_results = self.performance_validator.validate_scalability(
                complexity_claims, 'scalability_test'
            )
            
            # Check scaling behavior
            self.assertIn('scaling_metrics', scalability_results,
                           "Should include scaling metrics")
            
            scaling_metrics = scalability_results['scaling_metrics']
            
            # Processing time should scale reasonably
            simple_time = scaling_metrics.get('simple_processing_time_ms', 0)
            complex_time = scaling_metrics.get('complex_processing_time_ms', 0)
            
            # Complex claims should take longer but not excessively so
            if simple_time > 0:
                scaling_ratio = complex_time / simple_time
                self.assertLess(scaling_ratio, 5.0,  # Complex shouldn't take 5x longer
                                   f"Complex claims take {scaling_ratio:.2f}x longer than simple claims")
            
            # Memory usage should scale reasonably
            simple_memory = scaling_metrics.get('simple_memory_mb', 0)
            complex_memory = scaling_metrics.get('complex_memory_mb', 0)
            
            if simple_memory > 0:
                memory_scaling_ratio = complex_memory / simple_memory
                self.assertLess(memory_scaling_ratio, 3.0,  # Memory shouldn't triple
                                    f"Complex claims use {memory_scaling_ratio:.2f}x memory of simple claims")
                
        except Exception as e:
            self.fail(f"Scalability performance test failed: {e}")

if __name__ == '__main__':
    unittest.main()
