"""
Comprehensive Testing Suite for Enhanced Insurance Fraud Detection Components
Tests all AIML-compliant enhancements including BERT classifier, SAFE features, and performance validation
"""

import unittest
import numpy as np
import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import tempfile
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestEnhancedBERTClassifier(unittest.TestCase):
    """Test suite for Enhanced BERT Classifier"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock memory manager to avoid actual memory constraints during testing
        with patch('enhanced_bert_classifier.get_memory_manager'):
            from enhanced_bert_classifier import get_enhanced_bert_classifier
            self.classifier = get_enhanced_bert_classifier()

    def test_classifier_initialization(self):
        """Test BERT classifier initialization"""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(self.classifier.model_name, 'distilbert-base-uncased')
        self.assertIsNotNone(self.classifier.insurance_vocab)

    def test_insurance_vocabulary_loading(self):
        """Test insurance-specific vocabulary loading"""
        vocab = self.classifier.insurance_vocab
        self.assertIn('accident_types', vocab)
        self.assertIn('fraud_indicators', vocab)
        self.assertIn('medical_terms', vocab)
        self.assertTrue(len(vocab['accident_types']) > 0)

    def test_domain_text_preprocessing(self):
        """Test domain-specific text preprocessing"""
        test_text = "Car accident on highway with DR. Smith present"
        processed = self.classifier._preprocess_text(test_text)
        self.assertIn('vehicle', processed.lower())
        self.assertIn('doctor', processed.lower())

    def test_multitask_classification(self):
        """Test multi-task classification functionality"""
        claim_text = "Vehicle collision at intersection causing severe damage to car"
        claim_data = {
            'claim_type': 'auto',
            'amount': 5000.0
        }

        with patch.object(self.classifier, 'base_model') as mock_model:
            # Mock BERT output
            mock_output = Mock()
            mock_output.pooler_output = Mock()
            mock_output.pooler_output.cpu.return_value.numpy.return_value = np.random.randn(1, 768)
            mock_model.return_value = mock_output

            with patch.object(self.classifier, 'domain_adapter') as mock_adapter:
                mock_adapter.return_value = np.random.randn(1, 128)

                with patch.object(self.classifier, 'classification_heads') as mock_heads:
                    # Mock task predictions
                    mock_logits = {task: np.random.randn(1, num_classes)
                                 for task, num_classes in [('driving_status', 5), ('accident_type', 12)]}
                    mock_heads.return_value = mock_logits

                    result = self.classifier.classify_claim(claim_text, claim_data)

                    self.assertIn('task_predictions', result)
                    self.assertIn('fraud_indicators', result)

    def test_fraud_risk_extraction(self):
        """Test fraud risk indicator extraction"""
        suspicious_text = "Late night accident with no witnesses, quick settlement requested"
        fraud_indicators = self.classifier._extract_fraud_risk_indicators(suspicious_text)

        self.assertIn('risk_indicators', fraud_indicators)
        self.assertIn('total_risk_score', fraud_indicators)
        self.assertGreater(fraud_indicators['total_risk_score'], 0)

    def test_structured_feature_extraction(self):
        """Test structured feature extraction for ML"""
        claim_text = "Vehicle accident with minor damage"
        features = self.classifier.extract_structured_features(claim_text)

        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 200)  # Target feature length
        self.assertTrue(all(isinstance(f, (int, float)) for f in features))

    def test_memory_usage_reporting(self):
        """Test memory usage reporting"""
        memory_info = self.classifier.get_memory_usage()
        self.assertIn('component', memory_info)
        self.assertIn('model_name', memory_info)
        self.assertEqual(memory_info['component'], 'enhanced_bert_classifier')

class TestEnhancedSAFEFeatures(unittest.TestCase):
    """Test suite for Enhanced SAFE Feature Engineering"""

    def setUp(self):
        """Set up test fixtures"""
        with patch('enhanced_safe_features.get_memory_manager'):
            from enhanced_safe_features import get_enhanced_safe_features
            self.safe_features = get_enhanced_safe_features(max_features=200)

    def test_safe_initialization(self):
        """Test SAFE features initialization"""
        self.assertIsNotNone(self.safe_features)
        self.assertEqual(self.safe_features.max_features, 200)
        self.assertIsNotNone(self.safe_features.interaction_generator)
        self.assertIsNotNone(self.safe_features.math_transformer)

    def test_historical_data_loading(self):
        """Test enhanced historical data loading"""
        historical_data = self.safe_features.historical_data
        self.assertIn('avg_amounts', historical_data)
        self.assertIn('amount_percentiles', historical_data)
        self.assertIn('location_risks', historical_data)

        # Check for claim type-specific data
        avg_amounts = historical_data['avg_amounts']
        self.assertIn('auto', avg_amounts)
        self.assertIn('collision', avg_amounts['auto'])

    def test_enhanced_temporal_features(self):
        """Test enhanced temporal feature extraction"""
        claim_data = {
            'accident_time': '23:30',
            'accident_date': '2024-12-25'
        }

        temporal_features = self.safe_features._extract_enhanced_temporal_features(claim_data)
        self.assertIn('temporal_peak_risk', temporal_features)
        self.assertIn('temporal_weekend_risk', temporal_features)
        self.assertIn('temporal_is_holiday', temporal_features)
        self.assertIn('temporal_high_season', temporal_features)

    def test_enhanced_amount_features(self):
        """Test enhanced amount feature extraction"""
        claim_data = {
            'amount': 8500.0,
            'claim_type': 'auto',
            'claim_subtype': 'collision'
        }

        amount_features = self.safe_features._extract_enhanced_amount_features(claim_data)
        self.assertIn('amount_log', amount_features)
        self.assertIn('amount_sqrt', amount_features)
        self.assertIn('amount_percentile', amount_features)
        self.assertIn('amount_is_high', amount_features)

    def test_behavioral_features(self):
        """Test new behavioral feature extraction"""
        claim_data = {
            'description': 'URGENT need immediate settlement for car accident, very angry about situation',
            'contact_method': 'phone'
        }

        behavioral_features = self.safe_features._extract_behavioral_features(claim_data)
        self.assertIn('behavioral_urgency_score', behavioral_features)
        self.assertIn('behavioral_emotional_score', behavioral_features)
        self.assertIn('behavioral_phone_contact', behavioral_features)

    def test_feature_interaction_generation(self):
        """Test smart feature interaction generation"""
        base_features = {
            'amount': 5000.0,
            'claim_type_auto': 1.0,
            'temporal_risk': 0.7,
            'frequency_recent_claims': 0.3
        }

        interactions = self.safe_features.interaction_generator.generate_interactions(base_features)
        self.assertGreater(len(interactions), 0)
        self.assertTrue(any('_x_' in key for key in interactions.keys()))

    def test_mathematical_transformations(self):
        """Test mathematical feature transformations"""
        base_features = {
            'amount': 1000.0,
            'claim_frequency': 3.0,
            'risk_score': 0.8
        }

        transformations = self.safe_features.math_transformer.transform_features(base_features)
        self.assertIn('log_amount', transformations)
        self.assertIn('sqrt_claim_frequency', transformations)
        self.assertIn('squared_risk_score', transformations)

    def test_comprehensive_feature_generation(self):
        """Test comprehensive feature generation"""
        claim_data = {
            'claim_id': 'test_001',
            'customer_id': 'cust_123',
            'claim_type': 'auto',
            'description': 'Vehicle collision at highway intersection during evening hours',
            'amount': 7500.0,
            'accident_time': '19:45',
            'accident_date': '2024-11-18',
            'location': 'highway_intersection',
            'claimant_age': 32,
            'policy_age_months': 18
        }

        features = self.safe_features.generate_comprehensive_features(claim_data)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 100)  # Should generate substantial features

    def test_feature_importance_calculation(self):
        """Test feature importance scoring"""
        importance = self.safe_features._calculate_feature_importance('amount_risk_score', 0.75)
        self.assertGreaterEqual(importance, 0.0)
        self.assertLessEqual(importance, 1.0)

class TestMemoryManager(unittest.TestCase):
    """Test suite for Memory Manager"""

    def setUp(self):
        """Set up test fixtures"""
        from memory_manager import MemoryManager, ResourceLimits
        self.memory_manager = MemoryManager(ResourceLimits(max_ram_mb=100))

    def test_memory_manager_initialization(self):
        """Test memory manager initialization"""
        self.assertIsNotNone(self.memory_manager)
        self.assertEqual(self.memory_manager.limits.max_ram_mb, 100)

    def test_memory_usage_checking(self):
        """Test memory usage checking"""
        memory_info = self.memory_manager.check_memory_usage()
        self.assertIn('current_usage_mb', memory_info)
        self.assertIn('usage_percentage', memory_info)
        self.assertIn('status', memory_info)

    def test_memory_allocation_checking(self):
        """Test memory allocation checking"""
        # Test small allocation that should succeed
        allocation_result = self.memory_manager.can_allocate(10, 'test_component')
        self.assertIn('can_allocate', allocation_result)

        # Test large allocation that should fail
        large_allocation = self.memory_manager.can_allocate(200, 'test_component')
        self.assertFalse(large_allocation['can_allocate'])

    def test_component_memory_management(self):
        """Test component memory allocation/release"""
        # Test allocation
        allocated = self.memory_manager.allocate_component_memory('test_component', 20)
        self.assertTrue(allocated)
        self.assertEqual(self.memory_manager.component_usage['test_component'], 20)

        # Test release
        released = self.memory_manager.release_component_memory('test_component')
        self.assertTrue(released)
        self.assertEqual(self.memory_manager.component_usage['test_component'], 0)

    def test_garbage_collection(self):
        """Test garbage collection functionality"""
        gc_result = self.memory_manager.force_garbage_collection()
        self.assertIn('collected_objects', gc_result)
        self.assertIn('memory_freed_mb', gc_result)

    def test_efficiency_scoring(self):
        """Test memory efficiency scoring"""
        efficiency_score = self.memory_manager.get_memory_efficiency_score()
        self.assertGreaterEqual(efficiency_score, 0.0)
        self.assertLessEqual(efficiency_score, 1.0)

    def test_recommendations(self):
        """Test memory optimization recommendations"""
        recommendations = self.memory_manager.get_recommendations()
        self.assertIsInstance(recommendations, list)

class TestPerformanceValidator(unittest.TestCase):
    """Test suite for Performance Validator"""

    def setUp(self):
        """Set up test fixtures"""
        with patch('performance_validator.get_memory_manager'):
            from performance_validator import get_performance_validator
            self.validator = get_performance_validator()

    def test_validator_initialization(self):
        """Test performance validator initialization"""
        self.assertIsNotNone(self.validator)
        self.assertIsNotNone(self.validator.metrics_collector)
        self.assertEqual(len(self.validator.test_data), 100)  # Default synthetic data

    def test_synthetic_test_data_generation(self):
        """Test synthetic test data generation"""
        test_data = self.validator.test_data
        self.assertEqual(len(test_data), 100)

        # Check data structure
        sample = test_data[0]
        required_fields = ['id', 'claim_type', 'description', 'is_fraud', 'amount']
        for field in required_fields:
            self.assertIn(field, sample)

    def test_test_description_generation(self):
        """Test test claim description generation"""
        legitimate_desc = self.validator._generate_test_description('auto', False)
        fraudulent_desc = self.validator._generate_test_description('auto', True)

        self.assertIsInstance(legitimate_desc, str)
        self.assertIsInstance(fraudulent_desc, str)
        self.assertNotEqual(legitimate_desc, fraudulent_desc)

    def test_metrics_to_object_conversion(self):
        """Test conversion of metrics dict to PerformanceMetrics object"""
        test_metrics = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.90,
            'f1_score': 0.85,
            'auc_roc': 0.88,
            'processing_time': 1.5,
            'memory_usage': 200
        }

        performance_obj = self.validator._dict_to_metrics(test_metrics)
        self.assertEqual(performance_obj.accuracy, 0.85)
        self.assertEqual(performance_obj.processing_time, 1.5)

    def test_improvement_calculation(self):
        """Test improvement percentage calculation"""
        control_metrics = {'f1_score': 0.75, 'avg_processing_time_ms': 1000}
        treatment_metrics = {'f1_score': 0.85, 'avg_processing_time_ms': 800}

        improvement = self.validator._calculate_improvement(control_metrics, treatment_metrics)
        self.assertGreater(improvement, 0)

    def test_recommendation_generation(self):
        """Test test result recommendation generation"""
        # Test strong improvement
        recommendation1 = self.validator._generate_recommendation(15.0, True, 'text_classifier')
        self.assertIn('STRONGLY RECOMMENDED', recommendation1)

        # Test decline
        recommendation2 = self.validator._generate_recommendation(-10.0, True, 'safe_features')
        self.assertIn('NOT RECOMMENDED', recommendation2)

class TestCRFLayer(unittest.TestCase):
    """Test suite for CRF Layer Implementation"""

    def setUp(self):
        """Set up test fixtures"""
        from crf_layer import get_memory_efficient_crf
        with patch('crf_layer.get_memory_manager'):
            self.crf = get_memory_efficient_crf(num_tags=5)

    def test_crf_initialization(self):
        """Test CRF layer initialization"""
        self.assertIsNotNone(self.crf)
        self.assertEqual(self.crf.num_tags, 5)

    def test_crf_forward_pass(self):
        """Test CRF forward pass"""
        import torch
        emissions = torch.randn(2, 10, 5)  # batch=2, seq_len=10, num_tags=5

        output = self.crf(emissions)
        self.assertEqual(output.shape, emissions.shape)

    def test_viterbi_decoding(self):
        """Test Viterbi decoding"""
        import torch
        emissions = torch.randn(1, 5, 5)  # batch=1, seq_len=5, num_tags=5

        predictions = self.crf.viterbi_decode(emissions)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], list)

class TestAIMLMultiTaskHeads(unittest.TestCase):
    """Test suite for AIML Multi-Task Classification Heads"""

    def setUp(self):
        """Set up test fixtures"""
        from aiml_multi_task_heads import get_aiml_multi_task_heads
        self.multi_task_heads = get_aiml_multi_task_heads(input_dim=128, hidden_dim=64)

    def test_multi_task_initialization(self):
        """Test multi-task heads initialization"""
        self.assertIsNotNone(self.multi_task_heads)
        self.assertEqual(len(self.multi_task_heads.task_specs), 6)  # AIML: 6 tasks

        # Check task specifications
        self.assertIn('driving_status', self.multi_task_heads.task_specs)
        self.assertIn('accident_type', self.multi_task_heads.task_specs)
        self.assertEqual(self.multi_task_heads.task_specs['driving_status'].num_classes, 5)
        self.assertEqual(self.multi_task_heads.task_specs['accident_type'].num_classes, 12)

    def test_forward_pass(self):
        """Test multi-task forward pass"""
        import torch
        batch_size = 2
        input_dim = 128

        x = torch.randn(batch_size, input_dim)
        outputs = self.multi_task_heads(x)

        self.assertIn('task_predictions', outputs)
        task_predictions = outputs['task_predictions']

        # Check all tasks have predictions
        for task_name in self.multi_task_heads.task_specs.keys():
            self.assertIn(task_name, task_predictions)
            self.assertIn('predictions', task_predictions[task_name])
            self.assertIn('confidence', task_predictions[task_name])

    def test_loss_computation(self):
        """Test multi-task loss computation"""
        import torch

        # Create dummy outputs and targets
        outputs = {}
        targets = {}

        for task_name, task_spec in self.multi_task_heads.task_specs.items():
            batch_size = 2
            outputs[task_name] = {
                'logits': torch.randn(batch_size, task_spec.num_classes)
            }
            targets[task_name] = torch.randint(0, task_spec.num_classes, (batch_size,))

        loss_dict = self.multi_task_heads.compute_loss(outputs, targets)

        self.assertIn('total_loss', loss_dict)
        self.assertIn('task_losses', loss_dict)
        self.assertIsInstance(loss_dict['total_loss'], torch.Tensor)

class TestIntegration(unittest.TestCase):
    """Integration tests for the entire enhanced system"""

    def setUp(self):
        """Set up integration test fixtures"""
        # Mock memory manager for all components
        with patch('enhanced_bert_classifier.get_memory_manager'), \
             patch('enhanced_safe_features.get_memory_manager'), \
             patch('performance_validator.get_memory_manager'):

            from enhanced_bert_classifier import get_enhanced_bert_classifier
            from enhanced_safe_features import get_enhanced_safe_features
            from performance_validator import get_performance_validator

            self.text_classifier = get_enhanced_bert_classifier()
            self.safe_features = get_enhanced_safe_features()
            self.performance_validator = get_performance_validator()

    def test_end_to_end_claim_processing(self):
        """Test end-to-end claim processing pipeline"""
        claim_data = {
            'claim_id': 'integration_test_001',
            'customer_id': 'cust_integration_123',
            'claim_type': 'auto',
            'description': 'Vehicle collision on highway during evening rush hour, moderate damage to front bumper',
            'amount': 4500.0,
            'accident_time': '18:30',
            'accident_date': '2024-11-18',
            'location': 'highway',
            'claimant_age': 28
        }

        # Mock the BERT model to avoid heavy computation in tests
        with patch.object(self.text_classifier, 'base_model') as mock_model:
            mock_output = Mock()
            mock_output.pooler_output = Mock()
            mock_output.pooler_output.cpu.return_value.numpy.return_value = np.random.randn(1, 768)
            mock_model.return_value = mock_output

            # Process text classification
            with patch.object(self.text_classifier, 'domain_adapter'), \
                 patch.object(self.text_classifier, 'classification_heads'):

                classification_result = self.text_classifier.classify_claim(
                    claim_data['description'], claim_data
                )

                # Process feature generation
                features = self.safe_features.generate_comprehensive_features(claim_data)

                # Verify results
                self.assertIsNotNone(classification_result)
                self.assertIsNotNone(features)
                self.assertIsInstance(features, dict)
                self.assertGreater(len(features), 50)  # Should have substantial features

    def test_performance_validation_integration(self):
        """Test performance validation integration"""
        # Create mock components for testing
        mock_text_classifier = Mock()
        mock_text_classifier.classify_claim.return_value = {
            'fraud_indicators': {'total_risk_score': 0.3, 'analysis_confidence': 0.8}
        }

        mock_safe_features = Mock()
        mock_safe_features.generate_comprehensive_features.return_value = {
            'feature_1': 0.5, 'feature_2': 1.0, 'total_features': 150
        }

        # Test text classification validation
        text_metrics = self.performance_validator.validate_text_processing(
            mock_text_classifier, 'integration_test_classifier'
        )
        self.assertIn('accuracy', text_metrics)
        self.assertIn('processing_times', text_metrics)

        # Test feature engineering validation
        feature_metrics = self.performance_validator.validate_feature_engineering(
            mock_safe_features, 'integration_test_features'
        )
        self.assertIn('total_unique_features', feature_metrics)
        self.assertIn('avg_processing_time_ms', feature_metrics)

    def test_memory_constraints_simulation(self):
        """Test system behavior under memory constraints"""
        from memory_manager import MemoryManager, ResourceLimits

        # Create memory manager with low limit for testing
        low_memory_manager = MemoryManager(ResourceLimits(max_ram_mb=50))

        # Test allocation under constraints
        small_allocation = low_memory_manager.can_allocate(10, 'test_component')
        self.assertTrue(small_allocation['can_allocate'])

        large_allocation = low_memory_manager.can_allocate(100, 'test_component')
        self.assertFalse(large_allocation['can_allocate'])

    def test_real_time_metrics_collection(self):
        """Test real-time metrics collection"""
        from performance_validator import RealTimeMetricsCollector

        metrics_collector = RealTimeMetricsCollector(max_history=100)

        # Record some mock predictions
        for i in range(10):
            metrics_collector.record_prediction(
                {'fraud_score': 0.3 + i * 0.05, 'confidence': 0.8},
                processing_time=0.1 + i * 0.01,
                success=True
            )

        # Get statistics
        stats = metrics_collector.get_real_time_stats()
        self.assertEqual(stats['total_requests'], 10)
        self.assertEqual(stats['total_errors'], 0)
        self.assertGreater(stats['avg_processing_time_ms'], 0)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_empty_claim_data(self):
        """Test handling of empty claim data"""
        with patch('enhanced_safe_features.get_memory_manager'):
            from enhanced_safe_features import get_enhanced_safe_features
            safe_features = get_enhanced_safe_features()

            empty_data = {}
            features = safe_features.generate_comprehensive_features(empty_data)
            self.assertIsInstance(features, dict)

    def test_invalid_text_input(self):
        """Test handling of invalid text input"""
        with patch('enhanced_bert_classifier.get_memory_manager'):
            from enhanced_bert_classifier import get_enhanced_bert_classifier

            # This should handle gracefully when BERT is not properly initialized
            try:
                classifier = get_enhanced_bert_classifier()
                result = classifier.classify_claim("", {})
                self.assertIsNotNone(result)
            except Exception as e:
                # Expected if BERT models are not available
                self.assertIsInstance(e, (ImportError, ModuleNotFoundError, AttributeError))

    def test_memory_exhaustion_simulation(self):
        """Test behavior when memory is exhausted"""
        with patch('enhanced_safe_features.get_memory_manager') as mock_mm:
            # Mock memory manager that always fails allocation
            mock_memory_manager = Mock()
            mock_memory_manager.can_allocate.return_value = {'can_allocate': False}
            mock_memory_manager.check_memory_usage.return_value = {'current_usage_mb': 950}
            mock_mm.return_value = mock_memory_manager

            from enhanced_safe_features import get_enhanced_safe_features
            safe_features = get_enhanced_safe_features(max_features=10)  # Small limit

            claim_data = {'amount': 1000}
            features = safe_features.generate_comprehensive_features(claim_data)
            # Should return empty dict or handle gracefully
            self.assertIsInstance(features, dict)

def run_comprehensive_tests():
    """Run all enhanced component tests"""
    print("=" * 70)
    print("COMPREHENSIVE TESTING SUITE FOR ENHANCED INSURANCE FRAUD DETECTION")
    print("=" * 70)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestEnhancedBERTClassifier,
        TestEnhancedSAFEFeatures,
        TestMemoryManager,
        TestPerformanceValidator,
        TestCRFLayer,
        TestAIMLMultiTaskHeads,
        TestIntegration,
        TestErrorHandling
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    if result.wasSuccessful():
        print("\n🎉 All tests passed! Enhanced components are ready for deployment.")
    else:
        print("\n⚠️  Some tests failed. Please review and fix issues before deployment.")

    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)