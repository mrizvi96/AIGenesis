"""
Performance Validator for Enhanced Insurance Fraud Detection System
Comprehensive A/B testing, metrics tracking, and benchmarking framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
import json
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from .memory_manager import get_memory_manager
except ImportError:
    from memory_manager import get_memory_manager
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    processing_time: float
    memory_usage: float

@dataclass
class ABTestResult:
    """A/B test result container"""
    test_name: str
    control_metrics: PerformanceMetrics
    treatment_metrics: PerformanceMetrics
    significance_level: float
    p_value: float
    is_significant: bool
    improvement_percentage: float
    recommendation: str

@dataclass
class BenchmarkResult:
    """Benchmark result container"""
    component_name: str
    test_data_size: int
    avg_processing_time: float
    max_processing_time: float
    min_processing_time: float
    memory_peak: float
    memory_average: float
    throughput_per_second: float
    success_rate: float
    error_rate: float

class RealTimeMetricsCollector:
    """
    Real-time metrics collection with thread-safe operations
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize real-time metrics collector

        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.memory_manager = get_memory_manager()

        # Thread-safe queues for real-time data
        self.metrics_queue = queue.Queue(maxsize=max_history)
        self.performance_history = []
        self.error_history = []

        # Lock for thread safety
        self._lock = threading.RLock()

        # Metrics aggregation
        self.current_session_start = datetime.now()
        self.total_requests = 0
        self.total_errors = 0
        self.total_processing_time = 0.0

    def record_prediction(self, prediction_data: Dict[str, Any], processing_time: float, success: bool = True):
        """
        Record a prediction with its metrics

        Args:
            prediction_data: Prediction result data
            processing_time: Time taken for prediction
            success: Whether prediction was successful
        """
        try:
            timestamp = datetime.now()
            memory_info = self.memory_manager.check_memory_usage()

            metric_entry = {
                'timestamp': timestamp,
                'processing_time': processing_time,
                'memory_usage_mb': memory_info.get('current_usage_mb', 0),
                'success': success,
                'prediction_confidence': prediction_data.get('confidence', 0.0),
                'fraud_score': prediction_data.get('fraud_score', 0.0)
            }

            # Add to queue (non-blocking)
            try:
                self.metrics_queue.put_nowait(metric_entry)
            except queue.Full:
                # Remove oldest entry if queue is full
                try:
                    self.metrics_queue.get_nowait()
                    self.metrics_queue.put_nowait(metric_entry)
                except queue.Empty:
                    pass

            # Update aggregates
            with self._lock:
                self.total_requests += 1
                self.total_processing_time += processing_time
                if not success:
                    self.total_errors += 1

        except Exception as e:
            logger.error(f"Failed to record prediction metric: {e}")

    def get_real_time_stats(self) -> Dict[str, Any]:
        """
        Get real-time performance statistics

        Returns:
            Dictionary with real-time statistics
        """
        with self._lock:
            # Process queue to update history
            while not self.metrics_queue.empty():
                try:
                    metric = self.metrics_queue.get_nowait()
                    self.performance_history.append(metric)

                    # Keep only recent history
                    if len(self.performance_history) > self.max_history:
                        self.performance_history.pop(0)
                except queue.Empty:
                    break

            if not self.performance_history:
                return self._empty_stats()

            recent_metrics = self.performance_history[-100:]  # Last 100 metrics

            processing_times = [m['processing_time'] for m in recent_metrics]
            memory_usages = [m['memory_usage_mb'] for m in recent_metrics]
            confidences = [m['prediction_confidence'] for m in recent_metrics if m['prediction_confidence'] > 0]

            session_duration = (datetime.now() - self.current_session_start).total_seconds()

            return {
                'session_duration_seconds': session_duration,
                'total_requests': self.total_requests,
                'total_errors': self.total_errors,
                'error_rate': self.total_errors / max(self.total_requests, 1),
                'avg_processing_time_ms': np.mean(processing_times) * 1000 if processing_times else 0,
                'max_processing_time_ms': np.max(processing_times) * 1000 if processing_times else 0,
                'min_processing_time_ms': np.min(processing_times) * 1000 if processing_times else 0,
                'p95_processing_time_ms': np.percentile(processing_times, 95) * 1000 if processing_times else 0,
                'current_memory_mb': memory_usages[-1] if memory_usages else 0,
                'avg_memory_mb': np.mean(memory_usages) if memory_usages else 0,
                'peak_memory_mb': np.max(memory_usages) if memory_usages else 0,
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'requests_per_second': self.total_requests / max(session_duration, 1),
                'uptime_hours': session_duration / 3600,
                'success_rate': (self.total_requests - self.total_errors) / max(self.total_requests, 1)
            }

    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics structure"""
        return {
            'session_duration_seconds': 0, 'total_requests': 0, 'total_errors': 0,
            'error_rate': 0.0, 'avg_processing_time_ms': 0.0, 'max_processing_time_ms': 0.0,
            'min_processing_time_ms': 0.0, 'p95_processing_time_ms': 0.0, 'current_memory_mb': 0.0,
            'avg_memory_mb': 0.0, 'peak_memory_mb': 0.0, 'avg_confidence': 0.0,
            'requests_per_second': 0.0, 'uptime_hours': 0.0, 'success_rate': 1.0
        }

    def reset_session(self):
        """Reset session statistics"""
        with self._lock:
            self.current_session_start = datetime.now()
            self.total_requests = 0
            self.total_errors = 0
            self.total_processing_time = 0.0
            self.performance_history.clear()
            self.error_history.clear()

class PerformanceValidator:
    """
    Comprehensive performance validation and A/B testing framework
    """

    def __init__(self, test_data_path: Optional[str] = None):
        """
        Initialize performance validator

        Args:
            test_data_path: Path to test data file
        """
        self.memory_manager = get_memory_manager()
        self.metrics_collector = RealTimeMetricsCollector()

        # Test data
        self.test_data = self._load_test_data(test_data_path)

        # Baseline metrics storage
        self.baseline_metrics = {}
        self.current_metrics = {}

        # A/B test results
        self.ab_test_history = []

        # Benchmark results
        self.benchmark_results = {}

        # Performance targets from AIML paper
        self.aiml_targets = {
            'text_classification_f1': 0.85,  # AIML achieved 0.79-0.93
            'feature_count': 200,  # From 33 to 200 features
            'auc_improvement': 0.03,  # 3% AUC improvement
            'memory_usage_mb': 400,  # Under 400MB
            'processing_time_ms': 2000  # Under 2 seconds
        }

        logger.info(f"PerformanceValidator initialized with {len(self.test_data)} test samples")

    def _load_test_data(self, test_data_path: Optional[str]) -> List[Dict[str, Any]]:
        """
        Load test data for validation

        Args:
            test_data_path: Path to test data file

        Returns:
            List of test samples
        """
        if test_data_path and test_data_path.endswith('.json'):
            try:
                with open(test_data_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load test data from {test_data_path}: {e}")

        # Generate synthetic test data if no file provided
        return self._generate_synthetic_test_data()

    def _generate_synthetic_test_data(self) -> List[Dict[str, Any]]:
        """
        Generate synthetic test data for validation

        Returns:
            List of synthetic test samples
        """
        synthetic_data = []

        # Generate diverse test cases
        claim_types = ['auto', 'home', 'health', 'travel', 'life']
        fraud_patterns = ['legitimate', 'suspicious_timing', 'exaggerated_damage', 'staged_accident', 'multiple_claims']

        for i in range(100):  # Generate 100 test samples
            claim_type = np.random.choice(claim_types)
            is_fraud = np.random.random() < 0.3  # 30% fraud rate

            sample = {
                'id': f'test_{i}',
                'claim_type': claim_type,
                'amount': np.random.choice([
                    np.random.uniform(500, 2000),   # Small claims
                    np.random.uniform(2000, 8000),  # Medium claims
                    np.random.uniform(8000, 25000) # Large claims
                ], p=[0.5, 0.35, 0.15]),
                'description': self._generate_test_description(claim_type, is_fraud),
                'is_fraud': is_fraud,
                'fraud_pattern': np.random.choice(fraud_patterns) if is_fraud else 'legitimate',
                'accident_time': f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}",
                'location': np.random.choice(['downtown', 'suburban', 'rural', 'highway', 'parking_lot']),
                'customer_id': f"customer_{np.random.randint(1, 1000)}",
                'claimant_age': np.random.randint(18, 80),
                'policy_age_months': np.random.randint(1, 120)
            }

            synthetic_data.append(sample)

        return synthetic_data

    def _generate_test_description(self, claim_type: str, is_fraud: bool) -> str:
        """
        Generate test claim description

        Args:
            claim_type: Type of claim
            is_fraud: Whether claim is fraudulent

        Returns:
            Generated description
        """
        legitimate_descriptions = {
            'auto': [
                "Minor collision at intersection, bumper damage, no injuries",
                "Rear-ended in traffic, whiplash injury, rear bumper damage",
                "Hail damage to vehicle and windshield, parked during storm",
                "Single vehicle accident, hit patch of ice, minor vehicle damage"
            ],
            'home': [
                "Water damage from burst pipe in kitchen, affected flooring",
                "Storm damage to roof, missing shingles, minor ceiling leak",
                "Theft of electronics from living room, forced entry through back door",
                "Fire damage in kitchen from grease fire, smoke damage throughout"
            ],
            'health': [
                "Emergency room visit for broken arm from fall at home",
                "Appendicitis surgery, 2-day hospital stay, full recovery",
                "Physical therapy for sports injury, 12 sessions prescribed",
                "Diagnostic imaging for persistent headaches, MRI scan performed"
            ],
            'travel': [
                "Flight cancellation due to mechanical issues, missed connection",
                "Lost luggage on international flight, compensation claim for essentials",
                "Medical emergency abroad, food poisoning, hospital treatment",
                "Trip interruption due to family emergency, early return flights"
            ],
            'life': [
                "Term life policy claim for accidental death, workplace incident",
                "Critical illness diagnosis, cancer treatment coverage",
                "Disability claim following workplace injury, long-term care needed",
                "Accidental death benefit, car accident, policy active"
            ]
        }

        fraudulent_descriptions = {
            'auto': [
                "Major collision but no police report, inconsistent details",
                "Claims vehicle was totaled but minimal damage visible",
                "Multiple passengers claim injuries but no medical records",
                "Accident occurred at 3 AM but reported immediately next day"
            ],
            'home': [
                "Fire damage claimed but no fire department report available",
                "Theft of high-value items but no proof of ownership",
                "Water damage claimed but no signs of water damage present",
                "Multiple break-ins reported in short time period"
            ],
            'health': [
                "Excessive medical procedures for minor injury",
                "Claims for treatments never received according to providers",
                "Multiple specialists for simple condition, over-treatment",
                "Emergency room visit for minor complaint with costly tests"
            ],
            'travel': [
                "Medical emergency claim but no medical documentation",
                "Lost expensive jewelry not covered by policy",
                "Trip cancellation for vague reasons, no proof",
                "Multiple small claims from same trip period"
            ],
            'life': [
                "Claim filed immediately after policy purchase",
                "Suspicious circumstances around death claim",
                "Inconsistent medical history for critical illness claim",
                "Multiple policies on same person, excessive coverage"
            ]
        }

        descriptions = fraudulent_descriptions if is_fraud else legitimate_descriptions
        category_descriptions = descriptions.get(claim_type, ["Standard claim description"])
        return np.random.choice(category_descriptions)

    def validate_text_processing(self, text_classifier, component_name: str = "text_classifier") -> Dict[str, Any]:
        """
        Validate enhanced text processing performance

        Args:
            text_classifier: Text classifier instance
            component_name: Name of component being tested

        Returns:
            Validation results
        """
        logger.info(f"Validating text processing: {component_name}")

        predictions = []
        true_labels = []
        processing_times = []
        confidences = []

        for sample in self.test_data:
            try:
                start_time = time.time()

                # Get prediction
                result = text_classifier.classify_claim(sample['description'], sample)

                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Extract fraud prediction
                fraud_score = result.get('fraud_indicators', {}).get('total_risk_score', 0.0)
                confidence = result.get('fraud_indicators', {}).get('analysis_confidence', 0.0)

                predictions.append(fraud_score)
                true_labels.append(int(sample['is_fraud']))
                confidences.append(confidence)

                # Record in real-time metrics
                self.metrics_collector.record_prediction(
                    {'fraud_score': fraud_score, 'confidence': confidence},
                    processing_time,
                    success=True
                )

            except Exception as e:
                logger.error(f"Text processing validation failed for sample {sample['id']}: {e}")
                processing_times.append(0.0)
                self.metrics_collector.record_prediction({}, 0.0, success=False)

        # Calculate metrics
        binary_predictions = [1 if score > 0.5 else 0 for score in predictions]

        metrics = {
            'accuracy': accuracy_score(true_labels, binary_predictions),
            'precision': precision_score(true_labels, binary_predictions, average='weighted', zero_division=0),
            'recall': recall_score(true_labels, binary_predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(true_labels, binary_predictions, average='weighted', zero_division=0),
            'auc_roc': roc_auc_score(true_labels, predictions) if len(set(true_labels)) > 1 else 0.0,
            'processing_times': processing_times,
            'avg_processing_time_ms': np.mean(processing_times) * 1000,
            'max_processing_time_ms': np.max(processing_times) * 1000,
            'avg_confidence': np.mean(confidences),
            'total_samples': len(predictions),
            'component_name': component_name
        }

        # Calculate additional metrics
        tn, fp, fn, tp = confusion_matrix(true_labels, binary_predictions).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Memory usage
        memory_info = self.memory_manager.check_memory_usage()
        metrics['memory_usage_mb'] = memory_info.get('current_usage_mb', 0)

        # Compare with targets
        metrics['meets_f1_target'] = metrics['f1_score'] >= self.aiml_targets['text_classification_f1']
        metrics['meets_time_target'] = metrics['avg_processing_time_ms'] <= self.aiml_targets['processing_time_ms']

        # Store metrics
        self.current_metrics[component_name] = metrics

        return metrics

    def validate_feature_engineering(self, safe_engine, component_name: str = "safe_features") -> Dict[str, Any]:
        """
        Validate enhanced SAFE feature engineering

        Args:
            safe_engine: SAFE feature engine instance
            component_name: Name of component being tested

        Returns:
            Validation results
        """
        logger.info(f"Validating feature engineering: {component_name}")

        feature_sets = {}
        processing_times = []
        feature_counts = []

        for sample in self.test_data:
            try:
                start_time = time.time()

                # Generate features
                features = safe_engine.generate_comprehensive_features(sample)

                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                feature_counts.append(len(features))

                feature_sets[sample['id']] = features

                # Record in real-time metrics
                self.metrics_collector.record_prediction(
                    {'feature_count': len(features)},
                    processing_time,
                    success=True
                )

            except Exception as e:
                logger.error(f"Feature engineering validation failed for sample {sample['id']}: {e}")
                processing_times.append(0.0)
                feature_counts.append(0)
                self.metrics_collector.record_prediction({}, 0.0, success=False)

        # Analyze features
        all_features = set()
        for features in feature_sets.values():
            all_features.update(features.keys())

        metrics = {
            'total_unique_features': len(all_features),
            'avg_features_per_sample': np.mean(feature_counts),
            'max_features_per_sample': np.max(feature_counts),
            'min_features_per_sample': np.min(feature_counts),
            'avg_processing_time_ms': np.mean(processing_times) * 1000,
            'max_processing_time_ms': np.max(processing_times) * 1000,
            'feature_generation_success_rate': len([f for f in feature_counts if f > 0]) / len(feature_counts),
            'total_samples': len(feature_sets),
            'component_name': component_name
        }

        # Memory usage
        memory_info = self.memory_manager.check_memory_usage()
        metrics['memory_usage_mb'] = memory_info.get('current_usage_mb', 0)

        # Compare with targets
        metrics['meets_feature_count_target'] = metrics['total_unique_features'] >= self.aiml_targets['feature_count']
        metrics['meets_time_target'] = metrics['avg_processing_time_ms'] <= self.aiml_targets['processing_time_ms']

        # Feature analysis
        if feature_sets:
            feature_analysis = self._analyze_features(feature_sets)
            metrics.update(feature_analysis)

        # Store metrics
        self.current_metrics[component_name] = metrics

        return metrics

    def _analyze_features(self, feature_sets: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze generated features

        Args:
            feature_sets: Dictionary of feature sets

        Returns:
            Feature analysis results
        """
        all_feature_names = set()
        for features in feature_sets.values():
            all_feature_names.update(features.keys())

        feature_stats = {}
        for feature_name in all_feature_names:
            values = [features.get(feature_name, 0) for features in feature_sets.values()]
            feature_stats[feature_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'non_zero_count': sum(1 for v in values if v != 0),
                'coverage': sum(1 for v in values if v != 0) / len(values)
            }

        # Categorize features
        feature_categories = {
            'high_coverage': [],
            'medium_coverage': [],
            'low_coverage': [],
            'high_variance': [],
            'low_variance': []
        }

        for feature_name, stats in feature_stats.items():
            if stats['coverage'] >= 0.8:
                feature_categories['high_coverage'].append(feature_name)
            elif stats['coverage'] >= 0.3:
                feature_categories['medium_coverage'].append(feature_name)
            else:
                feature_categories['low_coverage'].append(feature_name)

            if stats['std'] / max(stats['mean'], 0.001) >= 1.0:
                feature_categories['high_variance'].append(feature_name)
            else:
                feature_categories['low_variance'].append(feature_name)

        return {
            'total_features_analyzed': len(all_feature_names),
            'feature_statistics': feature_stats,
            'feature_categories': feature_categories,
            'avg_feature_coverage': np.mean([stats['coverage'] for stats in feature_stats.values()]),
            'avg_feature_std': np.mean([stats['std'] for stats in feature_stats.values()])
        }

    def run_ab_test(self, control_component, treatment_component, test_name: str,
                   component_type: str = 'text_classifier') -> ABTestResult:
        """
        Run A/B test comparing control vs treatment

        Args:
            control_component: Control/baseline component
            treatment_component: Treatment/enhanced component
            test_name: Name of the A/B test
            component_type: Type of component being tested

        Returns:
            A/B test result
        """
        logger.info(f"Running A/B test: {test_name}")

        # Test both components
        if component_type == 'text_classifier':
            control_metrics = self.validate_text_processing(control_component, f"{test_name}_control")
            treatment_metrics = self.validate_text_processing(treatment_component, f"{test_name}_treatment")
        elif component_type == 'safe_features':
            control_metrics = self.validate_feature_engineering(control_component, f"{test_name}_control")
            treatment_metrics = self.validate_feature_engineering(treatment_component, f"{test_name}_treatment")
        else:
            raise ValueError(f"Unknown component type: {component_type}")

        # Calculate improvements
        improvement_percentage = self._calculate_improvement(control_metrics, treatment_metrics)

        # Statistical significance (simplified)
        p_value = self._calculate_significance(control_metrics, treatment_metrics)
        significance_level = 0.05
        is_significant = p_value < significance_level

        # Generate recommendation
        recommendation = self._generate_recommendation(improvement_percentage, is_significant, component_type)

        # Create result object
        result = ABTestResult(
            test_name=test_name,
            control_metrics=self._dict_to_metrics(control_metrics),
            treatment_metrics=self._dict_to_metrics(treatment_metrics),
            significance_level=significance_level,
            p_value=p_value,
            is_significant=is_significant,
            improvement_percentage=improvement_percentage,
            recommendation=recommendation
        )

        # Store result
        self.ab_test_history.append(result)

        logger.info(f"A/B test completed: {test_name} - Improvement: {improvement_percentage:.2f}%, "
                   f"Significant: {is_significant}")

        return result

    def _calculate_improvement(self, control_metrics: Dict[str, Any], treatment_metrics: Dict[str, Any]) -> float:
        """
        Calculate percentage improvement between control and treatment

        Args:
            control_metrics: Control metrics
            treatment_metrics: Treatment metrics

        Returns:
            Improvement percentage
        """
        if 'f1_score' in control_metrics and 'f1_score' in treatment_metrics:
            # Primary metric: F1 score
            control_f1 = control_metrics['f1_score']
            treatment_f1 = treatment_metrics['f1_score']

            if control_f1 > 0:
                improvement = ((treatment_f1 - control_f1) / control_f1) * 100
            else:
                improvement = treatment_f1 * 100  # If baseline is 0

            # Consider processing time as secondary factor
            if 'avg_processing_time_ms' in control_metrics and 'avg_processing_time_ms' in treatment_metrics:
                time_improvement = ((control_metrics['avg_processing_time_ms'] - treatment_metrics['avg_processing_time_ms']) /
                                   max(control_metrics['avg_processing_time_ms'], 1)) * 100

                # Weighted improvement (70% F1, 30% time)
                improvement = 0.7 * improvement + 0.3 * time_improvement

        elif 'total_unique_features' in control_metrics and 'total_unique_features' in treatment_metrics:
            # For feature engineering: feature count improvement
            control_features = control_metrics['total_unique_features']
            treatment_features = treatment_metrics['total_unique_features']

            if control_features > 0:
                improvement = ((treatment_features - control_features) / control_features) * 100
            else:
                improvement = treatment_features

        else:
            improvement = 0.0

        return improvement

    def _calculate_significance(self, control_metrics: Dict[str, Any], treatment_metrics: Dict[str, Any]) -> float:
        """
        Calculate statistical significance (simplified version)

        Args:
            control_metrics: Control metrics
            treatment_metrics: Treatment metrics

        Returns:
            P-value (simplified)
        """
        # Simplified significance calculation
        # In a real implementation, you would use proper statistical tests

        if 'f1_score' in control_metrics and 'f1_score' in treatment_metrics:
            diff = abs(treatment_metrics['f1_score'] - control_metrics['f1_score'])

            # Rule of thumb: difference > 0.05 is likely significant
            if diff > 0.05:
                return 0.01  # Highly significant
            elif diff > 0.02:
                return 0.05  # Significant
            else:
                return 0.20  # Not significant
        else:
            return 0.10  # Default moderate significance

    def _generate_recommendation(self, improvement: float, is_significant: bool, component_type: str) -> str:
        """
        Generate recommendation based on test results

        Args:
            improvement: Improvement percentage
            is_significant: Whether results are statistically significant
            component_type: Type of component tested

        Returns:
            Recommendation string
        """
        if improvement > 10 and is_significant:
            return f"STRONGLY RECOMMENDED: Deploy enhanced {component_type} - {improvement:.1f}% improvement with statistical significance"
        elif improvement > 5 and is_significant:
            return f"RECOMMENDED: Deploy enhanced {component_type} - {improvement:.1f}% improvement with statistical significance"
        elif improvement > 0:
            return f"CONSIDER: Enhanced {component_type} shows {improvement:.1f}% improvement but may need further validation"
        elif improvement < -5:
            return f"NOT RECOMMENDED: Enhanced {component_type} performs worse than baseline ({improvement:.1f}% decline)"
        else:
            return f"NEUTRAL: Enhanced {component_type} shows minimal difference from baseline"

    def _dict_to_metrics(self, metrics_dict: Dict[str, Any]) -> PerformanceMetrics:
        """
        Convert metrics dictionary to PerformanceMetrics object

        Args:
            metrics_dict: Metrics dictionary

        Returns:
            PerformanceMetrics object
        """
        return PerformanceMetrics(
            accuracy=metrics_dict.get('accuracy', 0.0),
            precision=metrics_dict.get('precision', 0.0),
            recall=metrics_dict.get('recall', 0.0),
            f1_score=metrics_dict.get('f1_score', 0.0),
            auc_roc=metrics_dict.get('auc_roc', 0.0),
            specificity=metrics_dict.get('specificity', 0.0),
            false_positive_rate=metrics_dict.get('false_positive_rate', 0.0),
            false_negative_rate=metrics_dict.get('false_negative_rate', 0.0),
            processing_time=metrics_dict.get('avg_processing_time_ms', 0.0) / 1000,
            memory_usage=metrics_dict.get('memory_usage_mb', 0.0)
        )

    def run_benchmark(self, component, component_name: str, test_iterations: int = 100) -> BenchmarkResult:
        """
        Run performance benchmark for a component

        Args:
            component: Component to benchmark
            component_name: Name of component
            test_iterations: Number of test iterations

        Returns:
            Benchmark result
        """
        logger.info(f"Running benchmark: {component_name} ({test_iterations} iterations)")

        processing_times = []
        memory_usages = []
        success_count = 0
        error_count = 0

        test_sample = self.test_data[0] if self.test_data else {}

        for i in range(test_iterations):
            try:
                # Monitor memory before
                memory_before = self.memory_manager.check_memory_usage().get('current_usage_mb', 0)

                # Run component
                start_time = time.time()

                if hasattr(component, 'classify_claim'):
                    # Text classifier
                    component.classify_claim(test_sample.get('description', ''), test_sample)
                elif hasattr(component, 'generate_comprehensive_features'):
                    # Feature engineer
                    component.generate_comprehensive_features(test_sample)
                else:
                    raise ValueError(f"Unknown component type for benchmarking: {type(component)}")

                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Monitor memory after
                memory_after = self.memory_manager.check_memory_usage().get('current_usage_mb', 0)
                memory_usages.append(memory_after)

                success_count += 1

            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed: {e}")
                error_count += 1

        # Calculate benchmark metrics
        result = BenchmarkResult(
            component_name=component_name,
            test_data_size=test_iterations,
            avg_processing_time=np.mean(processing_times) if processing_times else 0,
            max_processing_time=np.max(processing_times) if processing_times else 0,
            min_processing_time=np.min(processing_times) if processing_times else 0,
            memory_peak=np.max(memory_usages) if memory_usages else 0,
            memory_average=np.mean(memory_usages) if memory_usages else 0,
            throughput_per_second=success_count / max(sum(processing_times), 1),
            success_rate=success_count / test_iterations,
            error_rate=error_count / test_iterations
        )

        # Store result
        self.benchmark_results[component_name] = result

        logger.info(f"Benchmark completed: {component_name} - {result.avg_processing_time:.3f}s avg, "
                   f"{result.throughput_per_second:.1f} req/sec")

        return result

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report

        Returns:
            Performance report
        """
        real_time_stats = self.metrics_collector.get_real_time_stats()

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_components_tested': len(self.current_metrics),
                'total_ab_tests': len(self.ab_test_history),
                'total_benchmarks': len(self.benchmark_results),
                'session_stats': real_time_stats
            },
            'current_metrics': self.current_metrics,
            'ab_test_history': [asdict(result) for result in self.ab_test_history],
            'benchmark_results': {name: asdict(result) for name, result in self.benchmark_results.items()},
            'aiml_targets': self.aiml_targets,
            'target_comparison': self._compare_with_targets()
        }

        return report

    def _compare_with_targets(self) -> Dict[str, Any]:
        """
        Compare current performance with AIML targets

        Returns:
            Target comparison results
        """
        comparison = {}

        for component_name, metrics in self.current_metrics.items():
            component_comparison = {}

            # F1 score target
            if 'f1_score' in metrics:
                f1_target = self.aiml_targets['text_classification_f1']
                component_comparison['f1_score'] = {
                    'current': metrics['f1_score'],
                    'target': f1_target,
                    'meets_target': metrics['f1_score'] >= f1_target,
                    'gap': metrics['f1_score'] - f1_target
                }

            # Processing time target
            if 'avg_processing_time_ms' in metrics:
                time_target = self.aiml_targets['processing_time_ms']
                component_comparison['processing_time'] = {
                    'current': metrics['avg_processing_time_ms'],
                    'target': time_target,
                    'meets_target': metrics['avg_processing_time_ms'] <= time_target,
                    'difference': metrics['avg_processing_time_ms'] - time_target
                }

            # Feature count target
            if 'total_unique_features' in metrics:
                feature_target = self.aiml_targets['feature_count']
                component_comparison['feature_count'] = {
                    'current': metrics['total_unique_features'],
                    'target': feature_target,
                    'meets_target': metrics['total_unique_features'] >= feature_target,
                    'surplus': metrics['total_unique_features'] - feature_target
                }

            # Memory usage target
            if 'memory_usage_mb' in metrics:
                memory_target = self.aiml_targets['memory_usage_mb']
                component_comparison['memory_usage'] = {
                    'current': metrics['memory_usage_mb'],
                    'target': memory_target,
                    'meets_target': metrics['memory_usage_mb'] <= memory_target,
                    'under_limit': memory_target - metrics['memory_usage_mb']
                }

            comparison[component_name] = component_comparison

        return comparison

    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for real-time performance dashboard

        Returns:
            Dashboard data
        """
        real_time_stats = self.metrics_collector.get_real_time_stats()

        return {
            'real_time_metrics': real_time_stats,
            'current_performance': self.current_metrics,
            'recent_ab_tests': [asdict(result) for result in self.ab_test_history[-5:]],  # Last 5 tests
            'system_health': {
                'memory_status': 'healthy' if real_time_stats['current_memory_mb'] < 800 else 'warning',
                'error_rate_status': 'healthy' if real_time_stats['error_rate'] < 0.05 else 'warning',
                'performance_status': 'healthy' if real_time_stats['avg_processing_time_ms'] < 1000 else 'warning'
            }
        }

# Global instance
_performance_validator = None

def get_performance_validator(test_data_path: Optional[str] = None) -> PerformanceValidator:
    """Get or create singleton instance"""
    global _performance_validator
    if _performance_validator is None:
        _performance_validator = PerformanceValidator(test_data_path)
    return _performance_validator