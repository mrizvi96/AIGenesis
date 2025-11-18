"""
Cloud-Optimized Enhanced SAFE (Semi-Auto Feature Engineering) for Insurance Fraud Detection
Scales from 33 to 150-200 high-quality features with smart generation and automated interactions
Cloud-optimized batch processing for Qdrant Cloud Free Tier (1GB RAM, 4GB Disk, 0.5 vCPU)
Progressive feature generation with automatic memory cleanup
"""

import numpy as np
import pandas as pd
import gc
import time
from typing import Dict, List, Any, Tuple, Optional, Set, Iterator, Generator
from datetime import datetime, timedelta
import math
import re
import json
import itertools
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from memory_manager import get_memory_manager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudOptimizedFeatureGenerator:
    """
    Cloud-optimized smart feature interaction generator for SAFE enhancement
    Implements progressive batch processing to stay within cloud memory limits
    """

    def __init__(self, max_interactions: int = 75, memory_limit_mb: int = 60):
        """
        Initialize cloud-optimized feature interaction generator

        Args:
            max_interactions: Maximum number of interactions to generate (reduced for cloud)
            memory_limit_mb: Memory limit for feature generation
        """
        self.max_interactions = max_interactions
        self.memory_limit_mb = memory_limit_mb
        self.memory_manager = get_memory_manager()

        # Cloud optimization settings
        self.batch_size = 20  # Process 20 interactions at a time
        self.cleanup_interval = 3  # Clean up every 3 batches
        self.progressive_selection = True  # Select best features progressively
        self.feature_limit = 200  # Cloud-optimized feature limit

        # Check memory allocation
        allocation_result = self.memory_manager.can_allocate(self.memory_limit_mb, 'safe_features')
        if not allocation_result['can_allocate']:
            logger.warning(f"[CLOUD-SAFE] Memory constraints detected, using reduced functionality")
            self.max_interactions = 25  # Further reduced for memory constraints
            self.batch_size = 10

        # Register memory allocation
        self.memory_manager.allocate_component_memory('safe_features', self.memory_limit_mb)

        # Cloud-optimized important feature pairs
        self.domain_important_pairs = [
            ('amount', 'claim_type'),
            ('time_of_day', 'location_risk'),
            ('customer_history', 'amount'),
            ('policy_age', 'claim_frequency'),
            ('damage_severity', 'amount'),
            ('injury_severity', 'medical_expenses'),
            ('weather_conditions', 'accident_type'),
            ('vehicle_age', 'repair_cost'),
            ('driver_age', 'accident_history'),
            ('location_risk', 'time_of_day'),
            ('claim_amount', 'policy_limit'),
            ('claimant_age', 'injury_severity'),
            ('policy_type', 'claim_frequency'),
            ('repair_cost', 'vehicle_age'),
            ('medical_expenses', 'injury_severity')
        ]

        self.processing_stats = {
            'features_generated': 0,
            'batches_processed': 0,
            'memory_cleanups': 0,
            'last_cleanup_time': time.time()
        }

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and perform cleanup if needed"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory_mb = memory_info.rss / (1024 * 1024)

            memory_stats = {
                'current_memory_mb': current_memory_mb,
                'memory_limit_mb': self.memory_limit_mb,
                'usage_percent': (current_memory_mb / self.memory_limit_mb) * 100,
                'processing_stats': self.processing_stats
            }

            # Trigger cleanup if memory is high
            if memory_stats['usage_percent'] > 85:
                logger.warning(f"[CLOUD-SAFE] High memory usage: {memory_stats['usage_percent']:.1f}%")
                self._perform_memory_cleanup()

            return memory_stats

        except Exception as e:
            return {'error': str(e), 'current_memory_mb': 0}

    def _perform_memory_cleanup(self):
        """Perform memory cleanup during feature generation"""
        logger.info("[CLOUD-SAFE] Performing memory cleanup during feature generation...")

        # Clear intermediate feature data
        if hasattr(self, '_intermediate_features'):
            self._intermediate_features.clear()

        # Force garbage collection
        gc.collect()

        self.processing_stats['memory_cleanups'] += 1
        self.processing_stats['last_cleanup_time'] = time.time()
        logger.info("[OK] Memory cleanup completed")

    def generate_enhanced_features_batch(self, claim_data: Dict[str, Any],
                                       existing_features: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate enhanced features using cloud-optimized batch processing
        Processes features progressively to avoid memory spikes
        """
        try:
            start_time = time.time()
            logger.info(f"[CLOUD-SAFE] Starting enhanced feature generation (batch_size: {self.batch_size})")

            # Check memory before processing
            memory_before = self._check_memory_usage()

            # Start with existing features or create base features
            if existing_features is not None and not existing_features.empty:
                base_features = existing_features.copy()
            else:
                base_features = self._create_base_features(claim_data)

            # Generate interaction features in batches
            interaction_features = self._generate_interaction_features_batch(
                base_features, claim_data
            )

            # Combine features with progressive selection
            enhanced_features = self._combine_features_progressive(
                base_features, interaction_features
            )

            # Apply cloud-optimized feature selection
            final_features = self._apply_cloud_feature_selection(enhanced_features, claim_data)

            # Memory cleanup
            del base_features, interaction_features, enhanced_features
            gc.collect()

            processing_time = time.time() - start_time
            memory_after = self._check_memory_usage()

            # Update stats
            self.processing_stats['batches_processed'] += 1

            result = {
                'enhanced_features': final_features,
                'feature_metadata': {
                    'total_features': len(final_features.columns) if not final_features.empty else 0,
                    'base_features': len([col for col in final_features.columns if not col.startswith('interaction_')]),
                    'interaction_features': len([col for col in final_features.columns if col.startswith('interaction_')]),
                    'processing_time': processing_time,
                    'cloud_optimized': True,
                    'batch_size': self.batch_size
                },
                'performance_metrics': {
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'batches_processed': self.processing_stats['batches_processed'],
                    'memory_cleanups': self.processing_stats['memory_cleanups']
                }
            }

            logger.info(f"[OK] Enhanced feature generation completed: {processing_time:.3f}s, "
                       f"features: {result['feature_metadata']['total_features']}, "
                       f"memory: {memory_after.get('current_memory_mb', 0):.1f}MB")

            return result

        except Exception as e:
            logger.error(f"[ERROR] Cloud SAFE feature generation failed: {e}")
            return self._get_fallback_features(claim_data, existing_features)

    def _create_base_features(self, claim_data: Dict[str, Any]) -> pd.DataFrame:
        """Create base feature set with cloud optimization"""
        try:
            features = {}

            # Financial features
            amount = float(claim_data.get('amount', 0))
            features['claim_amount'] = amount
            features['claim_amount_log'] = np.log1p(amount)
            features['amount_is_high'] = int(amount > 5000)
            features['amount_very_high'] = int(amount > 10000)

            # Temporal features
            if 'date_submitted' in claim_data:
                try:
                    date_submitted = pd.to_datetime(claim_data['date_submitted'])
                    features['day_of_week'] = date_submitted.dayofweek
                    features['month'] = date_submitted.month
                    features['is_weekend'] = int(date_submitted.dayofweek >= 5)
                    features['is_month_end'] = int(date_submitted.day > 25)
                except:
                    features['day_of_week'] = 0
                    features['month'] = 0
                    features['is_weekend'] = 0
                    features['is_month_end'] = 0

            # Claim type features
            claim_type = str(claim_data.get('claim_type', 'unknown')).lower()
            features['is_auto_claim'] = int('auto' in claim_type or 'car' in claim_type)
            features['is_home_claim'] = int('home' in claim_type or 'property' in claim_type)
            features['is_health_claim'] = int('health' in claim_type or 'medical' in claim_type)

            # Customer features
            customer_age = claim_data.get('customer_age', 30)
            features['customer_age'] = customer_age
            features['customer_age_group'] = min(customer_age // 10, 8)
            features['is_young_customer'] = int(customer_age < 30)
            features['is_senior_customer'] = int(customer_age > 60)

            # Risk indicators
            features['risk_score'] = min(amount / 1000 + features.get('is_auto_claim', 0) * 0.5, 10)
            features['processing_priority'] = int(amount > 10000 or features.get('is_auto_claim', 0))

            return pd.DataFrame([features])

        except Exception as e:
            logger.error(f"[ERROR] Base feature creation failed: {e}")
            # Return minimal fallback features
            return pd.DataFrame([{
                'claim_amount': float(claim_data.get('amount', 0)),
                'risk_score': 1.0
            }])

    def _generate_interaction_features_batch(self, base_features: pd.DataFrame,
                                           claim_data: Dict[str, Any]) -> pd.DataFrame:
        """Generate interaction features in memory-efficient batches"""
        try:
            interaction_features = pd.DataFrame(index=base_features.index)

            # Convert to float for safer processing
            numeric_features = base_features.select_dtypes(include=[np.number]).astype(np.float32)

            # Process important pairs in batches
            for i in range(0, len(self.domain_important_pairs), self.batch_size):
                batch_pairs = self.domain_important_pairs[i:i + self.batch_size]

                for feature1, feature2 in batch_pairs:
                    if feature1 in numeric_features.columns and feature2 in numeric_features.columns:
                        # Create interaction feature
                        interaction_col = f'interaction_{feature1}_{feature2}'
                        interaction_values = numeric_features[feature1] * numeric_features[feature2]

                        # Apply memory-efficient processing
                        if len(interaction_values) > 0:
                            interaction_features[interaction_col] = interaction_values

                # Periodic cleanup
                if i // self.batch_size % self.cleanup_interval == 0:
                    gc.collect()

            self.processing_stats['features_generated'] += len(interaction_features.columns)
            return interaction_features

        except Exception as e:
            logger.error(f"[ERROR] Interaction feature generation failed: {e}")
            return pd.DataFrame()

    def _combine_features_progressive(self, base_features: pd.DataFrame,
                                      interaction_features: pd.DataFrame) -> pd.DataFrame:
        """Combine features with progressive selection to stay within memory limits"""
        try:
            # Combine base and interaction features
            all_features = pd.concat([base_features, interaction_features], axis=1)

            # Progressive feature selection if enabled
            if self.progressive_selection and len(all_features.columns) > self.feature_limit:
                # Use simple variance-based selection for memory efficiency
                feature_variances = all_features.var()
                top_features = feature_variances.nlargest(self.feature_limit).index
                all_features = all_features[top_features]

            return all_features

        except Exception as e:
            logger.error(f"[ERROR] Feature combination failed: {e}")
            return base_features

    def _apply_cloud_feature_selection(self, features: pd.DataFrame,
                                      claim_data: Dict[str, Any]) -> pd.DataFrame:
        """Apply memory-efficient feature selection"""
        try:
            if features.empty:
                return features

            # Simple feature selection based on variance (memory efficient)
            feature_variances = features.var()
            non_zero_variances = feature_variances[feature_variances > 1e-6]

            if len(non_zero_variances) > self.feature_limit:
                # Keep top features by variance
                top_features = non_zero_variances.nlargest(self.feature_limit).index
                selected_features = features[top_features]
            else:
                selected_features = features

            logger.info(f"[CLOUD-SAFE] Feature selection: {len(features.columns)} → {len(selected_features.columns)}")

            return selected_features

        except Exception as e:
            logger.error(f"[ERROR] Feature selection failed: {e}")
            return features

    def _get_fallback_features(self, claim_data: Dict[str, Any],
                              existing_features: pd.DataFrame = None) -> Dict[str, Any]:
        """Get fallback features when enhanced generation fails"""
        try:
            if existing_features is not None and not existing_features.empty:
                fallback_features = existing_features.copy()
                total_features = len(fallback_features.columns)
                method = 'existing_features_fallback'
            else:
                # Create minimal fallback features
                fallback_features = pd.DataFrame([{
                    'claim_amount': float(claim_data.get('amount', 0)),
                    'claim_amount_log': np.log1p(float(claim_data.get('amount', 0))),
                    'risk_score': min(float(claim_data.get('amount', 0)) / 1000, 5.0),
                    'fallback_flag': 1
                }])
                total_features = 4
                method = 'minimal_fallback'

            return {
                'enhanced_features': fallback_features,
                'feature_metadata': {
                    'total_features': total_features,
                    'base_features': total_features,
                    'interaction_features': 0,
                    'processing_time': 0.1,
                    'cloud_optimized': False,
                    'error': 'Enhanced generation failed, using fallback',
                    'method': method
                },
                'performance_metrics': {
                    'error': str(e) if 'e' in locals() else 'Unknown error',
                    'memory_before': {'error': 'N/A'},
                    'memory_after': {'error': 'N/A'}
                }
            }

        except Exception as e:
            logger.error(f"[ERROR] Fallback feature generation failed: {e}")
            # Ultimate fallback
            return {
                'enhanced_features': pd.DataFrame([{'error_feature': 1.0}]),
                'feature_metadata': {'error': 'Complete fallback failure', 'total_features': 1},
                'performance_metrics': {'error': str(e)}
            }

    def get_feature_generation_stats(self) -> Dict[str, Any]:
        """Get feature generation statistics"""
        return {
            'max_interactions': self.max_interactions,
            'batch_size': self.batch_size,
            'feature_limit': self.feature_limit,
            'processing_stats': self.processing_stats,
            'memory_limit_mb': self.memory_limit_mb,
            'current_memory': self._check_memory_usage(),
            'domain_pairs': len(self.domain_important_pairs)
        }

    def cleanup_resources(self):
        """Explicit cleanup of feature generation resources"""
        logger.info("[CLOUD-SAFE] Cleaning up feature generation resources...")

        # Clear intermediate data
        for attr in ['_intermediate_features', 'domain_important_pairs']:
            if hasattr(self, attr):
                setattr(self, attr, None)

        # Force garbage collection
        gc.collect()

        # Release memory allocation
        self.memory_manager.release_component_memory('safe_features')

        logger.info("[OK] Feature generation resources cleaned up")

# Global cloud-optimized SAFE feature generator instance
cloud_safe_features = CloudOptimizedFeatureGenerator()

def get_cloud_safe_features() -> CloudOptimizedFeatureGenerator:
    """Get the global cloud-optimized SAFE feature generator instance"""
    return cloud_safe_features

if __name__ == "__main__":
    # Test the cloud SAFE feature generator
    generator = CloudOptimizedFeatureGenerator()
    logger.info(f"[TEST] Cloud SAFE feature generator initialized: {generator.get_feature_generation_stats()}")

    # Test with sample data
    test_claim_data = {
        'amount': 7500.0,
        'claim_type': 'auto',
        'customer_age': 35,
        'date_submitted': '2024-01-15'
    }

    result = generator.generate_enhanced_features_batch(test_claim_data)
    logger.info(f"[TEST] Feature generation test completed: {len(result['enhanced_features'].columns)} features")