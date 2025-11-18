"""
Cloud-Optimized Memory-Efficient Feature Fusion Architecture
BLOCK Tucker alternative for Qdrant Cloud resource constraints
Streaming fusion with progressive dimensionality reduction
Memory target: <80MB with automatic cleanup
Cloud-optimized: CPU-efficient algorithms for 0.5 vCPU limit
"""

import numpy as np
import gc
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Iterator, Generator
import json
import logging
from datetime import datetime
from memory_manager import get_memory_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudMemoryEfficientFusion:
    """
    Cloud-optimized multimodal feature fusion with streaming processing
    Implements progressive fusion strategy for Qdrant Cloud constraints
    Streaming architecture to avoid memory spikes
    """

    def __init__(self, memory_limit_mb: int = 80):
        """Initialize cloud-optimized fusion processor"""
        logger.info(f"[CLOUD-FUSION] Loading streaming fusion processor (memory_limit={memory_limit_mb}MB)...")

        # Cloud optimization settings
        self.memory_limit_mb = memory_limit_mb
        self.memory_manager = get_memory_manager()

        # Fusion configuration for cloud efficiency
        self.config = {
            'streaming_layers': 2,  # Reduced for cloud CPU constraints
            'dimensionality_reduction': True,
            'target_dimensions': 96,  # Further reduced for cloud efficiency
            'compression_ratio': 0.4,  # Aggressive compression for cloud
            'streaming_batch_size': 3,  # Process 3 features at a time
            'memory_cleanup_interval': 5,  # Clean up every 5 batches
            'progressive_fusion': True
        }

        # Initialize projection matrices (cloud-optimized)
        self._initialize_cloud_projections()

        # Cloud-specific state management
        self.memory_usage = 0.0
        self.processed_count = 0
        self.cleanup_counter = 0
        self.last_cleanup_time = time.time()

        # Check memory allocation
        allocation_result = self.memory_manager.can_allocate(self.memory_limit_mb, 'feature_engine')
        if not allocation_result['can_allocate']:
            logger.warning(f"[CLOUD-FUSION] Memory constraints detected, using reduced functionality")
            self.memory_limit_mb = 40  # Fallback to minimal memory usage

        # Register memory allocation
        self.memory_manager.allocate_component_memory('feature_engine', self.memory_limit_mb)

    def _initialize_cloud_projections(self):
        """Initialize projection matrices with cloud memory optimization"""
        try:
            # Cloud-optimized: smaller, sparse projection matrices
            target_dim = self.config['target_dimensions']

            # Initialize with smaller matrices for cloud efficiency
            self.text_projection = self._create_sparse_projection(384, target_dim)  # Assuming 384-dim text
            self.image_projection = self._create_sparse_projection(256, target_dim)  # Assuming 256-dim image
            self.audio_projection = self._create_sparse_projection(128, target_dim)  # Assuming 128-dim audio

            # Fusion weights (very small memory footprint)
            self.fusion_weights = np.array([0.4, 0.35, 0.25])  # Text, Image, Audio priorities

            # Progressive fusion matrices (smaller than traditional)
            self.progressive_matrices = [
                self._create_sparse_projection(target_dim * 3, target_dim * 2),
                self._create_sparse_projection(target_dim * 2, target_dim)
            ]

            logger.info(f"[OK] Cloud projection matrices initialized (target_dim: {target_dim})")

        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize projections: {e}")
            # Fallback to identity projections
            self._create_fallback_projections()

    def _create_sparse_projection(self, input_dim: int, output_dim: int) -> np.ndarray:
        """Create memory-efficient sparse projection matrix"""
        # Create a sparse-like matrix with only 20% non-zero elements
        projection = np.zeros((input_dim, output_dim), dtype=np.float32)

        # Randomly select connections (20% sparsity for memory efficiency)
        num_connections = int(input_dim * output_dim * 0.2)
        for _ in range(num_connections):
            i = np.random.randint(0, input_dim)
            j = np.random.randint(0, output_dim)
            projection[i, j] = np.random.normal(0, 0.1)

        return projection

    def _create_fallback_projections(self):
        """Create minimal fallback projections for memory constraints"""
        target_dim = self.config['target_dimensions'] // 2  # Even smaller fallback
        self.text_projection = np.eye(384)[:, :target_dim]  # Simple selection
        self.image_projection = np.eye(256)[:, :target_dim]
        self.audio_projection = np.eye(128)[:, :target_dim]
        self.fusion_weights = np.array([0.5, 0.3, 0.2])  # Simplified weights
        self.progressive_matrices = []  # No progressive fusion in fallback
        logger.warning("[FALLBACK] Using minimal projections due to memory constraints")

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
                'processed_count': self.processed_count,
                'cleanup_counter': self.cleanup_counter
            }

            # Trigger cleanup if memory is high
            if memory_stats['usage_percent'] > 85:
                logger.warning(f"[CLOUD-FUSION] High memory usage: {memory_stats['usage_percent']:.1f}%")
                self._perform_emergency_cleanup()

            return memory_stats

        except Exception as e:
            return {'error': str(e), 'current_memory_mb': 0}

    def _perform_emergency_cleanup(self):
        """Perform emergency memory cleanup"""
        logger.info("[CLOUD-FUSION] Performing emergency memory cleanup...")

        # Clear caches and intermediate results
        if hasattr(self, '_intermediate_cache'):
            self._intermediate_cache.clear()

        # Force garbage collection
        gc.collect()

        self.cleanup_counter += 1
        self.last_cleanup_time = time.time()
        logger.info("[OK] Emergency cleanup completed")

    def fuse_features_streaming(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cloud-optimized streaming feature fusion
        Processes features in small batches to avoid memory spikes
        """
        try:
            start_time = time.time()
            self.processed_count += 1

            # Check memory before processing
            memory_before = self._check_memory_usage()

            # Extract features in streaming manner
            feature_stream = self._extract_feature_stream(claim_data)

            # Process features in small batches
            fused_features = []
            for batch_features in self._process_feature_stream(feature_stream):
                batch_result = self._fuse_feature_batch(batch_features)
                fused_features.extend(batch_result)

                # Periodic cleanup
                if self.processed_count % self.config['memory_cleanup_interval'] == 0:
                    self._perform_emergency_cleanup()

            # Final fusion step
            final_fusion = self._final_fusion_step(fused_features)

            # Memory cleanup
            del fused_features
            gc.collect()

            processing_time = time.time() - start_time
            memory_after = self._check_memory_usage()

            result = {
                'fused_features': final_fusion,
                'fusion_metadata': {
                    'processing_time': processing_time,
                    'target_dimensions': self.config['target_dimensions'],
                    'compression_ratio': self.config['compression_ratio'],
                    'batches_processed': len(feature_stream) // self.config['streaming_batch_size'],
                    'memory_optimization': True
                },
                'performance_metrics': {
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'processed_count': self.processed_count,
                    'cleanup_count': self.cleanup_counter
                }
            }

            logger.info(f"[CLOUD-FUSION] Streaming fusion completed: {processing_time:.3f}s, "
                       f"dim: {len(final_fusion)}, memory: {memory_after.get('current_memory_mb', 0):.1f}MB")

            return result

        except Exception as e:
            logger.error(f"[ERROR] Cloud fusion failed: {e}")
            return self._get_fallback_fusion(claim_data)

    def _extract_feature_stream(self, claim_data: Dict[str, Any]) -> Generator[List[np.ndarray], None, None]:
        """Extract features in a streaming manner to avoid memory spikes"""
        # Extract text features
        text_features = claim_data.get('text_features', [])
        if text_features:
            yield [np.array(text_features, dtype=np.float32)]

        # Extract image features (streaming)
        image_features = claim_data.get('image_features', [])
        for i in range(0, len(image_features), self.config['streaming_batch_size']):
            batch = image_features[i:i + self.config['streaming_batch_size']]
            if batch:
                yield [np.array(img_feat, dtype=np.float32) for img_feat in batch]

        # Extract audio features (streaming)
        audio_features = claim_data.get('audio_features', [])
        if audio_features:
            yield [np.array(audio_features, dtype=np.float32)]

        # Extract other modalities
        other_features = claim_data.get('other_features', [])
        for i in range(0, len(other_features), self.config['streaming_batch_size']):
            batch = other_features[i:i + self.config['streaming_batch_size']]
            if batch:
                yield [np.array(other_feat, dtype=np.float32) for other_feat in batch]

    def _process_feature_stream(self, feature_stream: Generator[List[np.ndarray], None, None]) -> Generator[List[np.ndarray], None, None]:
        """Process feature stream with dimensionality reduction"""
        for batch_features in feature_stream:
            processed_batch = []

            for i, features in enumerate(batch_features):
                try:
                    # Choose projection based on feature type and size
                    projection = self._choose_projection(features)

                    # Apply dimensionality reduction
                    if projection is not None:
                        reduced_features = np.dot(features, projection)
                        processed_batch.append(reduced_features)
                    else:
                        # Use the features as-is if no projection available
                        processed_batch.append(features[:self.config['target_dimensions']])

                except Exception as e:
                    logger.warning(f"[WARN] Feature processing failed: {e}")
                    # Fallback: truncate features
                    processed_batch.append(features[:self.config['target_dimensions']])

            yield processed_batch

    def _choose_projection(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Choose appropriate projection based on feature dimensions"""
        feature_dim = features.shape[-1] if len(features.shape) > 0 else 0

        # Choose projection based on feature size
        if feature_dim >= 300:  # Text features
            return self.text_projection
        elif feature_dim >= 200:  # Image features
            return self.image_projection
        elif feature_dim >= 100:  # Audio features
            return self.audio_projection
        else:
            # For small features, no projection needed
            return None

    def _fuse_feature_batch(self, batch_features: List[np.ndarray]) -> List[np.ndarray]:
        """Fuse a batch of features using cloud-optimized fusion"""
        if not batch_features:
            return []

        try:
            # Stack features for batch processing
            stacked_features = np.stack(batch_features, axis=0)

            # Apply weighted fusion
            if len(self.fusion_weights) >= len(batch_features):
                weights = self.fusion_weights[:len(batch_features)]
                weights = weights / np.sum(weights)  # Normalize weights
                fused_batch = np.average(stacked_features, axis=0, weights=weights)
            else:
                # Simple average if not enough weights
                fused_batch = np.mean(stacked_features, axis=0)

            # Apply progressive fusion if enabled
            if self.config['progressive_fusion'] and self.progressive_matrices:
                for i, matrix in enumerate(self.progressive_matrices):
                    try:
                        fused_batch = np.dot(fused_batch, matrix)
                    except Exception as e:
                        logger.warning(f"[WARN] Progressive fusion step {i} failed: {e}")
                        break

            return [fused_batch]

        except Exception as e:
            logger.error(f"[ERROR] Batch fusion failed: {e}")
            # Fallback: return the first feature
            return [batch_features[0]]

    def _final_fusion_step(self, fused_features: List[np.ndarray]) -> np.ndarray:
        """Final fusion step to produce unified features"""
        if not fused_features:
            return np.zeros(self.config['target_dimensions'], dtype=np.float32)

        try:
            # Stack all fused features
            all_features = np.vstack(fused_features)

            # Final aggregation with attention to avoid memory spikes
            if len(all_features) > 10:  # Large number of features
                # Use attention-based aggregation for large feature sets
                attention_weights = np.ones(len(all_features)) / len(all_features)
                final_features = np.average(all_features, axis=0, weights=attention_weights)
            else:
                # Simple mean for smaller feature sets
                final_features = np.mean(all_features, axis=0)

            # Ensure final dimensions
            if len(final_features) > self.config['target_dimensions']:
                final_features = final_features[:self.config['target_dimensions']]
            elif len(final_features) < self.config['target_dimensions']:
                # Pad with zeros if too small
                padding_size = self.config['target_dimensions'] - len(final_features)
                final_features = np.pad(final_features, (0, padding_size), 'constant')

            return final_features.astype(np.float32)

        except Exception as e:
            logger.error(f"[ERROR] Final fusion step failed: {e}")
            # Fallback: return zeros of target dimension
            return np.zeros(self.config['target_dimensions'], dtype=np.float32)

    def _get_fallback_fusion(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback fusion result when streaming fails"""
        try:
            # Simple fallback: concatenate and truncate features
            all_features = []

            # Extract available features
            for feature_key in ['text_features', 'image_features', 'audio_features', 'other_features']:
                features = claim_data.get(feature_key, [])
                if features:
                    if isinstance(features, list) and len(features) > 0:
                        all_features.extend([np.array(f, dtype=np.float32) for f in features])

            if all_features:
                # Concatenate and truncate
                concatenated = np.concatenate(all_features, axis=0)
                target_dim = self.config['target_dimensions'] // 2  # Smaller fallback
                if len(concatenated) > target_dim:
                    final_features = concatenated[:target_dim]
                else:
                    final_features = np.pad(concatenated, (0, target_dim - len(concatenated)), 'constant')
            else:
                # No features available
                final_features = np.zeros(self.config['target_dimensions'] // 2, dtype=np.float32)

            return {
                'fused_features': final_features,
                'fusion_metadata': {
                    'processing_method': 'fallback',
                    'target_dimensions': len(final_features),
                    'error': 'Streaming fusion failed, used fallback'
                },
                'performance_metrics': {
                    'memory_before': {'error': 'N/A'},
                    'memory_after': {'error': 'N/A'},
                    'processed_count': 0,
                    'cleanup_count': 0
                }
            }

        except Exception as e:
            logger.error(f"[ERROR] Fallback fusion also failed: {e}")
            return {
                'fused_features': np.zeros(32, dtype=np.float32),  # Minimal fallback
                'fusion_metadata': {'error': str(e), 'processing_method': 'minimal_fallback'},
                'performance_metrics': {'error': 'Complete fusion failure'}
            }

    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion processing statistics"""
        return {
            'processed_count': self.processed_count,
            'cleanup_count': self.cleanup_counter,
            'memory_limit_mb': self.memory_limit_mb,
            'config': self.config,
            'current_memory': self._check_memory_usage(),
            'projection_matrices_available': hasattr(self, 'text_projection')
        }

    def cleanup_resources(self):
        """Explicit cleanup of fusion resources"""
        logger.info("[CLOUD-FUSION] Cleaning up fusion resources...")

        # Clear projection matrices
        for attr in ['text_projection', 'image_projection', 'audio_projection', 'progressive_matrices']:
            if hasattr(self, attr):
                setattr(self, attr, None)

        # Force garbage collection
        gc.collect()

        # Release memory allocation
        self.memory_manager.release_component_memory('feature_engine')

        logger.info("[OK] Fusion resources cleaned up")

# Global cloud-optimized fusion instance
cloud_fusion_processor = CloudMemoryEfficientFusion()

def get_cloud_fusion_processor() -> CloudMemoryEfficientFusion:
    """Get the global cloud-optimized fusion processor instance"""
    return cloud_fusion_processor

if __name__ == "__main__":
    # Test the cloud fusion processor
    processor = CloudMemoryEfficientFusion()
    logger.info(f"[TEST] Cloud fusion processor initialized: {processor.get_fusion_stats()}")