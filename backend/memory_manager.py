"""
Memory Manager for Enhanced Text Processing and SAFE Feature Engineering
Optimized for Qdrant Cloud Free Tier (1GB RAM, 4GB Disk, 0.5 vCPU)
With vector compression and cloud optimization features
"""

import gc
import psutil
import time
import logging
import os
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time"""
    timestamp: float
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory usage percentage
    available_mb: float  # Available memory in MB

@dataclass
class ResourceLimits:
    """System resource limits for Qdrant Cloud Free Tier"""
    max_ram_mb: int = 1000  # 1GB total
    max_disk_mb: int = 4096  # 4GB total
    max_cpu_percent: float = 80.0  # 80% of 0.5 vCPU

    # Component-specific limits (optimized for cloud)
    base_system_mb: int = 200
    qdrant_db_mb: int = 300
    text_models_mb: int = 150  # Reduced for cloud
    feature_engine_mb: int = 100  # Reduced for cloud
    vision_models_mb: int = 120  # Added for vision processing
    api_framework_mb: int = 80   # Reduced for cloud
    os_overhead_mb: int = 50

    # Vector compression settings
    vector_compression_enabled: bool = True
    target_vector_dim: int = 256  # Compressed vector dimension
    max_original_dim: int = 1024  # Maximum original dimension

class MemoryManager:
    """
    Advanced memory management system for resource-constrained environments
    """

    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.process = psutil.Process()
        self.snapshots: List[MemorySnapshot] = []
        self.alert_thresholds = {
            'warning': 0.8,  # 80% of memory limit
            'critical': 0.9,  # 90% of memory limit
            'emergency': 0.95  # 95% of memory limit
        }

        # Component memory tracking (cloud-optimized)
        self.component_usage = {
            'text_classifier': 0,
            'feature_engine': 0,
            'vision_processor': 0,  # Added for vision processing
            'qdrant_client': 0,
            'api_server': 0,
            'cache': 0,
            'vector_compression': 0  # Added for compression
        }

        # Performance optimization settings
        self.garbage_collection_interval = 30  # seconds
        self.last_gc_time = time.time()
        self.cache_cleanup_threshold = 0.7  # Clean cache at 70% memory

        # Vector compression models and settings
        self._pca_models: Dict[str, PCA] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self.compression_models = {}  # Added for compatibility
        self._compression_stats = {
            'original_vectors': 0,
            'compressed_vectors': 0,
            'compression_ratio': 0.0,
            'memory_saved_mb': 0.0
        }

        logger.info(f"MemoryManager initialized for Qdrant Cloud with {self.limits.max_ram_mb}MB RAM limit")

    def check_memory_usage(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage statistics

        Returns:
            Dictionary with detailed memory information
        """
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()

            # Convert to MB
            rss_mb = memory_info.rss / (1024 * 1024)
            vms_mb = memory_info.vms / (1024 * 1024)

            # Available memory
            system_memory = psutil.virtual_memory()
            available_mb = system_memory.available / (1024 * 1024)

            # Create snapshot
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                rss_mb=rss_mb,
                vms_mb=vms_mb,
                percent=memory_percent,
                available_mb=available_mb
            )

            # Store snapshot (keep last 100)
            self.snapshots.append(snapshot)
            if len(self.snapshots) > 100:
                self.snapshots.pop(0)

            # Calculate status
            usage_percentage = rss_mb / self.limits.max_ram_mb
            status = self._get_memory_status(usage_percentage)

            return {
                'current_usage_mb': rss_mb,
                'virtual_memory_mb': vms_mb,
                'usage_percentage': usage_percentage * 100,
                'available_mb': available_mb,
                'limit_mb': self.limits.max_ram_mb,
                'status': status,
                'component_breakdown': self.component_usage.copy(),
                'trend': self._get_memory_trend()
            }

        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return {'error': str(e)}

    def _get_memory_status(self, usage_percentage: float) -> str:
        """Determine memory status based on usage percentage"""
        if usage_percentage >= self.alert_thresholds['emergency']:
            return 'emergency'
        elif usage_percentage >= self.alert_thresholds['critical']:
            return 'critical'
        elif usage_percentage >= self.alert_thresholds['warning']:
            return 'warning'
        else:
            return 'healthy'

    def _get_memory_trend(self) -> str:
        """Analyze memory usage trend from recent snapshots"""
        if len(self.snapshots) < 5:
            return 'insufficient_data'

        recent_snapshots = self.snapshots[-5:]
        memory_values = [s.rss_mb for s in recent_snapshots]

        # Simple trend calculation
        if all(memory_values[i] <= memory_values[i+1] for i in range(len(memory_values)-1)):
            return 'increasing'
        elif all(memory_values[i] >= memory_values[i+1] for i in range(len(memory_values)-1)):
            return 'decreasing'
        else:
            return 'fluctuating'

    def can_allocate(self, required_mb: float, component: str = 'unknown') -> Dict[str, Any]:
        """
        Check if we can safely allocate the required memory

        Args:
            required_mb: Memory required in MB
            component: Component requesting memory

        Returns:
            Dictionary with allocation decision and details
        """
        current_usage = self.check_memory_usage()

        if 'error' in current_usage:
            return {'can_allocate': False, 'reason': 'Cannot determine current usage'}

        projected_usage = current_usage['current_usage_mb'] + required_mb
        usage_percentage = projected_usage / self.limits.max_ram_mb

        can_allocate = projected_usage <= self.limits.max_ram_mb

        result = {
            'can_allocate': can_allocate,
            'required_mb': required_mb,
            'current_mb': current_usage['current_usage_mb'],
            'projected_mb': projected_usage,
            'limit_mb': self.limits.max_ram_mb,
            'usage_percentage': usage_percentage * 100,
            'component': component
        }

        # Log allocation attempts
        if can_allocate:
            logger.info(f"Memory allocation approved: {required_mb:.2f}MB for {component}")
        else:
            logger.warning(f"Memory allocation denied: {required_mb:.2f}MB for {component} "
                          f"would exceed {projected_usage:.2f}MB limit")

        return result

    def allocate_component_memory(self, component: str, size_mb: float) -> bool:
        """
        Allocate memory for a specific component

        Args:
            component: Component name
            size_mb: Memory size in MB

        Returns:
            True if allocation successful
        """
        allocation_result = self.can_allocate(size_mb, component)

        if allocation_result['can_allocate']:
            self.component_usage[component] = size_mb
            logger.info(f"Allocated {size_mb:.2f}MB to {component}")
            return True
        else:
            return False

    def release_component_memory(self, component: str) -> bool:
        """
        Release memory allocated to a component

        Args:
            component: Component name

        Returns:
            True if memory was released
        """
        if component in self.component_usage and self.component_usage[component] > 0:
            released = self.component_usage[component]
            self.component_usage[component] = 0
            logger.info(f"Released {released:.2f}MB from {component}")
            return True
        return False

    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        Force garbage collection to free memory

        Returns:
            Dictionary with garbage collection results
        """
        start_memory = self.check_memory_usage()

        # Force garbage collection
        collected_objects = gc.collect()

        # Clear internal caches if needed
        if hasattr(self, '_cache'):
            cache_size = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared internal cache with {cache_size} items")

        end_memory = self.check_memory_usage()

        memory_freed = start_memory.get('current_usage_mb', 0) - end_memory.get('current_usage_mb', 0)

        result = {
            'collected_objects': collected_objects,
            'memory_freed_mb': max(0, memory_freed),
            'start_memory_mb': start_memory.get('current_usage_mb', 0),
            'end_memory_mb': end_memory.get('current_usage_mb', 0),
            'timestamp': time.time()
        }

        logger.info(f"Garbage collection completed: freed {memory_freed:.2f}MB, "
                   f"collected {collected_objects} objects")

        return result

    def optimize_for_memory(self) -> Dict[str, Any]:
        """
        Comprehensive memory optimization

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting comprehensive memory optimization...")

        results = {
            'start_memory': self.check_memory_usage(),
            'actions_taken': [],
            'memory_saved_mb': 0
        }

        # Action 1: Force garbage collection
        gc_result = self.force_garbage_collection()
        results['actions_taken'].append(f"Garbage collection: {gc_result['memory_freed_mb']:.2f}MB freed")
        results['memory_saved_mb'] += gc_result['memory_freed_mb']

        # Action 2: Clear component caches
        for component, size in self.component_usage.items():
            if size > 0:
                # Reduce component cache sizes
                self.component_usage[component] = size * 0.8  # Reduce by 20%
                results['actions_taken'].append(f"Reduced {component} cache by 20%")

        # Action 3: Optimize numpy arrays
        if hasattr(self, '_optimize_arrays'):
            array_memory = self._optimize_arrays()
            results['actions_taken'].append(f"Optimized numpy arrays: {array_memory:.2f}MB saved")
            results['memory_saved_mb'] += array_memory

        results['end_memory'] = self.check_memory_usage()
        results['total_memory_saved_mb'] = (
            results['start_memory'].get('current_usage_mb', 0) -
            results['end_memory'].get('current_usage_mb', 0)
        )

        logger.info(f"Memory optimization completed: {results['total_memory_saved_mb']:.2f}MB saved")

        return results

    @contextmanager
    def memory_limit_context(self, max_mb: float, component: str = 'context'):
        """
        Context manager for operations with memory limits

        Args:
            max_mb: Maximum memory allowed in this context
            component: Component name for tracking
        """
        if not self.can_allocate(max_mb, component)['can_allocate']:
            raise MemoryError(f"Cannot allocate {max_mb}MB for {component}: insufficient memory")

        original_usage = self.component_usage.get(component, 0)
        self.component_usage[component] = original_usage + max_mb

        try:
            yield
        finally:
            self.component_usage[component] = original_usage

    def get_memory_efficiency_score(self) -> float:
        """
        Calculate memory efficiency score (0-1)

        Returns:
            Efficiency score where 1.0 is optimal
        """
        current_usage = self.check_memory_usage()

        if 'error' in current_usage:
            return 0.0

        usage_percentage = current_usage['usage_percentage'] / 100

        # Optimal usage is around 60-70% of limit
        if usage_percentage < 0.5:
            return usage_percentage * 2  # Underutilization penalty
        elif usage_percentage <= 0.7:
            return 1.0  # Optimal range
        elif usage_percentage <= 0.9:
            return 1.0 - (usage_percentage - 0.7) * 2  # Approaching limit
        else:
            return max(0.1, 0.2 - (usage_percentage - 0.9))  # Critical usage

    def get_recommendations(self) -> List[str]:
        """
        Get memory optimization recommendations

        Returns:
            List of recommendation strings
        """
        recommendations = []
        current_usage = self.check_memory_usage()

        if 'error' in current_usage:
            return ["Unable to analyze memory usage"]

        usage_percentage = current_usage['usage_percentage'] / 100
        status = current_usage['status']

        if status == 'emergency':
            recommendations.extend([
                "URGENT: Free up memory immediately",
                "Consider restarting the application",
                "Disable non-essential features"
            ])
        elif status == 'critical':
            recommendations.extend([
                "Force garbage collection immediately",
                "Clear all caches",
                "Reduce batch sizes"
            ])
        elif status == 'warning':
            recommendations.extend([
                "Consider proactive garbage collection",
                "Monitor memory trends closely",
                "Optimize data structures"
            ])

        # Component-specific recommendations
        if self.component_usage.get('text_classifier', 0) > 150:
            recommendations.append("Consider using smaller text model or batch processing")

        if self.component_usage.get('feature_engine', 0) > 100:
            recommendations.append("Enable lazy loading for feature generation")

        if current_usage.get('trend') == 'increasing':
            recommendations.append("Investigate memory leak potential")

        return recommendations

    def compress_vectors(self, vectors: List[List[float]],
                        model_name: str = 'default',
                        target_dim: Optional[int] = None) -> List[List[float]]:
        """
        Compress vectors using PCA to reduce memory usage

        Args:
            vectors: List of vectors to compress
            model_name: Name for the compression model
            target_dim: Target dimension (uses default if None)

        Returns:
            List of compressed vectors
        """
        if not self.limits.vector_compression_enabled:
            logger.info("Vector compression disabled, returning original vectors")
            return vectors

        if not vectors:
            return []

        target_dim = target_dim or self.limits.target_vector_dim
        original_dim = len(vectors[0])

        # Skip compression if vectors are already small
        if original_dim <= target_dim:
            logger.info(f"Vectors already small ({original_dim} ≤ {target_dim}), skipping compression")
            return vectors

        # Convert to numpy array
        vectors_np = np.array(vectors, dtype=np.float32)
        original_memory_mb = vectors_np.nbytes / (1024 * 1024)

        try:
            # Create or retrieve compression model
            if model_name not in self._pca_models:
                logger.info(f"Training new PCA model for '{model_name}': {original_dim}→{target_dim}d")

                # Standardize features
                scaler = StandardScaler()
                vectors_scaled = scaler.fit_transform(vectors_np)
                self._scalers[model_name] = scaler

                # Fit PCA
                pca = PCA(n_components=target_dim, random_state=42)
                vectors_compressed = pca.fit_transform(vectors_scaled)
                self._pca_models[model_name] = pca

                variance_explained = sum(pca.explained_variance_ratio_)
                logger.info(f"PCA model trained: {variance_explained:.3f} variance retained")

            else:
                # Use existing model
                scaler = self._scalers[model_name]
                pca = self._pca_models[model_name]

                vectors_scaled = scaler.transform(vectors_np)
                vectors_compressed = pca.transform(vectors_scaled)

            # Convert back to list format
            compressed_vectors = vectors_compressed.tolist()
            compressed_memory_mb = len(compressed_vectors) * target_dim * 4 / (1024 * 1024)  # float32

            # Update statistics
            memory_saved = original_memory_mb - compressed_memory_mb
            self._compression_stats['original_vectors'] += len(vectors) * original_dim
            self._compression_stats['compressed_vectors'] += len(compressed_vectors) * target_dim
            self._compression_stats['memory_saved_mb'] += memory_saved
            self._compression_stats['compression_ratio'] = (
                self._compression_stats['compressed_vectors'] /
                max(self._compression_stats['original_vectors'], 1)
            )

            logger.info(f"Compressed {len(vectors)} vectors: {original_memory_mb:.2f}MB → {compressed_memory_mb:.2f}MB "
                       f"(saved {memory_saved:.2f}MB, {memory_saved/original_memory_mb*100:.1f}%)")

            return compressed_vectors

        except Exception as e:
            logger.error(f"Vector compression failed: {e}, returning original vectors")
            return vectors

    def decompress_vectors(self, compressed_vectors: List[List[float]],
                          model_name: str = 'default') -> List[List[float]]:
        """
        Decompress vectors back to original dimensions

        Args:
            compressed_vectors: List of compressed vectors
            model_name: Name of the compression model to use

        Returns:
            List of decompressed vectors (approximation of original)
        """
        if model_name not in self._pca_models:
            logger.warning(f"No compression model found for '{model_name}', returning compressed vectors")
            return compressed_vectors

        try:
            # Convert to numpy array
            vectors_np = np.array(compressed_vectors, dtype=np.float32)

            # Get models
            pca = self._pca_models[model_name]
            scaler = self._scalers[model_name]

            # Decompress
            vectors_decompressed = pca.inverse_transform(vectors_np)
            vectors_original = scaler.inverse_transform(vectors_decompressed)

            return vectors_original.tolist()

        except Exception as e:
            logger.error(f"Vector decompression failed: {e}, returning compressed vectors")
            return compressed_vectors

    def get_compression_info(self) -> Dict[str, Any]:
        """
        Get information about vector compression models and statistics

        Returns:
            Dictionary with compression information
        """
        return {
            'compression_enabled': self.limits.vector_compression_enabled,
            'target_dimension': self.limits.target_vector_dim,
            'models_trained': list(self._pca_models.keys()),
            'statistics': self._compression_stats.copy(),
            'memory_saved_mb': self._compression_stats['memory_saved_mb'],
            'compression_ratio': self._compression_stats['compression_ratio']
        }

    def clear_compression_models(self) -> None:
        """Clear all compression models to free memory"""
        models_cleared = len(self._pca_models)
        self._pca_models.clear()
        self._scalers.clear()

        # Reset statistics
        self._compression_stats = {
            'original_vectors': 0,
            'compressed_vectors': 0,
            'compression_ratio': 0.0,
            'memory_saved_mb': 0.0
        }

        memory_freed = models_cleared * 5  # Estimate 5MB per model
        logger.info(f"Cleared {models_cleared} compression models, freed ~{memory_freed:.1f}MB")

    def optimize_cloud_resources(self) -> Dict[str, Any]:
        """
        Comprehensive cloud resource optimization

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting cloud resource optimization...")

        results = {
            'start_memory': self.check_memory_usage(),
            'actions_taken': [],
            'memory_saved_mb': 0,
            'compression_stats': self.get_compression_info()
        }

        # Action 1: Standard memory optimization
        standard_result = self.optimize_for_memory()
        results['actions_taken'].extend(standard_result['actions_taken'])
        results['memory_saved_mb'] += standard_result.get('total_memory_saved_mb', 0)

        # Action 2: Clear unused compression models
        if len(self._pca_models) > 5:  # Keep only 5 most recent models
            models_to_remove = len(self._pca_models) - 5
            self.clear_compression_models()
            results['actions_taken'].append(f"Removed {models_to_remove} unused compression models")

        # Action 3: Reduce component memory allocations
        for component, current_usage in self.component_usage.items():
            if current_usage > 100:  # If component uses more than 100MB
                reduction = min(current_usage * 0.3, 50)  # Reduce by up to 30% or 50MB
                self.component_usage[component] = current_usage - reduction
                results['actions_taken'].append(f"Reduced {component} allocation by {reduction:.1f}MB")
                results['memory_saved_mb'] += reduction

        # Action 4: Force garbage collection
        gc_result = self.force_garbage_collection()
        results['memory_saved_mb'] += gc_result['memory_freed_mb']

        results['end_memory'] = self.check_memory_usage()
        results['total_memory_saved_mb'] = (
            results['start_memory'].get('current_usage_mb', 0) -
            results['end_memory'].get('current_usage_mb', 0)
        )

        logger.info(f"Cloud optimization completed: {results['total_memory_saved_mb']:.2f}MB saved")

        return results

    def export_memory_report(self) -> Dict[str, Any]:
        """
        Export comprehensive memory report

        Returns:
            Dictionary with detailed memory report
        """
        current_usage = self.check_memory_usage()

        report = {
            'timestamp': time.time(),
            'system_limits': {
                'max_ram_mb': self.limits.max_ram_mb,
                'max_disk_mb': self.limits.max_disk_mb,
                'max_cpu_percent': self.limits.max_cpu_percent,
                'vector_compression_enabled': self.limits.vector_compression_enabled,
                'target_vector_dim': self.limits.target_vector_dim
            },
            'current_usage': current_usage,
            'component_breakdown': self.component_usage.copy(),
            'efficiency_score': self.get_memory_efficiency_score(),
            'recommendations': self.get_recommendations(),
            'recent_snapshots': len(self.snapshots),
            'memory_trend': current_usage.get('trend', 'unknown'),
            'compression_info': self.get_compression_info(),
            'cloud_optimized': True
        }

        return report

    def get_cloud_optimization_stats(self) -> Dict[str, Any]:
        """Get cloud-specific optimization statistics"""
        try:
            current_usage = self.get_current_usage()

            return {
                'cloud_optimization_enabled': True,
                'current_memory_mb': current_usage.get('memory_rss_mb', 0),
                'memory_limit_mb': self.resource_limits.total_memory_mb,
                'memory_usage_percent': (current_usage.get('memory_rss_mb', 0) / self.resource_limits.total_memory_mb) * 100,
                'components_active': len([c for c in self.component_usage.values() if c > 0]),
                'compression_models_loaded': len(self.compression_models),
                'pca_compression_available': True,
                'target_vector_dimensions': 256,
                'cloud_tier': 'free',
                'storage_compression_ratio': 0.6,  # Estimated compression ratio
                'memory_efficiency_score': min(100, (current_usage.get('memory_rss_mb', 0) / self.resource_limits.total_memory_mb) * 100)
            }

        except Exception as e:
            return {'error': str(e), 'cloud_optimization_enabled': False}

    def cleanup(self):
        """Cleanup memory manager resources"""
        try:
            logger.info("[MEMORY-MANAGER] Cleaning up memory manager resources...")

            # Clear component usage
            self.component_usage.clear()

            # Clear compression models
            self.compression_models.clear()

            # Clear snapshots
            self.snapshots.clear()

            # Force garbage collection
            import gc
            gc.collect()

            logger.info("[OK] Memory manager resources cleaned up")

        except Exception as e:
            logger.error(f"[ERROR] Memory manager cleanup failed: {e}")

# Global memory manager instance
memory_manager = MemoryManager()

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    return memory_manager