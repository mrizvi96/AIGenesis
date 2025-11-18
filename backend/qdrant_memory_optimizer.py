"""
Qdrant Memory Optimization for 4GiB Disk Constraint
Implements payload filtering, compression, and LRU caching
Research-backed: Hierarchical search with business rules
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
import pickle
import hashlib
from dataclasses import dataclass
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_vectors: int
    total_payload_size_mb: float
    index_size_mb: float
    cache_size_mb: float
    available_disk_mb: float
    compression_ratio: float

class PayloadCompressor:
    """Compresses and decompresses vector payloads"""

    def __init__(self):
        self.compression_methods = ['json', 'pickle', 'base64']

    def compress_payload(self, payload: Dict[str, Any]) -> Tuple[bytes, str, float]:
        """Compress payload and return (compressed_data, method, compression_ratio)"""
        try:
            # Method 1: JSON + gzip
            json_data = json.dumps(payload, separators=(',', ':')).encode('utf-8')
            compressed_json = self._gzip_compress(json_data)
            json_ratio = len(json_data) / len(compressed_json) if compressed_json else 1.0

            # Method 2: Pickle
            pickle_data = pickle.dumps(payload)
            compressed_pickle = self._gzip_compress(pickle_data)
            pickle_ratio = len(pickle_data) / len(compressed_pickle) if compressed_pickle else 1.0

            # Method 3: Simple (no compression)
            simple_data = json_data
            simple_ratio = 1.0

            # Choose best method
            methods = [
                (compressed_json, 'json_gzip', json_ratio),
                (compressed_pickle, 'pickle_gzip', pickle_ratio),
                (simple_data, 'simple', simple_ratio)
            ]

            best_method = min(methods, key=lambda x: len(x[0]))

            return best_method[0], best_method[1], best_method[2]

        except Exception as e:
            logger.error(f"[ERROR] Payload compression failed: {e}")
            return json.dumps(payload).encode('utf-8'), 'simple', 1.0

    def decompress_payload(self, compressed_data: bytes, method: str) -> Dict[str, Any]:
        """Decompress payload"""
        try:
            if method == 'json_gzip':
                decompressed = self._gzip_decompress(compressed_data)
                return json.loads(decompressed.decode('utf-8'))
            elif method == 'pickle_gzip':
                decompressed = self._gzip_decompress(compressed_data)
                return pickle.loads(decompressed)
            elif method == 'simple':
                return json.loads(compressed_data.decode('utf-8'))
            else:
                raise ValueError(f"Unknown compression method: {method}")

        except Exception as e:
            logger.error(f"[ERROR] Payload decompression failed: {e}")
            return {}

    def _gzip_compress(self, data: bytes) -> bytes:
        """Compress data with gzip"""
        import gzip
        return gzip.compress(data)

    def _gzip_decompress(self, compressed_data: bytes) -> bytes:
        """Decompress gzip data"""
        import gzip
        return gzip.decompress(compressed_data)

class LRUCache:
    """Memory-efficient LRU cache for frequently accessed vectors"""

    def __init__(self, max_size: int = 1000, memory_limit_mb: float = 100.0):
        self.max_size = max_size
        self.memory_limit_mb = memory_limit_mb
        self.current_memory_mb = 0.0
        self.cache = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hit_count += 1
            return value
        else:
            self.miss_count += 1
            return None

    def put(self, key: str, value: Any):
        """Put item in cache"""
        # Calculate memory usage of value
        value_size_mb = self._calculate_size(value)

        # Remove items if necessary
        while (len(self.cache) >= self.max_size or
               self.current_memory_mb + value_size_mb > self.memory_limit_mb):
            if self.cache:
                oldest_key, oldest_value = self.cache.popitem(last=False)
                self.current_memory_mb -= self._calculate_size(oldest_value)
            else:
                break

        # Add new item
        if key in self.cache:
            # Update existing item
            old_value = self.cache.pop(key)
            self.current_memory_mb -= self._calculate_size(old_value)

        self.cache[key] = value
        self.current_memory_mb += value_size_mb

    def _calculate_size(self, value: Any) -> float:
        """Calculate approximate memory usage in MB"""
        try:
            if isinstance(value, dict):
                size = len(json.dumps(value).encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                size = len(str(value).encode('utf-8'))
            elif isinstance(value, str):
                size = len(value.encode('utf-8'))
            else:
                size = len(str(value).encode('utf-8'))

            return size / (1024 * 1024)  # Convert to MB
        except:
            return 0.01  # Default small size

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0

        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'current_memory_mb': self.current_memory_mb,
            'memory_limit_mb': self.memory_limit_mb,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate
        }

class QdrantMemoryOptimizer:
    """
    Qdrant optimization for strict resource constraints
    Implements memory-aware vector storage and retrieval
    Optimized for 4GiB disk and 1GiB RAM constraints
    """

    def __init__(self, disk_limit_gb: float = 4.0, ram_limit_gb: float = 1.0):
        """Initialize Qdrant memory optimizer"""
        logger.info(f"[QDRANT-OPT] Loading Qdrant memory optimizer (disk={disk_limit_gb}GB, ram={ram_limit_gb}GB)...")

        self.disk_limit_bytes = disk_limit_gb * 1024 * 1024 * 1024
        self.ram_limit_bytes = ram_limit_gb * 1024 * 1024 * 1024

        # Configuration for memory optimization
        self.config = {
            'vectors': {
                'size': 256,          # Optimized vector size
                'distance': 'Cosine',     # Most memory efficient
                'indexing': 'HNSW'        # Hierarchical Navigable Small World
            },
            'payload': {
                'indexing': True,        # Enable payload indexing
                'compression': True,     # Compress payloads
                'batch_size': 100          # Process in batches
            },
            'cache': {
                'max_entries': 1000,      # Cache recent queries
                'ttl_seconds': 3600,        # 1 hour TTL
                'memory_limit_mb': 100      # Cache memory limit
            }
        }

        # Initialize components
        self.payload_compressor = PayloadCompressor()
        self.vector_cache = LRUCache(
            max_size=self.config['cache']['max_entries'],
            memory_limit_mb=self.config['cache']['memory_limit_mb']
        )

        # Memory tracking
        self.memory_stats = MemoryStats(0, 0.0, 0.0, 0.0, 0.0, 1.0)

        # Business rule filters
        self.business_rules = self._initialize_business_rules()

        logger.info("[OK] Qdrant memory optimizer initialized")

    def _initialize_business_rules(self) -> Dict[str, Any]:
        """Initialize business rules for memory optimization"""
        return {
            'claim_filters': {
                'min_amount': 100,      # Ignore claims below this amount
                'max_age_days': 365,     # Archive old claims
                'duplicate_threshold': 0.95  # Similarity threshold for duplicates
            },
            'vector_retention': {
                'high_priority_days': 90,    # Keep high-value claims for 90 days
                'medium_priority_days': 30,  # Keep medium-value claims for 30 days
                'low_priority_days': 7       # Keep low-value claims for 7 days
            },
            'compression_rules': {
                'min_payload_size': 1000,    # Compress payloads larger than this
                'text_compression': True,
                'metadata_compression': True
            }
        }

    def optimize_vector_storage(self, vectors: List[Dict[str, Any]],
                               payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize vector storage for memory efficiency
        Implements compression, filtering, and batching
        """
        try:
            logger.info(f"[QDRANT-OPT] Optimizing {len(vectors)} vectors for storage...")

            initial_stats = self._estimate_storage_requirements(vectors, payloads)

            # Step 1: Apply business rules filtering
            filtered_indices = self._apply_business_rules_filters(vectors, payloads)
            filtered_vectors = [vectors[i] for i in filtered_indices]
            filtered_payloads = [payloads[i] for i in filtered_indices]

            # Step 2: Compress payloads
            compressed_payloads = []
            compression_stats = []

            for payload in filtered_payloads:
                compressed_data, method, ratio = self.payload_compressor.compress_payload(payload)
                compressed_payloads.append({
                    'data': compressed_data,
                    'method': method,
                    'compression_ratio': ratio
                })
                compression_stats.append(ratio)

            # Step 3: Batch processing for memory efficiency
            batches = self._create_optimal_batches(filtered_vectors, compressed_payloads)

            # Step 4: Generate storage plan
            storage_plan = self._generate_storage_plan(
                batches, initial_stats, compression_stats
            )

            result = {
                'optimized_vectors': filtered_vectors,
                'compressed_payloads': compressed_payloads,
                'batches': batches,
                'storage_plan': storage_plan,
                'optimization_stats': {
                    'original_count': len(vectors),
                    'filtered_count': len(filtered_vectors),
                    'compression_savings': np.mean(compression_stats) if compression_stats else 1.0,
                    'memory_reduction': initial_stats['estimated_size_mb'] / storage_plan['estimated_size_mb'],
                    'filtering_removed': len(vectors) - len(filtered_vectors)
                }
            }

            logger.info(f"[OK] Optimization complete - {len(filtered_vectors)} vectors, {storage_plan['estimated_size_mb']:.1f}MB")
            return result

        except Exception as e:
            logger.error(f"[ERROR] Vector storage optimization failed: {e}")
            return {
                'optimized_vectors': vectors[:100],  # Fallback to smaller subset
                'compressed_payloads': [{'data': json.dumps(p).encode(), 'method': 'simple', 'compression_ratio': 1.0} for p in payloads[:100]],
                'batches': [],
                'storage_plan': self._get_fallback_plan(),
                'optimization_stats': {'error': str(e)}
            }

    def _apply_business_rules_filters(self, vectors: List[Dict[str, Any]],
                                    payloads: List[Dict[str, Any]]) -> List[int]:
        """Apply business rules to filter vectors"""
        try:
            valid_indices = []
            rules = self.business_rules['claim_filters']

            for i, (vector, payload) in enumerate(zip(vectors, payloads)):
                try:
                    # Amount filter
                    amount = float(payload.get('amount', 0))
                    if amount < rules['min_amount']:
                        continue

                    # Age filter
                    claim_date = payload.get('claim_date', '')
                    if claim_date:
                        try:
                            claim_dt = datetime.strptime(claim_date, '%Y-%m-%d')
                            age_days = (datetime.now() - claim_dt).days
                            if age_days > rules['max_age_days']:
                                continue
                        except:
                            pass  # If date parsing fails, keep the vector

                    # Duplicate detection (simplified)
                    if i > 0 and self._is_duplicate_vector(vector, vectors[valid_indices[-1]]):
                        continue

                    valid_indices.append(i)

                except Exception as e:
                    logger.warning(f"[WARNING] Error filtering vector {i}: {e}")
                    valid_indices.append(i)  # Keep on error

            return valid_indices

        except Exception as e:
            logger.error(f"[ERROR] Business rules filtering failed: {e}")
            return list(range(len(vectors)))  # Return all indices on error

    def _is_duplicate_vector(self, vector1: Dict[str, Any], vector2: Dict[str, Any]) -> bool:
        """Check if two vectors are duplicates"""
        try:
            v1 = np.array(vector1.get('embedding', []))
            v2 = np.array(vector2.get('embedding', []))

            if len(v1) == 0 or len(v2) == 0:
                return False

            # Calculate cosine similarity
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return similarity > self.business_rules['claim_filters']['duplicate_threshold']

        except Exception:
            return False

    def _create_optimal_batches(self, vectors: List[Dict[str, Any]],
                             payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create optimal batches for memory-efficient processing"""
        try:
            batch_size = self.config['payload']['batch_size']
            batches = []

            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                batch_payloads = payloads[i:i + batch_size]

                batch = {
                    'vectors': batch_vectors,
                    'payloads': batch_payloads,
                    'batch_id': i // batch_size,
                    'size': len(batch_vectors),
                    'estimated_memory_mb': self._estimate_batch_memory(batch_vectors, batch_payloads)
                }

                batches.append(batch)

            return batches

        except Exception as e:
            logger.error(f"[ERROR] Batch creation failed: {e}")
            return [{'vectors': vectors[:50], 'payloads': payloads[:50], 'batch_id': 0, 'size': len(vectors[:50]), 'estimated_memory_mb': 10.0}]

    def _estimate_batch_memory(self, vectors: List[Dict[str, Any]], payloads: List[Dict[str, Any]]) -> float:
        """Estimate memory usage for a batch"""
        try:
            # Vector memory (256 dimensions * 4 bytes per float)
            vector_memory = len(vectors) * 256 * 4 / (1024 * 1024)  # MB

            # Payload memory (compressed)
            payload_memory = sum(len(p['data']) for p in payloads) / (1024 * 1024)  # MB

            # Indexing overhead (estimate)
            indexing_overhead = vector_memory * 0.2  # 20% overhead

            return vector_memory + payload_memory + indexing_overhead

        except Exception:
            return 50.0  # Default estimate

    def _generate_storage_plan(self, batches: List[Dict[str, Any]],
                             initial_stats: Dict[str, Any],
                             compression_stats: List[float]) -> Dict[str, Any]:
        """Generate storage optimization plan"""
        try:
            total_memory = sum(batch['estimated_memory_mb'] for batch in batches)
            avg_compression = np.mean(compression_stats) if compression_stats else 1.0

            plan = {
                'estimated_size_mb': total_memory,
                'disk_utilization_percent': (total_memory * 1024) / (self.disk_limit_bytes / (1024 * 1024)) * 100,
                'batch_count': len(batches),
                'compression_ratio': avg_compression,
                'memory_efficiency_score': min(1.0, avg_compression * (self.disk_limit_bytes / (total_memory * 1024 * 1024))),
                'recommendations': self._generate_storage_recommendations(total_memory, avg_compression),
                'hierarchy_levels': self._plan_hierarchy_levels(batches)
            }

            return plan

        except Exception as e:
            logger.error(f"[ERROR] Storage plan generation failed: {e}")
            return self._get_fallback_plan()

    def _generate_storage_recommendations(self, total_memory_mb: float, compression_ratio: float) -> List[str]:
        """Generate storage optimization recommendations"""
        recommendations = []

        if total_memory_mb > 3500:  # >3.5GB
            recommendations.append("Consider increasing vector compression (reduce dimensions to 128)")

        if compression_ratio < 1.5:
            recommendations.append("Enable aggressive payload compression")

        if total_memory_mb > 3000:  # >3GB
            recommendations.append("Implement stricter business rules filtering")

        if len(recommendations) == 0:
            recommendations.append("Memory optimization looks good")

        return recommendations

    def _plan_hierarchy_levels(self, batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan hierarchical search levels for memory efficiency"""
        return {
            'level_1': {  # Hot data - frequently accessed
                'batch_count': max(1, len(batches) // 3),
                'access_pattern': 'high_frequency',
                'memory_priority': 'high'
            },
            'level_2': {  # Warm data - occasionally accessed
                'batch_count': max(1, len(batches) // 3),
                'access_pattern': 'medium_frequency',
                'memory_priority': 'medium'
            },
            'level_3': {  # Cold data - rarely accessed
                'batch_count': max(1, len(batches) - 2 * (len(batches) // 3)),
                'access_pattern': 'low_frequency',
                'memory_priority': 'low'
            }
        }

    def _estimate_storage_requirements(self, vectors: List[Dict[str, Any]],
                                     payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate storage requirements"""
        try:
            # Vector storage
            vector_size = len(vectors[0].get('embedding', [])) if vectors else 256
            vector_memory = len(vectors) * vector_size * 4 / (1024 * 1024)  # MB

            # Payload storage
            payload_size = sum(len(json.dumps(p).encode()) for p in payloads) / (1024 * 1024)  # MB

            # Indexing overhead
            indexing_memory = vector_memory * 0.3  # 30% overhead

            total_memory = vector_memory + payload_size + indexing_memory

            return {
                'estimated_size_mb': total_memory,
                'vector_memory_mb': vector_memory,
                'payload_memory_mb': payload_size,
                'indexing_overhead_mb': indexing_memory,
                'disk_utilization_percent': (total_memory * 1024) / (self.disk_limit_bytes / (1024 * 1024)) * 100
            }

        except Exception as e:
            logger.error(f"[ERROR] Storage estimation failed: {e}")
            return {'estimated_size_mb': 100.0}  # Default estimate

    def _get_fallback_plan(self) -> Dict[str, Any]:
        """Get fallback storage plan"""
        return {
            'estimated_size_mb': 1000.0,
            'disk_utilization_percent': 25.0,
            'batch_count': 1,
            'compression_ratio': 1.0,
            'memory_efficiency_score': 0.5,
            'recommendations': ['Using fallback due to errors'],
            'hierarchy_levels': {'level_1': {'batch_count': 1, 'access_pattern': 'fallback'}}
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            cache_stats = self.vector_cache.get_stats()

            return {
                'limits': {
                    'disk_limit_gb': self.disk_limit_bytes / (1024**3),
                    'ram_limit_gb': self.ram_limit_bytes / (1024**3),
                    'disk_limit_mb': self.disk_limit_bytes / (1024**2),
                    'ram_limit_mb': self.ram_limit_bytes / (1024**2)
                },
                'usage': {
                    'estimated_usage_mb': self.memory_stats.total_payload_size_mb,
                    'cache_usage_mb': cache_stats['current_memory_mb'],
                    'cache_hit_rate': cache_stats['hit_rate'],
                    'compression_ratio': self.memory_stats.compression_ratio,
                    'vector_count': self.memory_stats.total_vectors
                },
                'config': self.config,
                'cache_stats': cache_stats,
                'business_rules': self.business_rules
            }

        except Exception as e:
            logger.error(f"[ERROR] Memory stats retrieval failed: {e}")
            return {'error': str(e)}

# Global instance for memory efficiency
_qdrant_memory_optimizer = None

def get_qdrant_memory_optimizer(disk_limit_gb: float = 4.0, ram_limit_gb: float = 1.0) -> QdrantMemoryOptimizer:
    """Get or create Qdrant memory optimizer instance"""
    global _qdrant_memory_optimizer
    if _qdrant_memory_optimizer is None:
        _qdrant_memory_optimizer = QdrantMemoryOptimizer(disk_limit_gb, ram_limit_gb)
    return _qdrant_memory_optimizer