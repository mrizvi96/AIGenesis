"""
Intelligent Search & Memory Management
Hierarchical search with business rule integration
Optimized for 0.5 vCPU constraint
Research-backed: Multi-tiered search strategy
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import logging
from datetime import datetime, timedelta
import heapq
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchTier(Enum):
    """Search tier classification"""
    TIER_1 = "tier_1"  # High priority, cached results
    TIER_2 = "tier_2"  # Medium priority, recent results
    TIER_3 = "tier_3"  # Low priority, full search

class SearchStrategy(Enum):
    """Search strategy types"""
    EXACT_MATCH = "exact_match"
    SIMILARITY_SEARCH = "similarity_search"
    HYBRID_SEARCH = "hybrid_search"
    BUSINESS_RULES = "business_rules"

@dataclass
class SearchResult:
    """Search result with metadata"""
    id: str
    score: float
    tier: SearchTier
    metadata: Dict[str, Any]
    cache_hit: bool = False
    processing_time_ms: float = 0.0

class SearchCache:
    """Memory-efficient search result cache"""

    def __init__(self, max_size: int = 500, ttl_minutes: int = 60):
        self.max_size = max_size
        self.ttl_minutes = ttl_minutes
        self.cache = {}  # query_hash -> (result, timestamp)
        self.hit_count = 0
        self.miss_count = 0

    def get(self, query_hash: str) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        if query_hash in self.cache:
            result, timestamp = self.cache[query_hash]

            # Check TTL
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60
            if age_minutes < self.ttl_minutes:
                self.hit_count += 1
                return result
            else:
                # Expired
                del self.cache[query_hash]

        self.miss_count += 1
        return None

    def put(self, query_hash: str, results: List[SearchResult]):
        """Store search results in cache"""
        # Remove oldest if at capacity
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[query_hash] = (results, datetime.now())

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0

        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate
        }

class IntelligentSearchManager:
    """
    Intelligent search management with business rules
    Implements hierarchical search and memory optimization
    Optimized for CPU and memory constraints
    """

    def __init__(self, cpu_limit_vcpu: float = 0.5):
        """Initialize intelligent search manager"""
        logger.info(f"[SEARCH-MGR] Loading intelligent search manager (cpu_limit={cpu_limit_vcpu}vCPU)...")

        self.cpu_limit_vcpu = cpu_limit_vcpu

        # Search configuration for resource efficiency
        self.config = {
            'max_results_per_tier': {
                'tier_1': 10,
                'tier_2': 20,
                'tier_3': 50
            },
            'search_limits': {
                'max_searches_per_minute': 100,
                'batch_size': 10,
                'timeout_seconds': 5
            },
            'business_rules': {
                'priority_thresholds': {
                    'high_value': 10000.0,
                    'medium_value': 5000.0,
                    'low_value': 1000.0
                },
                'recent_threshold_days': 30,
                'duplicate_threshold': 0.95
            }
        }

        # Initialize components
        self.search_cache = SearchCache()
        self.business_rule_engine = BusinessRuleEngine(self.config['business_rules'])

        # Performance tracking
        self.performance_metrics = {
            'searches_performed': 0,
            'total_search_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info("[OK] Intelligent search manager initialized")

    def intelligent_search(self, query_vector: List[float],
                         query_filters: Dict[str, Any] = None,
                         max_results: int = 20) -> Dict[str, Any]:
        """
        Perform intelligent search with hierarchical strategy
        Optimizes for resource constraints and business rules
        """
        try:
            logger.info(f"[SEARCH-MGR] Starting intelligent search (max_results={max_results})...")

            search_start_time = datetime.now()

            # Step 1: Create query signature for caching
            query_hash = self._create_query_hash(query_vector, query_filters, max_results)

            # Step 2: Check cache first
            cached_results = self.search_cache.get(query_hash)
            if cached_results:
                logger.debug("[CACHE] Search results found in cache")
                return self._format_search_results(cached_results[:max_results], True)

            # Step 3: Determine search strategy
            search_strategy = self._determine_search_strategy(query_vector, query_filters)

            # Step 4: Execute hierarchical search
            search_results = self._execute_hierarchical_search(query_vector, query_filters, search_strategy)

            # Step 5: Apply business rules
            filtered_results = self.business_rule_engine.apply_rules(search_results, query_filters)

            # Step 6: Cache results
            self.search_cache.put(query_hash, filtered_results)

            # Step 7: Format and return results
            search_time = (datetime.now() - search_start_time).total_seconds() * 1000
            formatted_results = self._format_search_results(filtered_results[:max_results], False, search_time)

            # Update metrics
            self._update_performance_metrics(search_time, False)

            logger.info(f"[OK] Intelligent search completed - {len(formatted_results['results'])} results in {search_time:.1f}ms")
            return formatted_results

        except Exception as e:
            logger.error(f"[ERROR] Intelligent search failed: {e}")
            return self._get_fallback_search_results()

    def _create_query_hash(self, query_vector: List[float], filters: Dict[str, Any], max_results: int) -> str:
        """Create hash for query caching"""
        try:
            import hashlib

            # Combine all query components
            query_data = {
                'vector': tuple(query_vector),
                'filters': filters or {},
                'max_results': max_results
            }

            # Create hash
            query_str = json.dumps(query_data, sort_keys=True)
            return hashlib.md5(query_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"[ERROR] Query hash creation failed: {e}")
            return f"fallback_{datetime.now().timestamp()}"

    def _determine_search_strategy(self, query_vector: List[float], filters: Dict[str, Any]) -> SearchStrategy:
        """Determine optimal search strategy based on query characteristics"""
        try:
            # Strategy selection based on query complexity and filters
            if filters and any(key in filters for key in ['claim_type', 'amount_range', 'date_range']):
                return SearchStrategy.BUSINESS_RULES

            if len(query_vector) > 128:
                return SearchStrategy.HYBRID_SEARCH

            if filters and filters.get('exact_match', False):
                return SearchStrategy.EXACT_MATCH

            return SearchStrategy.SIMILARITY_SEARCH

        except Exception as e:
            logger.error(f"[ERROR] Strategy determination failed: {e}")
            return SearchStrategy.SIMILARITY_SEARCH

    def _execute_hierarchical_search(self, query_vector: List[float],
                                   query_filters: Dict[str, Any],
                                   strategy: SearchStrategy) -> List[SearchResult]:
        """Execute hierarchical search across tiers"""
        try:
            all_results = []

            # Tier 1: High priority cached results
            tier1_results = self._search_tier_1(query_vector, query_filters, strategy)
            all_results.extend(tier1_results)

            # If we have enough results from Tier 1, we might skip deeper searches
            if len(all_results) < 10:  # Need more results
                # Tier 2: Recent results
                tier2_results = self._search_tier_2(query_vector, query_filters, strategy)
                all_results.extend(tier2_results)

            # If still need more results
            if len(all_results) < self.config['max_results_per_tier']['tier_3'] // 2:
                # Tier 3: Full database search
                tier3_results = self._search_tier_3(query_vector, query_filters, strategy)
                all_results.extend(tier3_results)

            # Remove duplicates and rank by score
            unique_results = self._deduplicate_and_rank(all_results)

            return unique_results

        except Exception as e:
            logger.error(f"[ERROR] Hierarchical search failed: {e}")
            return []

    def _search_tier_1(self, query_vector: List[float], filters: Dict[str, Any], strategy: SearchStrategy) -> List[SearchResult]:
        """Search in Tier 1 (high priority cached results)"""
        try:
            # Simulate Tier 1 search - high value recent claims
            results = []

            # Generate mock results for demonstration
            for i in range(self.config['max_results_per_tier']['tier_1']):
                result = SearchResult(
                    id=f"tier1_{i}",
                    score=0.9 - (i * 0.05),  # High scores
                    tier=SearchTier.TIER_1,
                    metadata={
                        'priority': 'high',
                        'cache_level': 'hot',
                        'last_accessed': datetime.now().isoformat(),
                        'business_rules_applied': True
                    },
                    cache_hit=True
                )
                results.append(result)

            logger.debug(f"[TIER-1] Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"[ERROR] Tier 1 search failed: {e}")
            return []

    def _search_tier_2(self, query_vector: List[float], filters: Dict[str, Any], strategy: SearchStrategy) -> List[SearchResult]:
        """Search in Tier 2 (medium priority recent results)"""
        try:
            results = []

            # Generate mock results for demonstration
            for i in range(self.config['max_results_per_tier']['tier_2']):
                result = SearchResult(
                    id=f"tier2_{i}",
                    score=0.7 - (i * 0.03),  # Medium scores
                    tier=SearchTier.TIER_2,
                    metadata={
                        'priority': 'medium',
                        'cache_level': 'warm',
                        'last_accessed': datetime.now().isoformat(),
                        'days_old': np.random.randint(1, 30)
                    },
                    cache_hit=False
                )
                results.append(result)

            logger.debug(f"[TIER-2] Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"[ERROR] Tier 2 search failed: {e}")
            return []

    def _search_tier_3(self, query_vector: List[float], filters: Dict[str, Any], strategy: SearchStrategy) -> List[SearchResult]:
        """Search in Tier 3 (full database search)"""
        try:
            results = []

            # Generate mock results for demonstration
            for i in range(self.config['max_results_per_tier']['tier_3']):
                result = SearchResult(
                    id=f"tier3_{i}",
                    score=0.5 - (i * 0.01),  # Lower scores
                    tier=SearchTier.TIER_3,
                    metadata={
                        'priority': 'low',
                        'cache_level': 'cold',
                        'last_accessed': datetime.now().isoformat(),
                        'days_old': np.random.randint(30, 365)
                    },
                    cache_hit=False,
                    processing_time_ms=np.random.uniform(10, 50)
                )
                results.append(result)

            logger.debug(f"[TIER-3] Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"[ERROR] Tier 3 search failed: {e}")
            return []

    def _deduplicate_and_rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicates and rank by score"""
        try:
            seen_ids = set()
            unique_results = []

            # Sort by score (descending) and tier priority
            sorted_results = sorted(results, key=lambda x: (-x.score, x.tier.value))

            for result in sorted_results:
                if result.id not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result.id)

            return unique_results

        except Exception as e:
            logger.error(f"[ERROR] Deduplication failed: {e}")
            return results[:50]  # Fallback: limit results

    def _format_search_results(self, results: List[SearchResult], cache_hit: bool, search_time_ms: float = 0.0) -> Dict[str, Any]:
        """Format search results for API response"""
        try:
            formatted_results = []
            total_processing_time = 0.0

            for result in results:
                formatted_result = {
                    'id': result.id,
                    'score': result.score,
                    'tier': result.tier.value,
                    'metadata': result.metadata
                }
                formatted_results.append(formatted_result)
                total_processing_time += result.processing_time_ms

            return {
                'results': formatted_results,
                'search_metadata': {
                    'total_results': len(results),
                    'cache_hit': cache_hit,
                    'search_time_ms': search_time_ms,
                    'total_processing_time_ms': search_time_ms + total_processing_time,
                    'tiers_used': list(set(r.tier.value for r in results)),
                    'strategy_used': self._get_strategy_name(),
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"[ERROR] Result formatting failed: {e}")
            return {
                'results': [],
                'search_metadata': {'error': str(e)},
                'error': True
            }

    def _get_strategy_name(self) -> str:
        """Get current search strategy name"""
        # This would typically be stored during search execution
        # For now, return a default
        return SearchStrategy.SIMILARITY_SEARCH.value

    def _update_performance_metrics(self, search_time_ms: float, cache_hit: bool):
        """Update performance metrics"""
        self.performance_metrics['searches_performed'] += 1
        self.performance_metrics['total_search_time_ms'] += search_time_ms

        if cache_hit:
            self.performance_metrics['cache_hits'] += 1
        else:
            self.performance_metrics['cache_misses'] += 1

    def _get_fallback_search_results(self) -> Dict[str, Any]:
        """Get fallback search results"""
        return {
            'results': [],
            'search_metadata': {
                'total_results': 0,
                'cache_hit': False,
                'search_time_ms': 0.0,
                'error': 'Search failed - using fallback',
                'strategy_used': 'fallback'
            },
            'error': True
        }

    def get_search_stats(self) -> Dict[str, Any]:
        """Get comprehensive search statistics"""
        try:
            cache_stats = self.search_cache.get_stats()

            avg_search_time = 0.0
            if self.performance_metrics['searches_performed'] > 0:
                avg_search_time = self.performance_metrics['total_search_time_ms'] / self.performance_metrics['searches_performed']

            return {
                'performance': {
                    'searches_performed': self.performance_metrics['searches_performed'],
                    'avg_search_time_ms': avg_search_time,
                    'cache_hit_rate': cache_stats['hit_rate'],
                    'total_cache_hits': self.performance_metrics['cache_hits'],
                    'total_cache_misses': self.performance_metrics['cache_misses']
                },
                'cache': cache_stats,
                'config': self.config,
                'business_rules': self.config['business_rules']
            }

        except Exception as e:
            logger.error(f"[ERROR] Search stats retrieval failed: {e}")
            return {'error': str(e)}

class BusinessRuleEngine:
    """Applies business rules to search results"""

    def __init__(self, business_rules: Dict[str, Any]):
        self.rules = business_rules

    def apply_rules(self, results: List[SearchResult], query_filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Apply business rules to filter and rank results"""
        try:
            filtered_results = []

            for result in results:
                # Apply priority filtering
                if self._meets_priority_criteria(result, query_filters):
                    # Apply scoring adjustments
                    adjusted_result = self._adjust_score(result, query_filters)
                    filtered_results.append(adjusted_result)

            # Re-sort after adjustments
            filtered_results.sort(key=lambda x: x.score, reverse=True)

            return filtered_results

        except Exception as e:
            logger.error(f"[ERROR] Business rules application failed: {e}")
            return results

    def _meets_priority_criteria(self, result: SearchResult, filters: Dict[str, Any]) -> bool:
        """Check if result meets priority criteria"""
        try:
            metadata = result.metadata

            # High priority always passes
            if metadata.get('priority') == 'high':
                return True

            # Medium priority passes if recent
            if metadata.get('priority') == 'medium':
                days_old = metadata.get('days_old', 0)
                return days_old <= self.rules['recent_threshold_days']

            # Low priority passes if very recent
            if metadata.get('priority') == 'low':
                days_old = metadata.get('days_old', 0)
                return days_old <= 7

            return False

        except Exception:
            return True  # Pass on error

    def _adjust_score(self, result: SearchResult, filters: Dict[str, Any]) -> SearchResult:
        """Adjust result score based on business rules"""
        try:
            metadata = result.metadata
            original_score = result.score

            # Tier-based adjustments
            tier_multipliers = {
                SearchTier.TIER_1: 1.2,
                SearchTier.TIER_2: 1.0,
                SearchTier.TIER_3: 0.8
            }

            multiplier = tier_multipliers.get(result.tier, 1.0)

            # Time-based decay
            days_old = metadata.get('days_old', 0)
            time_decay = max(0.5, 1.0 - (days_old / 365))

            # Apply adjustments
            adjusted_score = original_score * multiplier * time_decay
            result.score = min(1.0, adjusted_score)  # Cap at 1.0

            return result

        except Exception as e:
            logger.error(f"[ERROR] Score adjustment failed: {e}")
            return result

# Global instance for memory efficiency
_intelligent_search_manager = None

def get_intelligent_search_manager(cpu_limit_vcpu: float = 0.5) -> IntelligentSearchManager:
    """Get or create intelligent search manager instance"""
    global _intelligent_search_manager
    if _intelligent_search_manager is None:
        _intelligent_search_manager = IntelligentSearchManager(cpu_limit_vcpu)
    return _intelligent_search_manager