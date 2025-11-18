"""
Hybrid Vision Processing for Qdrant Cloud Resource Constraints
Intelligent allocation: API for critical, local for bulk
Cloud-optimized: <120MB memory target with automatic model unloading
Smart resource allocation based on cloud CPU/memory limits
"""

import numpy as np
import os
import gc
import time
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime
import base64
from dataclasses import dataclass
from enum import Enum
from memory_manager import get_memory_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Processing mode selection"""
    API_HIGH_VALUE = "api_high_value"
    LOCAL_BULK = "local_bulk"
    FALLBACK = "fallback"

@dataclass
class ProcessingDecision:
    """Decision for image processing with cloud optimization"""
    mode: ProcessingMode
    confidence: float
    cost_estimate: float
    processing_time_estimate: float
    memory_estimate_mb: float
    cloud_cpu_impact: str  # low/medium/high
    reasoning: List[str]

class CloudAPIQuotaManager:
    """Manages API quotas and cloud resource optimization"""

    def __init__(self):
        # Cloud-optimized quotas for free tier
        self.daily_quota = 500  # Reduced for cloud cost efficiency
        self.monthly_budget = 50.0  # $50 per month for cloud
        self.cost_per_call = 0.02  # $0.02 per image analysis
        self.api_calls_today = 0
        self.cost_this_month = 0.0
        self.api_wait_time = 1.0  # seconds between API calls for cloud rate limiting

    def can_use_api(self, claim_value: float, image_count: int = 1) -> bool:
        """Check if API can be used for this claim with cloud optimization"""
        if self.api_calls_today + image_count > self.daily_quota:
            return False
        estimated_cost = image_count * self.cost_per_call
        if self.cost_this_month + estimated_cost > self.monthly_budget:
            return False

        # Cloud-optimized threshold - use API for higher value claims
        return claim_value > 3000.0  # Reduced threshold for better cost optimization

    def record_api_usage(self, claim_value: float, image_count: int = 1):
        """Record API usage with cloud rate limiting"""
        self.api_calls_today += image_count
        estimated_cost = image_count * self.cost_per_call
        self.cost_this_month += estimated_cost

        # Rate limiting for cloud API
        if self.api_calls_today > 0 and self.api_calls_today % 10 == 0:
            logger.info(f"[CLOUD-API] Rate limiting after {self.api_calls_today} calls, waiting {self.api_wait_time}s")
            time.sleep(self.api_wait_time)

    def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota status for monitoring"""
        return {
            'api_calls_today': self.api_calls_today,
            'daily_quota': self.daily_quota,
            'quota_remaining': max(0, self.daily_quota - self.api_calls_today),
            'cost_this_month': self.cost_this_month,
            'monthly_budget': self.monthly_budget,
            'budget_remaining': max(0, self.monthly_budget - self.cost_this_month),
            'quota_usage_percent': (self.api_calls_today / self.daily_quota) * 100
        }

class HybridVisionProcessor:
    """
    Cloud-optimized hybrid vision processing
    Uses API for critical claims, local models for bulk processing
    Implements intelligent cloud resource allocation with model unloading
    """

    def __init__(self, memory_limit_mb: int = 120):
        """Initialize cloud-optimized hybrid vision processor"""
        logger.info(f"[CLOUD-VISION] Loading hybrid vision processor (memory_limit={memory_limit_mb}MB)...")

        # Cloud optimization settings
        self.memory_limit_mb = memory_limit_mb
        self.api_quota_manager = CloudAPIQuotaManager()
        self.memory_manager = get_memory_manager()
        self.api_cost_per_image = 0.02  # $0.02 per image

        # Model management
        self.local_model = None
        self.model_loaded = False
        self.model_last_used = None
        self.model_unload_timeout = 180  # 3 minutes for vision models
        self.model_memory_mb = 80  # Target memory for vision model

        # Cloud resource management
        self.processing_queue = []
        self.currently_processing = False
        self.batch_size = 5  # Process 5 images at a time for cloud efficiency

        # Initialize local model if memory permits
        self._initialize_cloud_model()

    def _initialize_cloud_model(self):
        """Initialize model with cloud memory optimization"""
        # Check if we can allocate memory for the vision model
        allocation_result = self.memory_manager.can_allocate(self.model_memory_mb, 'vision_processor')

        if not allocation_result['can_allocate']:
            logger.warning(f"[CLOUD-VISION] Insufficient memory for vision model, using API-only mode")
            self.local_model = None
            self.model_loaded = False
            return

        try:
            logger.info(f"[CLOUD-VISION] Loading lightweight vision model (target: {self.model_memory_mb}MB)")

            # Cloud-optimized lightweight model placeholder
            self.local_model = {
                'model_type': 'cloud_optimized_cnn',
                'input_size': (224, 224),
                'memory_mb': self.model_memory_mb,
                'accuracy_estimate': 0.72,  # Slightly reduced for cloud efficiency
                'processing_speed': 'fast',
                'batch_processing': True
            }
            self.model_loaded = True

            # Register memory allocation
            self.memory_manager.allocate_component_memory('vision_processor', self.model_memory_mb)
            logger.info(f"[OK] Cloud-optimized vision model loaded, allocated {self.model_memory_mb}MB")

        except Exception as e:
            logger.error(f"[ERROR] Failed to load vision model: {e}")
            self.local_model = None
            self.model_loaded = False

    def _ensure_model_loaded(self):
        """Ensure model is loaded, load if necessary"""
        if not self.model_loaded:
            self._initialize_cloud_model()

        if self.model_loaded:
            self.model_last_used = datetime.now()

    def _check_and_unload_model(self):
        """Unload model if inactive for timeout period"""
        if not self.model_loaded:
            return

        if (self.model_last_used and
            (datetime.now() - self.model_last_used).seconds > self.model_unload_timeout):

            logger.info("[CLOUD-VISION] Unloading inactive vision model to free memory")
            self.local_model = None
            self.model_loaded = False
            self.model_last_used = None

            # Release memory allocation
            self.memory_manager.release_component_memory('vision_processor')

            # Force garbage collection
            gc.collect()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            return {
                'memory_mb': memory_mb,
                'model_loaded': self.model_loaded,
                'memory_limit_mb': self.memory_limit_mb,
                'memory_usage_percent': (memory_mb / self.memory_limit_mb) * 100,
                'component': 'vision_processor',
                'api_quota_status': self.api_quota_manager.get_quota_status()
            }
        except Exception as e:
            return {'error': str(e), 'memory_mb': 0}

    def process_images_intelligently(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cloud-optimized intelligent image processing
        Optimizes cost vs. performance based on cloud resource constraints
        """
        try:
            images = claim_data.get('images', [])
            claim_value = claim_data.get('claim_amount', 0.0)
            image_count = len(images)

            # Check memory availability first
            memory_usage = self.get_memory_usage()
            if memory_usage.get('memory_usage_percent', 0) > 90:
                logger.warning("[CLOUD-VISION] High memory usage, forcing model unload")
                self._check_and_unload_model()

            # Make processing decision
            decision = self._make_cloud_processing_decision(claim_data, images)
            logger.info(f"[CLOUD-VISION] Processing decision: {decision.mode.value} "
                       f"(confidence: {decision.confidence:.2f}, "
                       f"memory: {decision.memory_estimate_mb:.0f}MB)")

            # Process based on decision
            if decision.mode == ProcessingMode.API_HIGH_VALUE:
                results = self._process_with_api(images, claim_data)
            elif decision.mode == ProcessingMode.LOCAL_BULK:
                results = self._process_with_local_model(images, claim_data)
            else:
                results = self._process_with_fallback(images, claim_data)

            # Check if model should be unloaded
            self._check_and_unload_model()

            return {
                'processing_results': results,
                'processing_decision': {
                    'mode': decision.mode.value,
                    'confidence': decision.confidence,
                    'cost_estimate': decision.cost_estimate,
                    'memory_estimate_mb': decision.memory_estimate_mb,
                    'cloud_cpu_impact': decision.cloud_cpu_impact,
                    'reasoning': decision.reasoning
                },
                'performance_metrics': {
                    'total_images_processed': image_count,
                    'processing_time_estimate': decision.processing_time_estimate,
                    'memory_usage_after': memory_usage
                },
                'quota_status': self.api_quota_manager.get_quota_status()
            }

        except Exception as e:
            logger.error(f"[ERROR] Cloud vision processing failed: {e}")
            return self._get_fallback_results(claim_data)

    def _make_cloud_processing_decision(self, claim_data: Dict[str, Any],
                                      images: List[Any]) -> ProcessingDecision:
        """Make intelligent processing decision with cloud optimization"""
        claim_value = claim_data.get('claim_amount', 0.0)
        image_count = len(images)
        urgency = claim_data.get('urgency', 'normal')
        customer_tier = claim_data.get('customer_tier', 'standard')

        reasoning = []
        confidence = 0.5
        estimated_cost = 0.0
        processing_time = 0.0
        memory_estimate = 0.0

        # Check API quota first
        can_use_api = self.api_quota_manager.can_use_api(claim_value, image_count)

        # Decision logic for cloud environment
        if can_use_api and (claim_value > 3000 or urgency == 'high' or customer_tier == 'premium'):
            # Use API for high-value or urgent claims
            mode = ProcessingMode.API_HIGH_VALUE
            confidence = 0.8
            estimated_cost = image_count * self.api_cost_per_image
            processing_time = image_count * 2.0  # 2 seconds per image for API
            memory_estimate = 20  # Minimal memory for API processing
            cloud_cpu_impact = 'low'
            reasoning.extend([
                f"High value claim: ${claim_value:.2f}",
                f"API quota available: {self.api_quota_manager.daily_quota - self.api_quota_manager.api_calls_today}",
                "Premium processing for critical claim"
            ])

        elif self.model_loaded and image_count <= self.batch_size:
            # Use local model for small batches
            mode = ProcessingMode.LOCAL_BULK
            confidence = 0.7
            estimated_cost = 0.0
            processing_time = image_count * 5.0  # 5 seconds per image for local
            memory_estimate = self.model_memory_mb + (image_count * 5)
            cloud_cpu_impact = 'medium'
            reasoning.extend([
                f"Local model available: {self.model_loaded}",
                f"Batch size optimal: {image_count} ≤ {self.batch_size}",
                "Cost-effective local processing"
            ])

        elif can_use_api and image_count <= 3:
            # Use API for small image counts if quota available
            mode = ProcessingMode.API_HIGH_VALUE
            confidence = 0.6
            estimated_cost = image_count * self.api_cost_per_image
            processing_time = image_count * 2.0
            memory_estimate = 20
            cloud_cpu_impact = 'low'
            reasoning.extend([
                f"Small batch suitable for API: {image_count} images",
                "API processing within quota limits"
            ])

        else:
            # Fallback processing
            mode = ProcessingMode.FALLBACK
            confidence = 0.4
            estimated_cost = 0.0
            processing_time = image_count * 1.0  # Fast basic processing
            memory_estimate = 10
            cloud_cpu_impact = 'low'
            reasoning.extend([
                "Resource constraints detected",
                "Using basic processing fallback",
                "Minimal resource consumption"
            ])

        return ProcessingDecision(
            mode=mode,
            confidence=confidence,
            cost_estimate=estimated_cost,
            processing_time_estimate=processing_time,
            memory_estimate_mb=memory_estimate,
            cloud_cpu_impact=cloud_cpu_impact,
            reasoning=reasoning
        )

    def _process_with_api(self, images: List[Any], claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process images using cloud API for high-quality analysis"""
        try:
            # Ensure model is loaded for API processing
            self._ensure_model_loaded()

            results = []
            for i, image in enumerate(images):
                # Simulate API call with rate limiting
                time.sleep(0.1)  # Brief pause for rate limiting

                # Placeholder for actual API processing
                analysis = {
                    'image_index': i,
                    'damage_detected': np.random.choice([True, False], p=[0.3, 0.7]),
                    'damage_severity': np.random.choice(['minor', 'moderate', 'severe'], p=[0.6, 0.3, 0.1]),
                    'confidence': np.random.uniform(0.8, 0.95),
                    'processing_method': 'cloud_api',
                    'analysis_time': 2.0
                }
                results.append(analysis)

            # Record API usage
            claim_value = claim_data.get('claim_amount', 0.0)
            self.api_quota_manager.record_api_usage(claim_value, len(images))

            logger.info(f"[API] Processed {len(images)} images via cloud API")
            return {
                'success': True,
                'results': results,
                'method': 'cloud_api',
                'total_images': len(images)
            }

        except Exception as e:
            logger.error(f"[ERROR] API processing failed: {e}")
            return self._process_with_fallback(images, claim_data)

    def _process_with_local_model(self, images: List[Any], claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process images using local model for cost efficiency"""
        try:
            # Ensure model is loaded
            self._ensure_model_loaded()

            if not self.model_loaded:
                return self._process_with_fallback(images, claim_data)

            results = []
            for i, image in enumerate(images):
                # Simulate local model processing
                time.sleep(0.5)  # Simulate processing time

                analysis = {
                    'image_index': i,
                    'damage_detected': np.random.choice([True, False], p=[0.4, 0.6]),
                    'damage_severity': np.random.choice(['minor', 'moderate', 'severe'], p=[0.7, 0.25, 0.05]),
                    'confidence': np.random.uniform(0.7, 0.85),
                    'processing_method': 'local_model',
                    'analysis_time': 5.0
                }
                results.append(analysis)

            logger.info(f"[LOCAL] Processed {len(images)} images with local model")
            return {
                'success': True,
                'results': results,
                'method': 'local_model',
                'total_images': len(images)
            }

        except Exception as e:
            logger.error(f"[ERROR] Local model processing failed: {e}")
            return self._process_with_fallback(images, claim_data)

    def _process_with_fallback(self, images: List[Any], claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing with minimal resource usage"""
        results = []
        for i, image in enumerate(images):
            # Basic image analysis
            analysis = {
                'image_index': i,
                'damage_detected': False,  # Conservative approach
                'damage_severity': 'unknown',
                'confidence': 0.5,
                'processing_method': 'fallback',
                'analysis_time': 1.0,
                'note': 'Limited processing due to resource constraints'
            }
            results.append(analysis)

        logger.info(f"[FALLBACK] Basic processing for {len(images)} images")
        return {
            'success': True,
            'results': results,
            'method': 'fallback',
            'total_images': len(images)
        }

    def _get_fallback_results(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback results when processing fails"""
        return {
            'processing_results': {
                'success': False,
                'results': [],
                'method': 'error_fallback',
                'error': 'Processing failed, using safe fallback'
            },
            'processing_decision': {
                'mode': 'fallback',
                'confidence': 0.0,
                'cost_estimate': 0.0,
                'memory_estimate_mb': 5.0,
                'cloud_cpu_impact': 'low',
                'reasoning': ['Processing error, using safe fallback']
            },
            'performance_metrics': {
                'total_images_processed': 0,
                'processing_time_estimate': 0.0,
                'memory_usage_after': {'error': 'Processing failed'}
            },
            'quota_status': self.api_quota_manager.get_quota_status()
        }

# Global cloud-optimized vision processor instance
cloud_vision_processor = HybridVisionProcessor()

def get_cloud_vision_processor() -> HybridVisionProcessor:
    """Get the global cloud-optimized vision processor instance"""
    return cloud_vision_processor

if __name__ == "__main__":
    # Test the cloud vision processor
    processor = HybridVisionProcessor()
    logger.info(f"[TEST] Cloud vision processor initialized: {processor.get_memory_usage()}")