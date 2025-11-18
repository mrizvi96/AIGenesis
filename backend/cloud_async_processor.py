"""
Cloud-Optimized Async Processing and Queuing System
Handles resource constraints in Qdrant Cloud Free Tier with intelligent task scheduling
Implements progressive loading, batch processing, and automatic resource management
"""

import asyncio
import logging
import time
import queue
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels for cloud resource optimization"""
    CRITICAL = 1    # High-value claims, urgent processing
    HIGH = 2        # Premium customers, time-sensitive
    NORMAL = 3      # Standard processing
    LOW = 4         # Bulk processing, background tasks
    CLEANUP = 5     # Resource cleanup tasks

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

@dataclass
class CloudTask:
    """Cloud-optimized task with resource management"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    claim_data: Optional[Dict[str, Any]] = None

    # Resource requirements
    estimated_memory_mb: float = 0.0
    estimated_cpu_percent: float = 0.0
    estimated_time_seconds: float = 0.0

    # Cloud optimization settings
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 30
    can_fallback: bool = True
    batch_size: int = 1

    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Execution tracking
    result: Optional[Dict[str, Any]] = None
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

class CloudResourceMonitor:
    """Monitors cloud resource usage and constraints"""

    def __init__(self, memory_limit_mb: int = 1024, cpu_limit_percent: float = 50.0):
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_percent = cpu_limit_percent
        self.current_memory_mb = 0.0
        self.current_cpu_percent = 0.0
        self.active_tasks = 0
        self.max_concurrent_tasks = 3  # Cloud-optimized concurrency

    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            self.current_memory_mb = memory_info.rss / (1024 * 1024)
            self.current_cpu_percent = process.cpu_percent()

        except Exception as e:
            logger.warning(f"[WARN] Resource monitoring failed: {e}")

        return {
            'memory_mb': self.current_memory_mb,
            'memory_limit_mb': self.memory_limit_mb,
            'memory_usage_percent': (self.current_memory_mb / self.memory_limit_mb) * 100,
            'cpu_percent': self.current_cpu_percent,
            'cpu_limit_percent': self.cpu_limit_percent,
            'active_tasks': self.active_tasks,
            'max_concurrent_tasks': self.max_concurrent_tasks
        }

    def can_execute_task(self, task: CloudTask) -> bool:
        """Check if task can be executed within resource constraints"""
        usage = self.get_current_usage()

        # Check memory constraints
        projected_memory = usage['memory_mb'] + task.estimated_memory_mb
        if projected_memory > self.memory_limit_mb * 0.8:  # 80% safety margin
            return False

        # Check CPU constraints
        projected_cpu = usage['cpu_percent'] + task.estimated_cpu_percent
        if projected_cpu > self.cpu_limit_percent * 0.8:  # 80% safety margin
            return False

        # Check concurrent task limits
        if self.active_tasks >= self.max_concurrent_tasks:
            return False

        return True

class CloudAsyncProcessor:
    """Cloud-optimized async task processor with intelligent scheduling"""

    def __init__(self, memory_limit_mb: int = 1024):
        logger.info(f"[CLOUD-ASYNC] Initializing async processor (memory_limit={memory_limit_mb}MB)...")

        # Resource management
        self.resource_monitor = CloudResourceMonitor(memory_limit_mb)
        self.memory_limit_mb = memory_limit_mb

        # Task queues
        self.queues = {
            TaskPriority.CRITICAL: queue.PriorityQueue(),
            TaskPriority.HIGH: queue.PriorityQueue(),
            TaskPriority.NORMAL: queue.PriorityQueue(),
            TaskPriority.LOW: queue.PriorityQueue(),
            TaskPriority.CLEANUP: queue.PriorityQueue()
        }

        # Task tracking
        self.tasks: Dict[str, CloudTask] = {}
        self.completed_tasks: Dict[str, CloudTask] = {}

        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=3)  # Cloud-optimized
        self.running = False
        self.worker_thread = None

        # Performance tracking
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'average_task_time': 0.0,
            'resource_efficiency': 0.0
        }

        # Cloud optimization settings
        self.batch_processing_enabled = True
        self.auto_fallback_enabled = True
        self.progressive_loading_enabled = True
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()

        logger.info("[OK] Cloud async processor initialized")

    def submit_task(self,
                   task_type: str,
                   payload: Dict[str, Any],
                   priority: TaskPriority = TaskPriority.NORMAL,
                   claim_data: Optional[Dict[str, Any]] = None,
                   **kwargs) -> str:
        """Submit a task for cloud-optimized processing"""

        # Create task with resource estimation
        task = CloudTask(
            task_type=task_type,
            payload=payload,
            priority=priority,
            claim_data=claim_data,
            **kwargs
        )

        # Estimate resource requirements based on task type
        self._estimate_task_resources(task)

        # Store task
        self.tasks[task.task_id] = task

        # Add to appropriate queue
        priority_value = task.priority.value
        self.queues[task.priority].put((priority_value, time.time(), task))

        self.stats['total_tasks'] += 1

        logger.info(f"[TASK] Submitted {task_type} task (ID: {task.task_id[:8]}, "
                   f"priority: {task.priority.name}, memory: {task.estimated_memory_mb:.1f}MB)")

        return task.task_id

    def _estimate_task_resources(self, task: CloudTask):
        """Estimate resource requirements for cloud optimization"""

        # Base resource requirements by task type
        resource_estimates = {
            'classification': {
                'memory_mb': 25.0,
                'cpu_percent': 15.0,
                'time_seconds': 3.0
            },
            'vision_processing': {
                'memory_mb': 80.0,
                'cpu_percent': 25.0,
                'time_seconds': 5.0
            },
            'feature_generation': {
                'memory_mb': 60.0,
                'cpu_percent': 20.0,
                'time_seconds': 2.0
            },
            'vector_search': {
                'memory_mb': 15.0,
                'cpu_percent': 10.0,
                'time_seconds': 1.0
            },
            'fusion_processing': {
                'memory_mb': 40.0,
                'cpu_percent': 18.0,
                'time_seconds': 3.0
            },
            'data_storage': {
                'memory_mb': 20.0,
                'cpu_percent': 12.0,
                'time_seconds': 2.0
            }
        }

        # Get base estimate
        base_estimate = resource_estimates.get(task.task_type, {
            'memory_mb': 30.0,
            'cpu_percent': 15.0,
            'time_seconds': 2.0
        })

        # Adjust based on claim value and priority
        claim_value = task.claim_data.get('claim_amount', 0.0) if task.claim_data else 0.0
        value_multiplier = 1.0 + min(claim_value / 10000, 0.5)  # Up to 50% increase for high-value claims

        # Apply multipliers
        task.estimated_memory_mb = base_estimate['memory_mb'] * value_multiplier
        task.estimated_cpu_percent = base_estimate['cpu_percent'] * value_multiplier
        task.estimated_time_seconds = base_estimate['time_seconds'] * value_multiplier

        # Adjust for batch processing
        if task.batch_size > 1:
            # Non-linear scaling for batch efficiency
            batch_multiplier = 1.0 + (task.batch_size - 1) * 0.3
            task.estimated_memory_mb *= batch_multiplier
            task.estimated_time_seconds *= batch_multiplier

    def start_processing(self):
        """Start the cloud-optimized processing loop"""
        if self.running:
            logger.warning("[WARN] Processor already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.worker_thread.start()

        logger.info("[OK] Cloud async processing started")

    def stop_processing(self):
        """Stop processing and cleanup resources"""
        logger.info("[CLOUD-ASYNC] Stopping async processing...")

        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)

        self.executor.shutdown(wait=True)

        logger.info("[OK] Cloud async processing stopped")

    def _processing_loop(self):
        """Main processing loop with intelligent task scheduling"""

        while self.running:
            try:
                # Check for cleanup tasks
                if time.time() - self.last_cleanup > self.cleanup_interval:
                    self._perform_cleanup()
                    self.last_cleanup = time.time()

                # Get next task by priority
                task = self._get_next_task()
                if not task:
                    time.sleep(0.1)  # Brief pause if no tasks
                    continue

                # Check resource availability
                if not self.resource_monitor.can_execute_task(task):
                    # Task can't run now, add to retry queue
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.status = TaskStatus.RETRYING
                        self.queues[task.priority].put((task.priority.value + 1, time.time() + 1.0, task))
                        logger.info(f"[RETRY] Task {task.task_id[:8]} deferred (retry {task.retry_count})")
                    continue

                # Execute task asynchronously
                self.executor.submit(self._execute_task, task)

                # Brief pause between task submissions
                time.sleep(0.05)

            except Exception as e:
                logger.error(f"[ERROR] Processing loop error: {e}")
                time.sleep(1.0)

    def _get_next_task(self) -> Optional[CloudTask]:
        """Get next task by priority"""

        # Check queues in priority order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL,
                        TaskPriority.LOW, TaskPriority.CLEANUP]:
            try:
                if not self.queues[priority].empty():
                    priority_value, timestamp, task = self.queues[priority].get_nowait()

                    # Skip if task is already running or completed
                    if task.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        continue

                    return task

            except queue.Empty:
                continue

        return None

    def _execute_task(self, task: CloudTask):
        """Execute a single task with resource monitoring"""

        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self.resource_monitor.active_tasks += 1

            logger.info(f"[EXEC] Starting {task.task_type} (ID: {task.task_id[:8]})")

            # Execute based on task type
            if task.task_type == 'classification':
                result = self._execute_classification_task(task)
            elif task.task_type == 'vision_processing':
                result = self._execute_vision_task(task)
            elif task.task_type == 'feature_generation':
                result = self._execute_feature_task(task)
            elif task.task_type == 'vector_search':
                result = self._execute_search_task(task)
            elif task.task_type == 'fusion_processing':
                result = self._execute_fusion_task(task)
            elif task.task_type == 'data_storage':
                result = self._execute_storage_task(task)
            else:
                result = {'success': False, 'error': f'Unknown task type: {task.task_type}'}

            # Update task completion
            task.status = TaskStatus.COMPLETED if result.get('success', False) else TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.result = result

            if not result.get('success', False):
                task.error_message = result.get('error', 'Unknown error')
                self.stats['failed_tasks'] += 1
            else:
                self.stats['completed_tasks'] += 1

            # Calculate execution time
            execution_time = (task.completed_at - task.started_at).total_seconds()
            task.execution_metadata['execution_time'] = execution_time
            self.stats['total_execution_time'] += execution_time

            logger.info(f"[DONE] {task.task_type} completed (ID: {task.task_id[:8]}, "
                       f"status: {task.status.value}, time: {execution_time:.2f}s)")

        except Exception as e:
            logger.error(f"[ERROR] Task execution failed (ID: {task.task_id[:8]}): {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            self.stats['failed_tasks'] += 1

        finally:
            self.resource_monitor.active_tasks -= 1
            self.completed_tasks[task.task_id] = task

    def _execute_classification_task(self, task: CloudTask) -> Dict[str, Any]:
        """Execute classification task with cloud optimization"""
        try:
            from aiml_multi_task_classifier import get_aiml_multitask_classifier

            classifier = get_aiml_multitask_classifier()
            claim_data = task.claim_data or task.payload.get('claim_data', {})

            result = classifier.classify_claim(claim_data)

            return result

        except Exception as e:
            if task.can_fallback:
                # Fallback to keyword-based classification
                return self._fallback_classification(task)
            return {'success': False, 'error': str(e)}

    def _execute_vision_task(self, task: CloudTask) -> Dict[str, Any]:
        """Execute vision processing task with cloud optimization"""
        try:
            from hybrid_vision_processor import get_cloud_vision_processor

            processor = get_cloud_vision_processor()
            claim_data = task.claim_data or task.payload.get('claim_data', {})

            result = processor.process_images_intelligently(claim_data)

            return result

        except Exception as e:
            if task.can_fallback:
                # Fallback to basic image processing
                return self._fallback_vision_processing(task)
            return {'success': False, 'error': str(e)}

    def _execute_feature_task(self, task: CloudTask) -> Dict[str, Any]:
        """Execute feature generation task with cloud optimization"""
        try:
            from enhanced_safe_features import get_cloud_safe_features

            generator = get_cloud_safe_features()
            claim_data = task.claim_data or task.payload.get('claim_data', {})

            result = generator.generate_enhanced_features_batch(claim_data)

            return result

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_search_task(self, task: CloudTask) -> Dict[str, Any]:
        """Execute vector search task with cloud optimization"""
        try:
            from qdrant_manager import get_qdrant_manager

            qdrant = get_qdrant_manager()
            query_vector = task.payload.get('query_vector', [])
            limit = task.payload.get('limit', 5)

            result = qdrant.search_vectors(query_vector, limit)

            return result

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_fusion_task(self, task: CloudTask) -> Dict[str, Any]:
        """Execute fusion processing task with cloud optimization"""
        try:
            from efficient_fusion import get_cloud_fusion_processor

            processor = get_cloud_fusion_processor()
            claim_data = task.claim_data or task.payload.get('claim_data', {})

            result = processor.fuse_features_streaming(claim_data)

            return result

        except Exception as e:
            if task.can_fallback:
                # Fallback to simple concatenation
                return self._fallback_fusion_processing(task)
            return {'success': False, 'error': str(e)}

    def _execute_storage_task(self, task: CloudTask) -> Dict[str, Any]:
        """Execute data storage task with cloud optimization"""
        try:
            from qdrant_manager import get_qdrant_manager

            qdrant = get_qdrant_manager()
            claim_data = task.claim_data or task.payload.get('claim_data', {})

            # Store claim in appropriate collection
            result = qdrant.add_claim(claim_data)

            return result

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _fallback_classification(self, task: CloudTask) -> Dict[str, Any]:
        """Fallback classification using keywords"""
        try:
            claim_data = task.claim_data or {}
            text = claim_data.get('text_description', '').lower()

            # Simple keyword-based fraud detection
            fraud_keywords = ['suspicious', 'unusual', 'emergency', 'immediate']
            fraud_score = sum(1 for keyword in fraud_keywords if keyword in text) / len(fraud_keywords)

            return {
                'success': True,
                'fraud_probability': fraud_score,
                'complexity_score': 0.5,
                'method': 'keyword_fallback',
                'cloud_optimization': 'fallback_used'
            }

        except Exception as e:
            return {'success': False, 'error': f'Fallback classification failed: {e}'}

    def _fallback_vision_processing(self, task: CloudTask) -> Dict[str, Any]:
        """Fallback vision processing"""
        try:
            images = task.claim_data.get('images', []) if task.claim_data else []

            return {
                'success': True,
                'processing_results': {
                    'success': True,
                    'results': [{'image_index': i, 'damage_detected': False} for i in range(len(images))],
                    'method': 'basic_fallback'
                },
                'processing_decision': {
                    'mode': 'fallback',
                    'confidence': 0.5
                }
            }

        except Exception as e:
            return {'success': False, 'error': f'Fallback vision processing failed: {e}'}

    def _fallback_fusion_processing(self, task: CloudTask) -> Dict[str, Any]:
        """Fallback fusion processing"""
        try:
            # Return minimal fused features
            fused_features = np.zeros(32, dtype=np.float32)  # Minimal dimensions

            return {
                'success': True,
                'fused_features': fused_features.tolist(),
                'fusion_metadata': {
                    'processing_method': 'fallback',
                    'target_dimensions': 32
                }
            }

        except Exception as e:
            return {'success': False, 'error': f'Fallback fusion failed: {e}'}

    def _perform_cleanup(self):
        """Perform resource cleanup and optimization"""
        try:
            logger.info("[CLEANUP] Performing resource cleanup...")

            # Clean up old completed tasks (keep last 100)
            if len(self.completed_tasks) > 100:
                # Sort by completion time and keep recent tasks
                sorted_tasks = sorted(
                    self.completed_tasks.items(),
                    key=lambda x: x[1].completed_at or datetime.min,
                    reverse=True
                )

                # Keep only 100 most recent tasks
                self.completed_tasks = dict(sorted_tasks[:100])

            # Force garbage collection
            import gc
            gc.collect()

            # Update performance metrics
            self._update_performance_metrics()

            logger.info("[OK] Resource cleanup completed")

        except Exception as e:
            logger.error(f"[ERROR] Cleanup failed: {e}")

    def _update_performance_metrics(self):
        """Update performance and efficiency metrics"""
        try:
            if self.stats['completed_tasks'] > 0:
                self.stats['average_task_time'] = (
                    self.stats['total_execution_time'] / self.stats['completed_tasks']
                )

            # Calculate resource efficiency
            usage = self.resource_monitor.get_current_usage()
            self.stats['resource_efficiency'] = (
                usage['memory_usage_percent'] / 80.0 * 100  # Efficiency relative to 80% target
            )

        except Exception as e:
            logger.warning(f"[WARN] Performance metrics update failed: {e}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'status': task.status.value,
            'priority': task.priority.name,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'execution_time': task.execution_metadata.get('execution_time'),
            'error_message': task.error_message,
            'result_available': task.result is not None
        }

    def get_processor_stats(self) -> Dict[str, Any]:
        """Get comprehensive processor statistics"""
        usage = self.resource_monitor.get_current_usage()

        return {
            'processor_status': 'running' if self.running else 'stopped',
            'resource_usage': usage,
            'task_statistics': {
                'total_tasks': self.stats['total_tasks'],
                'completed_tasks': self.stats['completed_tasks'],
                'failed_tasks': self.stats['failed_tasks'],
                'success_rate': (
                    (self.stats['completed_tasks'] / self.stats['total_tasks']) * 100
                    if self.stats['total_tasks'] > 0 else 0
                )
            },
            'performance_metrics': {
                'average_task_time': self.stats['average_task_time'],
                'total_execution_time': self.stats['total_execution_time'],
                'resource_efficiency': self.stats['resource_efficiency']
            },
            'queue_status': {
                priority.name: queue.qsize()
                for priority, queue in self.queues.items()
            },
            'cloud_optimization': {
                'batch_processing_enabled': self.batch_processing_enabled,
                'auto_fallback_enabled': self.auto_fallback_enabled,
                'progressive_loading_enabled': self.progressive_loading_enabled,
                'cleanup_interval': self.cleanup_interval
            }
        }

# Global cloud async processor instance
cloud_async_processor = CloudAsyncProcessor()

def get_cloud_async_processor() -> CloudAsyncProcessor:
    """Get the global cloud async processor instance"""
    return cloud_async_processor

if __name__ == "__main__":
    # Test the cloud async processor
    processor = CloudAsyncProcessor()
    processor.start_processing()

    # Submit test tasks
    task_id = processor.submit_task(
        task_type='classification',
        payload={'claim_data': {'text_description': 'Test claim', 'amount': 5000}},
        priority=TaskPriority.HIGH
    )

    print(f"Submitted task: {task_id}")

    # Monitor for a few seconds
    time.sleep(5)

    # Print statistics
    stats = processor.get_processor_stats()
    print("\nProcessor Statistics:")
    print(json.dumps(stats, indent=2, default=str))

    processor.stop_processing()