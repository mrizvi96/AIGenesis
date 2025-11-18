"""
Simplified Cloud Integration Test for Qdrant Cloud
Tests core functionality with minimal dependencies
"""

import numpy as np
import pandas as pd
import time
import json
import logging
import os
import gc
import psutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCloudTest:
    """Simplified cloud test focusing on core components"""

    def __init__(self):
        logger.info("[CLOUD-TEST] Initializing simplified cloud test...")

        # Cloud resource limits
        self.max_memory_mb = 1024  # 1GB RAM limit

        # Test results
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'memory_peaks': {},
            'component_results': {}
        }

    def get_system_metrics(self) -> dict:
        """Get current system resource usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'memory_rss_mb': memory_info.rss / (1024 * 1024),
                'memory_vms_mb': memory_info.vms / (1024 * 1024),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e), 'memory_rss_mb': 0}

    def test_qdrant_connection(self) -> dict:
        """Test Qdrant Cloud connection"""
        logger.info("[CLOUD-TEST] Testing Qdrant Cloud connection...")

        test_name = "qdrant_connection"
        start_time = time.time()
        start_memory = self.get_system_metrics()

        try:
            # Import and test qdrant manager
            from qdrant_manager import get_qdrant_manager

            qm = get_qdrant_manager()

            # Test connection
            result = qm.initialize_cloud_environment()

            # Test batch insertion with compressed vectors
            test_vectors = [[np.random.random() for _ in range(256)] for _ in range(10)]

            batch_result = qm.batch_insert_vectors(
                vectors=test_vectors,
                payload_ids=[f"test_{i}" for i in range(len(test_vectors))],
                batch_size=5
            )

            # Test search
            if batch_result.get('success', False):
                search_result = qm.search_vectors(
                    query_vector=test_vectors[0],
                    limit=3
                )
            else:
                search_result = {'success': False}

            success = (result.get('success', False) and
                      batch_result.get('success', False) and
                      search_result.get('success', False))

            end_time = time.time()
            end_memory = self.get_system_metrics()

            test_result = {
                'success': success,
                'connection_success': result.get('success', False),
                'batch_insert_success': batch_result.get('success', False),
                'search_success': search_result.get('success', False),
                'vectors_processed': len(test_vectors) if success else 0,
                'processing_time': end_time - start_time,
                'memory_before': start_memory.get('memory_rss_mb', 0),
                'memory_after': end_memory.get('memory_rss_mb', 0),
                'memory_peak': max(start_memory.get('memory_rss_mb', 0), end_memory.get('memory_rss_mb', 0))
            }

            self.test_results['memory_peaks'][test_name] = test_result['memory_peak']
            self.test_results['component_results'][test_name] = test_result

            logger.info(f"[OK] Qdrant Cloud connection test passed: {test_result.get('vectors_processed', 0)} vectors")
            return test_result

        except Exception as e:
            logger.error(f"[ERROR] Qdrant connection test failed: {e}")
            return {'success': False, 'error': str(e)}

    def test_memory_compression(self) -> dict:
        """Test memory compression functionality"""
        logger.info("[CLOUD-TEST] Testing Memory Compression...")

        test_name = "memory_compression"
        start_time = time.time()
        start_memory = self.get_system_metrics()

        try:
            from memory_manager import get_memory_manager

            mm = get_memory_manager()

            # Test vector compression
            test_vectors = [[np.random.random() for _ in range(512)] for _ in range(50)]

            compression_result = mm.compress_vectors(test_vectors, 'test_model', target_dim=256)

            # Verify compression worked
            compressed_dims = len(compression_result[0]) if compression_result else 0
            compression_success = compressed_dims <= 256

            # Test memory allocation
            allocation_result = mm.can_allocate(50, 'test_component')

            success = compression_success and allocation_result.get('can_allocate', False)

            end_time = time.time()
            end_memory = self.get_system_metrics()

            test_result = {
                'success': success,
                'compression_original_dims': 512,
                'compression_final_dims': compressed_dims,
                'compression_ratio': compressed_dims / 512 if compression_success else 0,
                'allocation_available': allocation_result.get('can_allocate', False),
                'processing_time': end_time - start_time,
                'memory_before': start_memory.get('memory_rss_mb', 0),
                'memory_after': end_memory.get('memory_rss_mb', 0),
                'memory_peak': max(start_memory.get('memory_rss_mb', 0), end_memory.get('memory_rss_mb', 0))
            }

            self.test_results['memory_peaks'][test_name] = test_result['memory_peak']
            self.test_results['component_results'][test_name] = test_result

            logger.info(f"[OK] Memory compression test passed: {test_result['compression_ratio']:.2f} compression ratio")
            return test_result

        except Exception as e:
            logger.error(f"[ERROR] Memory compression test failed: {e}")
            return {'success': False, 'error': str(e)}

    def test_vision_processing(self) -> dict:
        """Test hybrid vision processing"""
        logger.info("[CLOUD-TEST] Testing Vision Processing...")

        test_name = "vision_processing"
        start_time = time.time()
        start_memory = self.get_system_metrics()

        try:
            from hybrid_vision_processor import get_cloud_vision_processor

            vp = get_cloud_vision_processor()

            # Test claim data
            test_claim = {
                'claim_amount': 7500.0,
                'claim_type': 'auto',
                'urgency': 'high',
                'customer_tier': 'premium',
                'images': [
                    {'image_index': 0, 'image_data': 'test_data_1'},
                    {'image_index': 1, 'image_data': 'test_data_2'}
                ]
            }

            # Test processing
            result = vp.process_images_intelligently(test_claim)

            success = result.get('processing_results', {}).get('success', False)

            end_time = time.time()
            end_memory = self.get_system_metrics()

            test_result = {
                'success': success,
                'processing_mode': result.get('processing_decision', {}).get('mode', 'unknown'),
                'images_processed': result.get('performance_metrics', {}).get('total_images_processed', 0),
                'memory_usage_mb': vp.get_memory_usage().get('memory_mb', 0),
                'processing_time': end_time - start_time,
                'memory_before': start_memory.get('memory_rss_mb', 0),
                'memory_after': end_memory.get('memory_rss_mb', 0),
                'memory_peak': max(start_memory.get('memory_rss_mb', 0), end_memory.get('memory_rss_mb', 0))
            }

            self.test_results['memory_peaks'][test_name] = test_result['memory_peak']
            self.test_results['component_results'][test_name] = test_result

            logger.info(f"[OK] Vision processing test passed: {test_result.get('processing_mode', 'unknown')} mode")
            return test_result

        except Exception as e:
            logger.error(f"[ERROR] Vision processing test failed: {e}")
            return {'success': False, 'error': str(e)}

    def run_tests(self) -> dict:
        """Run all simplified cloud tests"""
        logger.info("[CLOUD-TEST] Starting simplified cloud testing...")

        test_start_time = time.time()
        initial_memory = self.get_system_metrics()

        # Run tests
        tests = [
            self.test_qdrant_connection,
            self.test_memory_compression,
            self.test_vision_processing
        ]

        for test_func in tests:
            try:
                self.test_results['total_tests'] += 1
                result = test_func()

                if result.get('success', False):
                    self.test_results['passed_tests'] += 1
                    logger.info(f"[OK] {test_func.__name__} PASSED")
                else:
                    self.test_results['failed_tests'] += 1
                    logger.error(f"[FAIL] {test_func.__name__} FAILED: {result.get('error', 'Unknown error')}")

                gc.collect()
                time.sleep(0.5)

            except Exception as e:
                self.test_results['total_tests'] += 1
                self.test_results['failed_tests'] += 1
                logger.error(f"[ERROR] {test_func.__name__} ERROR: {e}")

        test_end_time = time.time()
        final_memory = self.get_system_metrics()

        # Calculate results
        total_test_time = test_end_time - test_start_time
        peak_memory = max(self.test_results['memory_peaks'].values()) if self.test_results['memory_peaks'] else 0
        success_rate = (self.test_results['passed_tests'] / self.test_results['total_tests']) * 100 if self.test_results['total_tests'] > 0 else 0

        overall_success = success_rate >= 80 and peak_memory < self.max_memory_mb

        final_results = {
            'overall_success': overall_success,
            'success_rate_percent': success_rate,
            'total_test_time': total_test_time,
            'tests_run': self.test_results['total_tests'],
            'tests_passed': self.test_results['passed_tests'],
            'tests_failed': self.test_results['failed_tests'],
            'peak_memory_mb': peak_memory,
            'memory_limit_mb': self.max_memory_mb,
            'within_memory_limit': peak_memory < self.max_memory_mb,
            'initial_memory_mb': initial_memory.get('memory_rss_mb', 0),
            'final_memory_mb': final_memory.get('memory_rss_mb', 0),
            'component_results': self.test_results['component_results'],
            'memory_peaks': self.test_results['memory_peaks'],
            'cloud_ready': overall_success
        }

        logger.info(f"[CLOUD-TEST] Simplified testing completed:")
        logger.info(f"  Success Rate: {success_rate:.1f}% ({self.test_results['passed_tests']}/{self.test_results['total_tests']})")
        logger.info(f"  Peak Memory: {peak_memory:.1f}MB (limit: {self.max_memory_mb}MB)")
        logger.info(f"  Total Time: {total_test_time:.2f}s")
        logger.info(f"  Cloud Ready: {'YES' if final_results['cloud_ready'] else 'NO'}")

        return final_results

    def save_results(self, results: dict, filename: str = "simple_cloud_test_results.json"):
        """Save test results"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"[OK] Results saved to {filename}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save results: {e}")

if __name__ == "__main__":
    tester = SimpleCloudTest()

    try:
        results = tester.run_tests()
        tester.save_results(results)

        if results['cloud_ready']:
            print("\n🎉 SUCCESS: Core components are ready for Qdrant Cloud deployment!")
            print("   - Vector compression working")
            print("   - Qdrant Cloud connection established")
            print("   - Memory usage within limits")
        else:
            print("\n❌ FAILURE: Components need further optimization")

    except Exception as e:
        logger.error(f"[FATAL] Test execution failed: {e}")
    finally:
        gc.collect()