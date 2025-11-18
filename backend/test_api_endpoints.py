"""
API Endpoint Testing Script for Enhanced Insurance Fraud Detection System
Tests all new and existing FastAPI endpoints
"""

import requests
import json
import time
from typing import Dict, Any, Optional
import unittest
from datetime import datetime

class EnhancedAPITester:
    """Test suite for enhanced API endpoints"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []

    def log_test(self, endpoint: str, method: str, status_code: int, success: bool, details: str = ""):
        """Log test results"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'success': success,
            'details': details
        }
        self.test_results.append(result)

        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {method} {endpoint} - {status_code} - {details}")

    def test_health_endpoint(self) -> bool:
        """Test health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            success = response.status_code == 200

            if success:
                data = response.json()
                has_enhanced = 'enhanced_components' in data.get('services', {})
                details = f"Enhanced components: {has_enhanced}"
                self.log_test("/health", "GET", response.status_code, success, details)
            else:
                self.log_test("/health", "GET", response.status_code, success, response.text)

            return success

        except Exception as e:
            self.log_test("/health", "GET", 0, False, str(e))
            return False

    def test_root_endpoint(self) -> bool:
        """Test root endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/")
            success = response.status_code == 200

            if success:
                data = response.json()
                version = data.get('version', 'unknown')
                details = f"Version: {version}"
                self.log_test("/", "GET", response.status_code, success, details)
            else:
                self.log_test("/", "GET", response.status_code, success, response.text)

            return success

        except Exception as e:
            self.log_test("/", "GET", 0, False, str(e))
            return False

    def test_enhanced_text_classification(self) -> bool:
        """Test enhanced text classification endpoint"""
        try:
            claim_data = {
                "claim_data": {
                    "claim_id": "test_enhanced_text_001",
                    "customer_id": "cust_test_123",
                    "policy_number": "POL_TEST123",
                    "claim_type": "auto",
                    "description": "Vehicle collision at highway intersection during evening rush hour causing significant damage to front bumper and radiator",
                    "amount": 7500.0,
                    "location": "highway_intersection"
                },
                "text_data": "Vehicle collision at highway intersection during evening rush hour causing significant damage to front bumper and radiator. Driver reports vehicle was totaled and requires immediate settlement for replacement vehicle."
            }

            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/enhanced_text_classification",
                json=claim_data,
                headers={"Content-Type": "application/json"}
            )
            processing_time = time.time() - start_time

            success = response.status_code == 200

            if success:
                data = response.json()
                if data.get('success'):
                    result_data = data.get('data', {})
                    model_type = result_data.get('model_type', 'unknown')
                    processing_time_ms = result_data.get('processing_time_ms', 0)
                    details = f"Model: {model_type}, Time: {processing_time_ms:.1f}ms"
                    self.log_test("/enhanced_text_classification", "POST", response.status_code, success, details)
                else:
                    self.log_test("/enhanced_text_classification", "POST", response.status_code, False, data.get('message', 'Unknown error'))
            else:
                self.log_test("/enhanced_text_classification", "POST", response.status_code, False, response.text)

            return success

        except Exception as e:
            self.log_test("/enhanced_text_classification", "POST", 0, False, str(e))
            return False

    def test_enhanced_feature_generation(self) -> bool:
        """Test enhanced feature generation endpoint"""
        try:
            claim_data = {
                "claim_data": {
                    "claim_id": "test_features_001",
                    "customer_id": "cust_feature_456",
                    "policy_number": "POL_FEATURE456",
                    "claim_type": "auto",
                    "description": "Minor collision in parking lot with scratches on passenger side door, witness present, police report filed",
                    "amount": 2500.0,
                    "location": "parking_lot",
                    "accident_time": "14:30",
                    "accident_date": "2024-11-18",
                    "claimant_age": 35,
                    "policy_age_months": 24,
                    "witness_count": 2,
                    "police_report": "yes"
                }
            }

            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/enhanced_feature_generation",
                json=claim_data,
                headers={"Content-Type": "application/json"}
            )
            processing_time = time.time() - start_time

            success = response.status_code == 200

            if success:
                data = response.json()
                if data.get('success'):
                    result_data = data.get('data', {})
                    feature_count = result_data.get('feature_statistics', {}).get('total_features', 0)
                    processing_time_ms = result_data.get('processing_time_ms', 0)
                    meets_target = result_data.get('meets_target', False)
                    details = f"Features: {feature_count}, Time: {processing_time_ms:.1f}ms, Target: {'✅' if meets_target else '❌'}"
                    self.log_test("/enhanced_feature_generation", "POST", response.status_code, success, details)
                else:
                    self.log_test("/enhanced_feature_generation", "POST", response.status_code, False, data.get('message', 'Unknown error'))
            else:
                self.log_test("/enhanced_feature_generation", "POST", response.status_code, False, response.text)

            return success

        except Exception as e:
            self.log_test("/enhanced_feature_generation", "POST", 0, False, str(e))
            return False

    def test_performance_validation(self) -> bool:
        """Test performance validation endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/performance_validation")
            success = response.status_code == 200

            if success:
                data = response.json()
                if data.get('success'):
                    result_data = data.get('data', {})
                    dashboard_data = result_data.get('dashboard_data', {})
                    session_stats = dashboard_data.get('real_time_metrics', {})
                    total_requests = session_stats.get('total_requests', 0)
                    success_rate = session_stats.get('success_rate', 0)
                    details = f"Requests: {total_requests}, Success rate: {success_rate:.1%}"
                    self.log_test("/performance_validation", "GET", response.status_code, success, details)
                else:
                    self.log_test("/performance_validation", "GET", response.status_code, False, data.get('message', 'Unknown error'))
            else:
                self.log_test("/performance_validation", "GET", response.status_code, False, response.text)

            return success

        except Exception as e:
            self.log_test("/performance_validation", "GET", 0, False, str(e))
            return False

    def test_memory_optimization(self) -> bool:
        """Test memory optimization endpoint"""
        try:
            response = self.session.post(f"{self.base_url}/optimize_memory")
            success = response.status_code == 200

            if success:
                data = response.json()
                if data.get('success'):
                    result_data = data.get('data', {})
                    memory_freed = result_data.get('memory_freed_mb', 0)
                    efficiency_score = result_data.get('current_efficiency_score', 0)
                    details = f"Freed: {memory_freed:.1f}MB, Efficiency: {efficiency_score:.2f}"
                    self.log_test("/optimize_memory", "POST", response.status_code, success, details)
                else:
                    self.log_test("/optimize_memory", "POST", response.status_code, False, data.get('message', 'Unknown error'))
            else:
                self.log_test("/optimize_memory", "POST", response.status_code, False, response.text)

            return success

        except Exception as e:
            self.log_test("/optimize_memory", "POST", 0, False, str(e))
            return False

    def test_ab_testing(self) -> bool:
        """Test A/B testing endpoint"""
        try:
            # Use form data for this endpoint
            form_data = {
                'test_name': 'api_integration_test',
                'component_type': 'text_classifier',
                'test_iterations': '10'
            }

            response = self.session.post(
                f"{self.base_url}/run_ab_test",
                data=form_data
            )
            success = response.status_code == 200

            if success:
                data = response.json()
                if data.get('success'):
                    result_data = data.get('data', {})
                    ab_result = result_data.get('ab_test_result', {})
                    improvement = ab_result.get('improvement_percentage', 0)
                    is_significant = ab_result.get('is_significant', False)
                    details = f"Improvement: {improvement:.1f}%, Significant: {'✅' if is_significant else '❌'}"
                    self.log_test("/run_ab_test", "POST", response.status_code, success, details)
                else:
                    self.log_test("/run_ab_test", "POST", response.status_code, False, data.get('message', 'Unknown error'))
            else:
                self.log_test("/run_ab_test", "POST", response.status_code, False, response.text)

            return success

        except Exception as e:
            self.log_test("/run_ab_test", "POST", 0, False, str(e))
            return False

    def test_system_benchmark(self) -> bool:
        """Test system benchmark endpoint"""
        try:
            form_data = {
                'component_name': 'enhanced_text_classifier',
                'test_iterations': '20'
            }

            response = self.session.post(
                f"{self.base_url}/run_benchmark",
                data=form_data
            )
            success = response.status_code == 200

            if success:
                data = response.json()
                if data.get('success'):
                    result_data = data.get('data', {})
                    benchmark_result = result_data.get('benchmark_result', {})
                    avg_time = benchmark_result.get('avg_processing_time_ms', 0)
                    throughput = benchmark_result.get('throughput_per_second', 0)
                    success_rate = benchmark_result.get('success_rate', 0)
                    details = f"Avg time: {avg_time:.1f}ms, Throughput: {throughput:.1f}/s, Success: {success_rate:.1%}"
                    self.log_test("/run_benchmark", "POST", response.status_code, success, details)
                else:
                    self.log_test("/run_benchmark", "POST", response.status_code, False, data.get('message', 'Unknown error'))
            else:
                self.log_test("/run_benchmark", "POST", response.status_code, False, response.text)

            return success

        except Exception as e:
            self.log_test("/run_benchmark", "POST", 0, False, str(e))
            return False

    def test_standard_endpoints(self) -> bool:
        """Test standard endpoints"""
        endpoints = [
            ("/search_claims", "POST", {
                "query": "vehicle collision highway intersection",
                "modality": "text_claims",
                "limit": 3
            }),
            ("/collections", "GET", None),
            ("/system_info", "GET", None)
        ]

        all_success = True

        for endpoint, method, payload in endpoints:
            try:
                if method == "GET":
                    response = self.session.get(f"{self.base_url}{endpoint}")
                else:  # POST
                    response = self.session.post(
                        f"{self.base_url}{endpoint}",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )

                success = response.status_code == 200
                if success:
                    details = f"Response received"
                else:
                    details = response.text[:100]

                self.log_test(endpoint, method, response.status_code, success, details)
                all_success = all_success and success

            except Exception as e:
                self.log_test(endpoint, method, 0, False, str(e))
                all_success = False

        return all_success

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests and return summary"""
        print("=" * 80)
        print("🧪 ENHANCED API ENDPOINT TESTING")
        print("=" * 80)
        print(f"Testing endpoints at: {self.base_url}")
        print()

        # Test basic connectivity first
        if not self.test_health_endpoint():
            print("\n❌ Server is not responding. Please ensure the API server is running.")
            return {"success": False, "details": "Server not responding"}

        # Run all tests
        test_functions = [
            self.test_root_endpoint,
            self.test_enhanced_text_classification,
            self.test_enhanced_feature_generation,
            self.test_performance_validation,
            self.test_memory_optimization,
            self.test_ab_testing,
            self.test_system_benchmark,
            self.test_standard_endpoints
        ]

        for test_func in test_functions:
            try:
                test_func()
                time.sleep(0.5)  # Small delay between tests
            except Exception as e:
                print(f"❌ Test {test_func.__name__} failed with exception: {e}")

        # Calculate summary
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result['success'])
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success rate: {success_rate:.1f}%")

        if success_rate >= 80:
            print("\n🎉 API testing completed successfully! Enhanced system is ready.")
        elif success_rate >= 60:
            print("\n⚠️  API testing partially successful. Some components may need attention.")
        else:
            print("\n❌ API testing failed. Please check server logs and fix issues.")

        # Detailed results
        print("\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status_icon = "✅" if result['success'] else "❌"
            print(f"  {status_icon} {result['method']:6} {result['endpoint']:30} - {result['details']}")

        return {
            "success": success_rate >= 80,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "results": self.test_results
        }

def main():
    """Main function to run API tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Enhanced Insurance Fraud Detection API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", help="Output JSON file for test results")

    args = parser.parse_args()

    tester = EnhancedAPITester(args.url)
    results = tester.run_all_tests()

    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Test results saved to: {args.output}")

    return 0 if results["success"] else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())