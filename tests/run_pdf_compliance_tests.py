"""
PDF Compliance Test Runner
Execute all PDF compliance tests and generate comprehensive report
"""

import unittest
import sys
import os
import json
import time
from datetime import datetime
from io import StringIO

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import test modules
from test_pdf_compliance.test_multitask_classification import *
from test_pdf_compliance.test_safe_feature_engineering import *
from test_pdf_compliance.test_integration_compliance import *
from test_pdf_compliance.test_performance_benchmarks import *

# Import utilities
from test_utils.pdf_comparator import PDFComplianceComparator
from test_utils.performance_analyzer import PerformanceAnalyzer

class PDFComplianceTestRunner:
    """Run all PDF compliance tests and generate comprehensive report"""
    
    def __init__(self):
        """Initialize test runner"""
        self.results = {
            'test_summary': {},
            'detailed_results': {},
            'compliance_analysis': {},
            'performance_analysis': {},
            'recommendations': []
        }
        
        # Initialize analyzers
        self.pdf_comparator = PDFComplianceComparator()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Test suite configuration
        self.test_modules = [
            'test_multitask_classification',
            'test_safe_feature_engineering', 
            'test_integration_compliance',
            'test_performance_benchmarks'
        ]
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all PDF compliance tests"""
        print("=" * 80)
        print("PDF COMPLIANCE TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        total_start_time = time.time()
        
        # Run each test module
        for module_name in self.test_modules:
            print(f"Running {module_name}...")
            module_results = self._run_test_module(module_name)
            self.results['detailed_results'][module_name] = module_results
            
            print(f"  ✓ {module_name} completed")
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # Generate summary
        self._generate_test_summary()
        self._generate_compliance_analysis()
        self._generate_recommendations()
        
        # Add metadata
        self.results['test_summary']['total_duration_seconds'] = total_duration
        self.results['test_summary']['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\nAll tests completed in {total_duration:.2f} seconds")
        
        return self.results
    
    def _run_test_module(self, module_name: str) -> Dict[str, Any]:
        """Run individual test module"""
        module_results = {
            'module_name': module_name,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'failures': [],
            'errors': [],
            'skipped': [],
            'duration': 0.0
        }
        
        try:
            # Create test suite for module
            loader = unittest.TestLoader()
            
            # Import the module
            if module_name == 'test_multitask_classification':
                from test_pdf_compliance import test_multitask_classification as test_module
            elif module_name == 'test_safe_feature_engineering':
                from test_pdf_compliance import test_safe_feature_engineering as test_module
            elif module_name == 'test_integration_compliance':
                from test_pdf_compliance import test_integration_compliance as test_module
            elif module_name == 'test_performance_benchmarks':
                from test_pdf_compliance import test_performance_benchmarks as test_module
            else:
                raise ImportError(f"Unknown test module: {module_name}")
            
            suite = loader.loadTestsFromModule(test_module)
            
            # Run tests with custom result handler
            stream = StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=0)
            
            start_time = time.time()
            result = runner.run(suite)
            end_time = time.time()
            
            # Parse results
            module_results['tests_run'] = result.testsRun
            module_results['tests_passed'] = result.testsRun - len(result.failures) - len(result.errors)
            module_results['tests_failed'] = len(result.failures)
            module_results['tests_skipped'] = len(result.skipped) if hasattr(result, 'skipped') else 0
            module_results['duration'] = end_time - start_time
            
            # Extract failure details
            for test, traceback in result.failures:
                module_results['failures'].append({
                    'test': str(test),
                    'error': traceback.split('\n')[-2] if traceback else 'Unknown'
                })
            
            for test, traceback in result.errors:
                module_results['errors'].append({
                    'test': str(test),
                    'error': traceback.split('\n')[-2] if traceback else 'Unknown'
                })
            
            if hasattr(result, 'skipped'):
                for test, reason in result.skipped:
                    module_results['skipped'].append({
                        'test': str(test),
                        'reason': str(reason)
                    })
            
        except Exception as e:
            module_results['errors'].append({
                'test': 'module_import',
                'error': str(e)
            })
            module_results['duration'] = 0.0
        
        return module_results
    
    def _generate_test_summary(self):
        """Generate overall test summary"""
        summary = {
            'total_modules': len(self.test_modules),
            'total_tests_run': 0,
            'total_tests_passed': 0,
            'total_tests_failed': 0,
            'total_tests_skipped': 0,
            'overall_success_rate': 0.0,
            'modules_with_failures': [],
            'modules_with_errors': []
        }
        
        # Aggregate results
        for module_name, module_results in self.results['detailed_results'].items():
            summary['total_tests_run'] += module_results['tests_run']
            summary['total_tests_passed'] += module_results['tests_passed']
            summary['total_tests_failed'] += module_results['tests_failed']
            summary['total_tests_skipped'] += module_results['tests_skipped']
            
            if module_results['tests_failed'] > 0:
                summary['modules_with_failures'].append(module_name)
            
            if module_results['errors']:
                summary['modules_with_errors'].append(module_name)
        
        # Calculate success rate
        if summary['total_tests_run'] > 0:
            summary['overall_success_rate'] = (
                summary['total_tests_passed'] / summary['total_tests_run'] * 100
            )
        
        self.results['test_summary'] = summary
    
    def _generate_compliance_analysis(self):
        """Generate PDF compliance analysis"""
        compliance_analysis = {
            'multitask_classification_compliance': {},
            'safe_feature_engineering_compliance': {},
            'integration_compliance': {},
            'performance_compliance': {},
            'overall_compliance_score': 0.0
        }
        
        try:
            # Analyze multi-task classification compliance
            if 'test_multitask_classification' in self.results['detailed_results']:
                mt_results = self.results['detailed_results']['test_multitask_classification']
                
                # Check key tests passed
                key_tests = [
                    'test_six_tasks_implemented',
                    'test_class_name_compliance',
                    'test_bert_base_encoder',
                    'test_crf_layer_implementation'
                ]
                
                passed_key_tests = 0
                for failure in mt_results['failures']:
                    if any(key_test in failure['test'] for key_test in key_tests):
                        passed_key_tests -= 1
                
                passed_key_tests = max(0, len(key_tests) + passed_key_tests - mt_results['tests_failed'])
                
                compliance_analysis['multitask_classification_compliance'] = {
                    'key_tests_passed': passed_key_tests,
                    'total_key_tests': len(key_tests),
                    'compliance_percentage': (passed_key_tests / len(key_tests)) * 100,
                    'status': 'PASS' if passed_key_tests == len(key_tests) else 'FAIL'
                }
            
            # Analyze SAFE feature engineering compliance
            if 'test_safe_feature_engineering' in self.results['detailed_results']:
                safe_results = self.results['detailed_results']['test_safe_feature_engineering']
                
                key_tests = [
                    'test_feature_scaling_capability',
                    'test_smart_interactions',
                    'test_mathematical_transformations',
                    'test_feature_importance_scoring'
                ]
                
                passed_key_tests = len(key_tests) - safe_results['tests_failed']
                
                compliance_analysis['safe_feature_engineering_compliance'] = {
                    'key_tests_passed': passed_key_tests,
                    'total_key_tests': len(key_tests),
                    'compliance_percentage': (passed_key_tests / len(key_tests)) * 100,
                    'status': 'PASS' if safe_results['tests_failed'] == 0 else 'FAIL'
                }
            
            # Analyze integration compliance
            if 'test_integration_compliance' in self.results['detailed_results']:
                int_results = self.results['detailed_results']['test_integration_compliance']
                
                key_tests = [
                    'test_text_feature_extraction',
                    'test_feature_fusion_logic',
                    'test_ensemble_feature_integration'
                ]
                
                passed_key_tests = len(key_tests) - int_results['tests_failed']
                
                compliance_analysis['integration_compliance'] = {
                    'key_tests_passed': passed_key_tests,
                    'total_key_tests': len(key_tests),
                    'compliance_percentage': (passed_key_tests / len(key_tests)) * 100,
                    'status': 'PASS' if int_results['tests_failed'] == 0 else 'FAIL'
                }
            
            # Analyze performance compliance
            if 'test_performance_benchmarks' in self.results['detailed_results']:
                perf_results = self.results['detailed_results']['test_performance_benchmarks']
                
                key_tests = [
                    'test_baseline_performance_targets',
                    'test_ensemble_performance_targets',
                    'test_overall_improvement_validation'
                ]
                
                passed_key_tests = len(key_tests) - perf_results['tests_failed']
                
                compliance_analysis['performance_compliance'] = {
                    'key_tests_passed': passed_key_tests,
                    'total_key_tests': len(key_tests),
                    'compliance_percentage': (passed_key_tests / len(key_tests)) * 100,
                    'status': 'PASS' if perf_results['tests_failed'] == 0 else 'FAIL'
                }
            
            # Calculate overall compliance score
            compliance_scores = []
            for component in [
                'multitask_classification_compliance',
                'safe_feature_engineering_compliance',
                'integration_compliance',
                'performance_compliance'
            ]:
                if compliance_analysis[component]:
                    compliance_scores.append(
                        compliance_analysis[component]['compliance_percentage']
                    )
            
            if compliance_scores:
                compliance_analysis['overall_compliance_score'] = sum(compliance_scores) / len(compliance_scores)
            
        except Exception as e:
            compliance_analysis['error'] = str(e)
        
        self.results['compliance_analysis'] = compliance_analysis
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Analyze test failures
        for module_name, module_results in self.results['detailed_results'].items():
            if module_results['tests_failed'] > 0:
                recommendations.append({
                    'category': 'test_failures',
                    'priority': 'high',
                    'module': module_name,
                    'message': f"Fix {module_results['tests_failed']} failing tests in {module_name}",
                    'details': module_results['failures'][:3]  # First 3 failures
                })
            
            if module_results['errors']:
                recommendations.append({
                    'category': 'test_errors',
                    'priority': 'critical',
                    'module': module_name,
                    'message': f"Resolve {len(module_results['errors'])} test errors in {module_name}",
                    'details': module_results['errors'][:3]  # First 3 errors
                })
        
        # Compliance-based recommendations
        compliance = self.results.get('compliance_analysis', {})
        
        if compliance.get('overall_compliance_score', 0) < 80:
            recommendations.append({
                'category': 'compliance',
                'priority': 'high',
                'message': f"Overall compliance score ({compliance.get('overall_compliance_score', 0):.1f}%) is below target",
                'details': 'Review implementation against PDF specifications'
            })
        
        # Component-specific recommendations
        mt_compliance = compliance.get('multitask_classification_compliance', {})
        if mt_compliance.get('compliance_percentage', 0) < 90:
            recommendations.append({
                'category': 'multitask_classification',
                'priority': 'high',
                'message': "Multi-task classification needs improvement",
                'details': "Focus on task specification compliance and BERT+CRF implementation"
            })
        
        safe_compliance = compliance.get('safe_feature_engineering_compliance', {})
        if safe_compliance.get('compliance_percentage', 0) < 90:
            recommendations.append({
                'category': 'safe_feature_engineering',
                'priority': 'high',
                'message': "SAFE feature engineering needs enhancement",
                'details': "Improve feature scaling, interactions, and mathematical transformations"
            })
        
        self.results['recommendations'] = recommendations
    
    def print_summary(self):
        """Print test summary to console"""
        print("\n" + "=" * 80)
        print("PDF COMPLIANCE TEST SUMMARY")
        print("=" * 80)
        
        summary = self.results['test_summary']
        
        print(f"Total Modules: {summary['total_modules']}")
        print(f"Total Tests Run: {summary['total_tests_run']}")
        print(f"Tests Passed: {summary['total_tests_passed']}")
        print(f"Tests Failed: {summary['total_tests_failed']}")
        print(f"Tests Skipped: {summary['total_tests_skipped']}")
        print(f"Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"Duration: {summary['total_duration_seconds']:.2f} seconds")
        
        if summary['modules_with_failures']:
            print(f"\nModules with Failures: {', '.join(summary['modules_with_failures'])}")
        
        if summary['modules_with_errors']:
            print(f"Modules with Errors: {', '.join(summary['modules_with_errors'])}")
        
        # Compliance analysis
        print("\n" + "-" * 50)
        print("COMPLIANCE ANALYSIS")
        print("-" * 50)
        
        compliance = self.results.get('compliance_analysis', {})
        overall_score = compliance.get('overall_compliance_score', 0)
        print(f"Overall Compliance Score: {overall_score:.1f}%")
        
        for component, comp_data in compliance.items():
            if isinstance(comp_data, dict) and 'status' in comp_data:
                print(f"  {component}: {comp_data['status']} ({comp_data['compliance_percentage']:.1f}%)")
        
        # Recommendations
        print("\n" + "-" * 50)
        print("KEY RECOMMENDATIONS")
        print("-" * 50)
        
        for rec in self.results['recommendations'][:5]:  # Top 5 recommendations
            priority_symbol = "🔴" if rec['priority'] == 'critical' else "🟡" if rec['priority'] == 'high' else "🟢"
            print(f"{priority_symbol} {rec['message']}")
        
        print("\n" + "=" * 80)
    
    def save_report(self, filepath: str = None):
        """Save comprehensive report to file"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"tests/pdf_compliance_report_{timestamp}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed report saved to: {filepath}")
        except Exception as e:
            print(f"Failed to save report: {e}")

def main():
    """Main function to run PDF compliance tests"""
    runner = PDFComplianceTestRunner()
    
    try:
        # Run all tests
        results = runner.run_all_tests()
        
        # Print summary
        runner.print_summary()
        
        # Save detailed report
        runner.save_report()
        
        # Exit with appropriate code
        summary = results['test_summary']
        if summary['total_tests_failed'] > 0 or len(summary['modules_with_errors']) > 0:
            sys.exit(1)  # Fail exit code
        else:
            sys.exit(0)  # Success exit code
            
    except Exception as e:
        print(f"Test runner failed: {e}")
        sys.exit(2)

if __name__ == '__main__':
    main()
