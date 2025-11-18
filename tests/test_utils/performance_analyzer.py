"""
Performance Analyzer
Analyze system performance against PDF targets
"""

import json
import time
import psutil
import numpy as np
from typing import Dict, List, Any, Tuple
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

try:
    from aiml_multi_task_heads import AIMLMultiTaskHeads, get_aiml_multi_task_heads
    from enhanced_bert_classifier import EnhancedBERTClassifier, get_enhanced_bert_classifier
    from enhanced_safe_features import EnhancedSAFE, get_enhanced_safe_features
    from enhanced_recommender_advanced import EnhancedRecommenderAdvanced
except ImportError as e:
    print(f"Import error: {e}")

class PerformanceAnalyzer:
    """Analyze system performance against PDF targets"""
    
    def __init__(self):
        """Initialize analyzer with PDF performance targets"""
        # PDF Table 2: Baseline Performance
        self.pdf_baseline = {
            'accuracy': 0.8364,
            'precision': 0.7095,
            'recall': 0.4441,
            'f1_score': 0.5462,
            'auc': 0.8325,
            'feature_count': 216
        }
        
        # PDF Table 4: Text Factors Performance
        self.pdf_text_factors = {
            'accuracy': 0.8481,
            'precision': 0.7473,
            'recall': 0.4755,
            'f1_score': 0.5812,
            'auc': 0.8410,
            'improvements': {
                'accuracy': 1.40,
                'auc': 1.02,
                'f1_score': 6.41
            },
            'text_features': 45
        }
        
        # PDF Table 10: Ensemble Performance
        self.pdf_ensemble = {
            'accuracy': 0.8713,
            'precision': 0.7143,
            'recall': 0.6107,
            'f1_score': 0.6584,
            'auc': 0.9344,
            'improvements': {
                'accuracy': 4.17,
                'precision': 0.68,
                'auc': 12.24,
                'f1_score': 20.54
            },
            'safe_features': 1155
        }
        
        # Memory constraints (Qdrant Free Tier)
        self.memory_constraints = {
            'max_ram_gb': 1.0,
            'max_disk_gb': 4.0
        }
    
    def calculate_improvement_percentages(self, baseline: Dict[str, float], 
                                    enhanced: Dict[str, float]) -> Dict[str, float]:
        """Calculate % improvements matching PDF format"""
        improvements = {}
        
        for metric in baseline.keys():
            if metric in enhanced and baseline[metric] > 0:
                improvement = ((enhanced[metric] - baseline[metric]) / baseline[metric]) * 100
                improvements[metric] = round(improvement, 2)
            else:
                improvements[metric] = 0.0
        
        return improvements
    
    def analyze_memory_efficiency(self, component_name: str) -> Dict[str, Any]:
        """Analyze memory usage vs constraints"""
        memory_info = {
            'component': component_name,
            'timestamp': time.time(),
            'system_memory': {},
            'process_memory': {},
            'compliance': {}
        }
        
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            memory_info['system_memory'] = {
                'total_gb': system_memory.total / (1024**3),
                'available_gb': system_memory.available / (1024**3),
                'used_gb': system_memory.used / (1024**3),
                'percentage': system_memory.percent
            }
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()
            memory_info['process_memory'] = {
                'rss_mb': process_memory.rss / (1024**2),  # Resident Set Size
                'vms_mb': process_memory.vms / (1024**2),  # Virtual Memory Size
                'percent': process.memory_percent()
            }
            
            # Check compliance with constraints
            process_gb = memory_info['process_memory']['rss_mb'] / 1024
            
            memory_info['compliance'] = {
                'ram_compliant': process_gb <= self.memory_constraints['max_ram_gb'],
                'ram_usage_gb': process_gb,
                'ram_limit_gb': self.memory_constraints['max_ram_gb'],
                'utilization_percentage': (process_gb / self.memory_constraints['max_ram_gb']) * 100
            }
            
        except Exception as e:
            memory_info['error'] = str(e)
        
        return memory_info
    
    def analyze_performance_scaling(self, test_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance scaling with claim complexity"""
        scaling_analysis = {
            'test_count': len(test_claims),
            'scaling_metrics': {},
            'performance_trends': {},
            'recommendations': []
        }
        
        try:
            # Categorize claims by complexity
            simple_claims = [c for c in test_claims if len(c.get('description', '').split()) < 10]
            moderate_claims = [c for c in test_claims if 10 <= len(c.get('description', '').split()) < 20]
            complex_claims = [c for c in test_claims if len(c.get('description', '').split()) >= 20]
            
            complexity_categories = {
                'simple': simple_claims,
                'moderate': moderate_claims,
                'complex': complex_claims
            }
            
            # Measure performance for each category
            for category, claims in complexity_categories.items():
                if not claims:
                    continue
                
                # Measure processing time
                start_time = time.time()
                
                for claim in claims:
                    # Simulate processing (would be actual implementation)
                    time.sleep(0.01)  # Placeholder for actual processing
                
                end_time = time.time()
                total_time = end_time - start_time
                
                if claims:
                    avg_time = total_time / len(claims)
                    scaling_analysis['scaling_metrics'][f'{category}_processing_time_ms'] = avg_time * 1000
                    scaling_analysis['scaling_metrics'][f'{category}_claim_count'] = len(claims)
            
            # Analyze scaling trends
            simple_time = scaling_analysis['scaling_metrics'].get('simple_processing_time_ms', 0)
            complex_time = scaling_analysis['scaling_metrics'].get('complex_processing_time_ms', 0)
            
            if simple_time > 0:
                scaling_ratio = complex_time / simple_time
                scaling_analysis['performance_trends']['complex_to_simple_ratio'] = scaling_ratio
                
                # Generate recommendations
                if scaling_ratio > 5.0:
                    scaling_analysis['recommendations'].append(
                        "Complex claims take significantly longer - consider optimization"
                    )
                elif scaling_ratio > 3.0:
                    scaling_analysis['recommendations'].append(
                        "Monitor performance with complex claims"
                    )
            
        except Exception as e:
            scaling_analysis['error'] = str(e)
        
        return scaling_analysis
    
    def benchmark_multitask_performance(self, multi_task_heads, 
                                     test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark multi-task classification performance"""
        benchmark_results = {
            'task_performance': {},
            'overall_performance': {},
            'f1_score_analysis': {},
            'compliance_with_pdf': {}
        }
        
        try:
            # Test each task
            for task_name in self.pdf_ensemble.keys():
                if task_name in ['improvements', 'safe_features']:
                    continue
                
                task_results = {
                    'predictions': [],
                    'processing_times': [],
                    'confidences': []
                }
                
                # Simulate task-specific testing
                for sample in test_samples:
                    start_time = time.time()
                    
                    # Placeholder for actual multi-task prediction
                    # prediction = multi_task_heads.predict_single_task(sample, task_name)
                    prediction = {'prediction': 'test', 'confidence': 0.85}
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    task_results['predictions'].append(prediction)
                    task_results['processing_times'].append(processing_time)
                    task_results['confidences'].append(prediction['confidence'])
                
                # Calculate task metrics
                if task_results['processing_times']:
                    avg_time = np.mean(task_results['processing_times'])
                    avg_confidence = np.mean(task_results['confidences'])
                    
                    benchmark_results['task_performance'][task_name] = {
                        'avg_processing_time_ms': avg_time * 1000,
                        'avg_confidence': avg_confidence,
                        'sample_count': len(task_results['predictions'])
                    }
            
            # Overall performance metrics
            all_times = []
            all_confidences = []
            
            for task_perf in benchmark_results['task_performance'].values():
                all_times.extend([task_perf['avg_processing_time_ms']])
                all_confidences.extend([task_perf['avg_confidence']])
            
            if all_times:
                benchmark_results['overall_performance'] = {
                    'avg_processing_time_ms': np.mean(all_times),
                    'max_processing_time_ms': np.max(all_times),
                    'min_processing_time_ms': np.min(all_times),
                    'avg_confidence': np.mean(all_confidences),
                    'total_tasks_tested': len(benchmark_results['task_performance'])
                }
            
        except Exception as e:
            benchmark_results['error'] = str(e)
        
        return benchmark_results
    
    def benchmark_safe_features(self, safe_features, 
                             test_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark SAFE feature generation performance"""
        benchmark_results = {
            'feature_generation_metrics': {},
            'feature_quality_metrics': {},
            'scaling_analysis': {},
            'pdf_compliance': {}
        }
        
        try:
            feature_counts = []
            generation_times = []
            feature_diversity_scores = []
            
            for claim in test_claims:
                start_time = time.time()
                
                # Generate features
                # features = safe_features.generate_comprehensive_features(claim)
                # Placeholder for actual feature generation
                features = {'test_feature': 1.0, 'amount': claim.get('amount', 0)}
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                feature_counts.append(len(features))
                generation_times.append(generation_time)
                
                # Calculate feature diversity (simplified)
                unique_categories = len(set(f.split('_')[0] for f in features.keys()))
                diversity_score = unique_categories / len(features) if features else 0
                feature_diversity_scores.append(diversity_score)
            
            # Calculate metrics
            if feature_counts:
                benchmark_results['feature_generation_metrics'] = {
                    'avg_feature_count': np.mean(feature_counts),
                    'max_feature_count': np.max(feature_counts),
                    'min_feature_count': np.min(feature_counts),
                    'avg_generation_time_ms': np.mean(generation_times) * 1000,
                    'max_generation_time_ms': np.max(generation_times) * 1000,
                    'total_claims_processed': len(test_claims)
                }
                
                benchmark_results['feature_quality_metrics'] = {
                    'avg_diversity_score': np.mean(feature_diversity_scores),
                    'min_diversity_score': np.min(feature_diversity_scores),
                    'max_diversity_score': np.max(feature_diversity_scores)
                }
            
            # PDF compliance check
            avg_features = np.mean(feature_counts) if feature_counts else 0
            pdf_target = self.pdf_ensemble['safe_features']
            
            benchmark_results['pdf_compliance'] = {
                'target_features': pdf_target,
                'actual_avg_features': avg_features,
                'compliance_percentage': (avg_features / pdf_target) * 100 if pdf_target > 0 else 0,
                'meets_target': avg_features >= pdf_target * 0.8  # 80% tolerance
            }
            
        except Exception as e:
            benchmark_results['error'] = str(e)
        
        return benchmark_results
    
    def generate_performance_dashboard(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance dashboard"""
        dashboard = {
            'summary': {
                'overall_performance_score': 0.0,
                'pdf_compliance_percentage': 0.0,
                'critical_issues': [],
                'performance_grade': 'Unknown'
            },
            'performance_metrics': {},
            'compliance_analysis': {},
            'trend_analysis': {},
            'recommendations': []
        }
        
        try:
            # Calculate overall performance score
            performance_scores = []
            
            # Multi-task performance
            if 'multitask_benchmark' in all_results:
                mt_perf = all_results['multitask_benchmark']
                if 'overall_performance' in mt_perf:
                    overall_perf = mt_perf['overall_performance']
                    # Score based on processing speed and confidence
                    time_score = max(0, 100 - overall_perf.get('avg_processing_time_ms', 100) / 10)
                    confidence_score = overall_perf.get('avg_confidence', 0) * 100
                    mt_score = (time_score + confidence_score) / 2
                    performance_scores.append(mt_score)
            
            # SAFE feature performance
            if 'safe_features_benchmark' in all_results:
                safe_perf = all_results['safe_features_benchmark']
                if 'pdf_compliance' in safe_perf:
                    compliance_pct = safe_perf['pdf_compliance'].get('compliance_percentage', 0)
                    performance_scores.append(compliance_pct)
            
            # Memory efficiency
            if 'memory_analysis' in all_results:
                mem_analysis = all_results['memory_analysis']
                if 'compliance' in mem_analysis:
                    utilization = mem_analysis['compliance'].get('utilization_percentage', 100)
                    memory_score = max(0, 100 - utilization)
                    performance_scores.append(memory_score)
            
            # Calculate overall score
            if performance_scores:
                dashboard['summary']['overall_performance_score'] = np.mean(performance_scores)
            
            # Determine performance grade
            score = dashboard['summary']['overall_performance_score']
            if score >= 90:
                dashboard['summary']['performance_grade'] = 'A'
            elif score >= 80:
                dashboard['summary']['performance_grade'] = 'B'
            elif score >= 70:
                dashboard['summary']['performance_grade'] = 'C'
            elif score >= 60:
                dashboard['summary']['performance_grade'] = 'D'
            else:
                dashboard['summary']['performance_grade'] = 'F'
            
            # Populate dashboard sections
            dashboard['performance_metrics'] = {
                'multitask_performance': all_results.get('multitask_benchmark', {}),
                'safe_features_performance': all_results.get('safe_features_benchmark', {}),
                'memory_efficiency': all_results.get('memory_analysis', {}),
                'scaling_analysis': all_results.get('scaling_analysis', {})
            }
            
            # Generate recommendations based on performance
            score = dashboard['summary']['overall_performance_score']
            if score < 70:
                dashboard['recommendations'].append("Overall performance needs significant improvement")
            if score < 80:
                dashboard['recommendations'].append("Consider optimizing processing speed")
            if score < 90:
                dashboard['recommendations'].append("Review memory usage patterns")
            
        except Exception as e:
            dashboard['error'] = str(e)
        
        return dashboard
    
    def save_performance_report(self, report: Dict[str, Any], filepath: str):
        """Save performance report to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Performance report saved to: {filepath}")
        except Exception as e:
            print(f"Failed to save performance report: {e}")

def main():
    """Main function for running performance analysis"""
    analyzer = PerformanceAnalyzer()
    
    # Example usage
    test_claims = [
        {'claim_id': 'TEST_001', 'amount': 5000.0, 'description': 'Simple claim'},
        {'claim_id': 'TEST_002', 'amount': 10000.0, 'description': 'More complex claim with detailed description of the accident involving multiple factors and circumstances'},
    ]
    
    # Run analyses
    memory_analysis = analyzer.analyze_memory_efficiency('test_component')
    scaling_analysis = analyzer.analyze_performance_scaling(test_claims)
    
    # Generate dashboard
    all_results = {
        'memory_analysis': memory_analysis,
        'scaling_analysis': scaling_analysis
    }
    
    dashboard = analyzer.generate_performance_dashboard(all_results)
    
    print(json.dumps(dashboard, indent=2))

if __name__ == '__main__':
    main()
