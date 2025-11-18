"""
PDF Compliance Comparator
Compare implementation against AIML paper specifications
"""

import json
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

class PDFComplianceComparator:
    """Compare implementation against PDF specifications"""
    
    def __init__(self):
        """Initialize comparator with PDF specifications"""
        # PDF Table 1: Task Specifications
        self.pdf_task_specs = {
            'driving_status': {
                'num_classes': 5,
                'class_names': ['driving', 'parked', 'stopped', 'passenger', 'unknown'],
                'target_f1': 0.93
            },
            'accident_type': {
                'num_classes': 12,
                'class_names': [
                    'collision', 'rollover', 'side_impact', 'rear_end', 'head_on',
                    'single_vehicle', 'multi_vehicle', 'pedestrian', 'animal', 'object',
                    'parking_lot', 'other'
                ],
                'target_f1': 0.84
            },
            'road_type': {
                'num_classes': 11,
                'class_names': [
                    'highway', 'urban', 'rural', 'parking', 'intersection',
                    'residential', 'commercial', 'industrial', 'bridge', 'tunnel', 'other'
                ],
                'target_f1': 0.79
            },
            'cause_accident': {
                'num_classes': 11,
                'class_names': [
                    'negligence', 'weather', 'mechanical', 'medical', 'intentional',
                    'distraction', 'fatigue', 'impaired', 'speed', 'road_condition', 'other'
                ],
                'target_f1': 0.85
            },
            'vehicle_count': {
                'num_classes': 4,
                'class_names': ['single', 'two', 'multiple', 'unknown'],
                'target_f1': 0.94
            },
            'parties_involved': {
                'num_classes': 5,
                'class_names': ['single', 'two', 'multiple', 'pedestrian', 'property_only'],
                'target_f1': 0.89
            }
        }
        
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
        
        # SAFE Feature Categories (from PDF Section 3.3)
        self.pdf_safe_categories = {
            'temporal': 15,
            'amount': 12,
            'frequency': 10,
            'geographic': 8,
            'policy': 8,
            'claimant': 6
        }
    
    def compare_task_specifications(self, implementation, pdf_specs=None) -> Dict[str, Any]:
        """Compare task specs against PDF Table 1"""
        if pdf_specs is None:
            pdf_specs = self.pdf_task_specs
            
        compliance_results = {
            'overall_compliance': True,
            'task_compliance': {},
            'missing_tasks': [],
            'incorrect_class_counts': [],
            'missing_classes': [],
            'f1_score_deviations': []
        }
        
        try:
            actual_tasks = implementation.task_specs
            
            # Check task count
            expected_tasks = set(pdf_specs.keys())
            actual_task_names = set(actual_tasks.keys())
            
            missing_tasks = expected_tasks - actual_task_names
            if missing_tasks:
                compliance_results['missing_tasks'] = list(missing_tasks)
                compliance_results['overall_compliance'] = False
            
            # Check each task
            for task_name, pdf_spec in pdf_specs.items():
                if task_name not in actual_tasks:
                    continue
                    
                actual_spec = actual_tasks[task_name]
                task_result = {
                    'task_name': task_name,
                    'compliant': True,
                    'issues': []
                }
                
                # Check class count
                if actual_spec.num_classes != pdf_spec['num_classes']:
                    task_result['compliant'] = False
                    task_result['issues'].append(
                        f"Class count: expected {pdf_spec['num_classes']}, got {actual_spec.num_classes}"
                    )
                    compliance_results['incorrect_class_counts'].append({
                        'task': task_name,
                        'expected': pdf_spec['num_classes'],
                        'actual': actual_spec.num_classes
                    })
                
                # Check class names
                missing_classes = set(pdf_spec['class_names']) - set(actual_spec.class_names)
                if missing_classes:
                    task_result['compliant'] = False
                    task_result['issues'].append(f"Missing classes: {list(missing_classes)}")
                    compliance_results['missing_classes'].append({
                        'task': task_name,
                        'missing': list(missing_classes)
                    })
                
                # Check F1 target
                if hasattr(actual_spec, 'target_f1'):
                    f1_diff = abs(actual_spec.target_f1 - pdf_spec['target_f1'])
                    if f1_diff > 0.05:  # Allow small tolerance
                        task_result['compliant'] = False
                        task_result['issues'].append(
                            f"F1 target: expected {pdf_spec['target_f1']}, got {actual_spec.target_f1}"
                        )
                        compliance_results['f1_score_deviations'].append({
                            'task': task_name,
                            'expected': pdf_spec['target_f1'],
                            'actual': actual_spec.target_f1,
                            'difference': f1_diff
                        })
                
                compliance_results['task_compliance'][task_name] = task_result
                
        except Exception as e:
            compliance_results['error'] = str(e)
            compliance_results['overall_compliance'] = False
        
        return compliance_results
    
    def compare_performance_metrics(self, results: Dict[str, float], 
                                 pdf_benchmarks: Dict[str, float],
                                 tolerance: float = 0.05) -> Dict[str, Any]:
        """Compare performance against PDF benchmarks"""
        comparison = {
            'overall_compliance': True,
            'metric_compliance': {},
            'deviations': [],
            'improvements': {}
        }
        
        for metric, pdf_value in pdf_benchmarks.items():
            if metric in results:
                actual_value = results[metric]
                deviation = abs(actual_value - pdf_value)
                
                metric_result = {
                    'metric': metric,
                    'pdf_value': pdf_value,
                    'actual_value': actual_value,
                    'deviation': deviation,
                    'compliant': deviation <= tolerance,
                    'percentage_difference': ((actual_value - pdf_value) / pdf_value) * 100
                }
                
                comparison['metric_compliance'][metric] = metric_result
                
                if not metric_result['compliant']:
                    comparison['overall_compliance'] = False
                    comparison['deviations'].append(metric_result)
                
                # Calculate improvement over baseline if applicable
                if metric == 'auc' and 'pdf_baseline' in locals():
                    baseline_auc = self.pdf_baseline['auc']
                    improvement = ((actual_value - baseline_auc) / baseline_auc) * 100
                    comparison['improvements']['auc'] = improvement
        
        return comparison
    
    def compare_feature_engineering(self, features: Dict[str, Any], 
                                 pdf_features: Dict[str, int]) -> Dict[str, Any]:
        """Compare feature engineering against PDF specifications"""
        comparison = {
            'overall_compliance': True,
            'feature_counts': {},
            'category_compliance': {},
            'missing_categories': []
        }
        
        # Count features by category
        feature_categories = {}
        for feature_name in features.keys():
            category_found = False
            for category in pdf_features.keys():
                if any(keyword in feature_name.lower() 
                      for keyword in [category.split('_')[0]]):
                    if category not in feature_categories:
                        feature_categories[category] = 0
                    feature_categories[category] += 1
                    category_found = True
                    break
            
            if not category_found:
                # Categorize as 'other'
                if 'other' not in feature_categories:
                    feature_categories['other'] = 0
                feature_categories['other'] += 1
        
        # Compare categories
        for category, expected_count in pdf_features.items():
            actual_count = feature_categories.get(category, 0)
            
            category_result = {
                'category': category,
                'expected_count': expected_count,
                'actual_count': actual_count,
                'compliant': actual_count >= expected_count * 0.8  # Allow 20% tolerance
            }
            
            comparison['category_compliance'][category] = category_result
            comparison['feature_counts'][category] = actual_count
            
            if not category_result['compliant']:
                comparison['overall_compliance'] = False
            elif actual_count == 0:
                comparison['missing_categories'].append(category)
        
        # Add total feature count
        total_features = len(features)
        comparison['total_features'] = total_features
        
        # Check SAFE feature scaling (PDF: 216 -> 1155 features)
        if total_features < 500:  # Should generate substantial features
            comparison['overall_compliance'] = False
        
        return comparison
    
    def generate_compliance_report(self, implementation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed compliance report"""
        report = {
            'summary': {
                'overall_compliance': True,
                'compliance_percentage': 0.0,
                'critical_issues': [],
                'recommendations': []
            },
            'task_specification_compliance': {},
            'performance_compliance': {},
            'feature_engineering_compliance': {},
            'architecture_compliance': {},
            'memory_constraint_compliance': {}
        }
        
        try:
            # Task specification compliance
            if 'multi_task_heads' in implementation_results:
                task_compliance = self.compare_task_specifications(
                    implementation_results['multi_task_heads']
                )
                report['task_specification_compliance'] = task_compliance
                
                if not task_compliance['overall_compliance']:
                    report['summary']['critical_issues'].append(
                        "Multi-task classification does not match PDF specifications"
                    )
                    report['summary']['recommendations'].append(
                        "Update task specifications to match PDF Table 1"
                    )
            
            # Performance compliance
            if 'performance_metrics' in implementation_results:
                perf_compliance = self.compare_performance_metrics(
                    implementation_results['performance_metrics'],
                    self.pdf_ensemble
                )
                report['performance_compliance'] = perf_compliance
                
                if not perf_compliance['overall_compliance']:
                    report['summary']['critical_issues'].append(
                        "Performance metrics do not meet PDF targets"
                    )
                    report['summary']['recommendations'].append(
                        "Optimize model to achieve PDF Table 10 performance"
                    )
            
            # Feature engineering compliance
            if 'engineered_features' in implementation_results:
                feature_compliance = self.compare_feature_engineering(
                    implementation_results['engineered_features'],
                    self.pdf_safe_categories
                )
                report['feature_engineering_compliance'] = feature_compliance
                
                if not feature_compliance['overall_compliance']:
                    report['summary']['critical_issues'].append(
                        "Feature engineering does not meet PDF SAFE specifications"
                    )
                    report['summary']['recommendations'].append(
                        "Enhance SAFE feature generation to match PDF Section 3.3"
                    )
            
            # Architecture compliance
            if 'bert_classifier' in implementation_results:
                arch_compliance = self._check_architecture_compliance(
                    implementation_results['bert_classifier']
                )
                report['architecture_compliance'] = arch_compliance
                
                if not arch_compliance['overall_compliance']:
                    report['summary']['critical_issues'].append(
                        "Architecture does not match PDF requirements"
                    )
            
            # Memory constraint compliance
            if 'memory_usage' in implementation_results:
                memory_compliance = self._check_memory_constraints(
                    implementation_results['memory_usage']
                )
                report['memory_constraint_compliance'] = memory_compliance
                
                if not memory_compliance['overall_compliance']:
                    report['summary']['critical_issues'].append(
                        "Memory usage exceeds constraints"
                    )
            
            # Calculate overall compliance percentage
            compliance_areas = [
                report['task_specification_compliance'].get('overall_compliance', True),
                report['performance_compliance'].get('overall_compliance', True),
                report['feature_engineering_compliance'].get('overall_compliance', True),
                report['architecture_compliance'].get('overall_compliance', True),
                report['memory_constraint_compliance'].get('overall_compliance', True)
            ]
            
            compliant_areas = sum(compliance_areas)
            total_areas = len(compliance_areas)
            report['summary']['compliance_percentage'] = (compliant_areas / total_areas) * 100
            report['summary']['overall_compliance'] = all(compliance_areas)
            
        except Exception as e:
            report['error'] = str(e)
            report['summary']['critical_issues'].append(f"Report generation failed: {e}")
        
        return report
    
    def _check_architecture_compliance(self, bert_classifier) -> Dict[str, Any]:
        """Check BERT+CRF architecture compliance"""
        compliance = {
            'overall_compliance': True,
            'checks': {},
            'issues': []
        }
        
        try:
            # Check BERT base model
            has_bert = hasattr(bert_classifier, 'base_model') and bert_classifier.base_model is not None
            compliance['checks']['bert_base_model'] = has_bert
            if not has_bert:
                compliance['overall_compliance'] = False
                compliance['issues'].append("Missing BERT base model")
            
            # Check domain adaptation
            has_domain_adapter = hasattr(bert_classifier, 'domain_adapter') and bert_classifier.domain_adapter is not None
            compliance['checks']['domain_adaptation'] = has_domain_adapter
            if not has_domain_adapter:
                compliance['overall_compliance'] = False
                compliance['issues'].append("Missing domain adaptation layer")
            
            # Check CRF layer
            has_crf = hasattr(bert_classifier, 'crf_layer') and bert_classifier.crf_layer is not None
            compliance['checks']['crf_layer'] = has_crf
            if not has_crf:
                compliance['overall_compliance'] = False
                compliance['issues'].append("Missing CRF layer")
            
            # Check multi-task heads
            has_multitask = hasattr(bert_classifier, 'classification_heads') and bert_classifier.classification_heads is not None
            compliance['checks']['multitask_heads'] = has_multitask
            if not has_multitask:
                compliance['overall_compliance'] = False
                compliance['issues'].append("Missing multi-task classification heads")
            
            # Check insurance vocabulary
            has_vocab = hasattr(bert_classifier, 'insurance_vocab') and bert_classifier.insurance_vocab is not None
            compliance['checks']['insurance_vocabulary'] = has_vocab
            if not has_vocab:
                compliance['overall_compliance'] = False
                compliance['issues'].append("Missing insurance domain vocabulary")
            
        except Exception as e:
            compliance['error'] = str(e)
            compliance['overall_compliance'] = False
            compliance['issues'].append(f"Architecture check failed: {e}")
        
        return compliance
    
    def _check_memory_constraints(self, memory_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Check memory constraint compliance (Qdrant Free Tier: 1GB RAM, 4GB Disk)"""
        compliance = {
            'overall_compliance': True,
            'checks': {},
            'issues': [],
            'constraints': {
                'max_ram_gb': 1.0,
                'max_disk_gb': 4.0
            }
        }
        
        try:
            # Check RAM usage
            ram_usage_gb = memory_usage.get('ram_usage_gb', 0)
            ram_compliant = ram_usage_gb <= compliance['constraints']['max_ram_gb']
            compliance['checks']['ram_usage'] = {
                'used_gb': ram_usage_gb,
                'max_gb': compliance['constraints']['max_ram_gb'],
                'compliant': ram_compliant
            }
            if not ram_compliant:
                compliance['overall_compliance'] = False
                compliance['issues'].append(f"RAM usage {ram_usage_gb}GB exceeds limit {compliance['constraints']['max_ram_gb']}GB")
            
            # Check disk usage
            disk_usage_gb = memory_usage.get('disk_usage_gb', 0)
            disk_compliant = disk_usage_gb <= compliance['constraints']['max_disk_gb']
            compliance['checks']['disk_usage'] = {
                'used_gb': disk_usage_gb,
                'max_gb': compliance['constraints']['max_disk_gb'],
                'compliant': disk_compliant
            }
            if not disk_compliant:
                compliance['overall_compliance'] = False
                compliance['issues'].append(f"Disk usage {disk_usage_gb}GB exceeds limit {compliance['constraints']['max_disk_gb']}GB")
            
        except Exception as e:
            compliance['error'] = str(e)
            compliance['overall_compliance'] = False
            compliance['issues'].append(f"Memory check failed: {e}")
        
        return compliance
    
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save compliance report to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Compliance report saved to: {filepath}")
        except Exception as e:
            print(f"Failed to save report: {e}")

def main():
    """Main function for running PDF compliance comparison"""
    comparator = PDFComplianceComparator()
    
    # Example usage (would need actual implementation results)
    sample_results = {
        'multi_task_heads': None,  # Would be actual multi-task heads
        'performance_metrics': {
            'accuracy': 0.87,
            'precision': 0.72,
            'recall': 0.61,
            'f1_score': 0.66,
            'auc': 0.93
        },
        'engineered_features': {},  # Would be actual feature dictionary
        'memory_usage': {
            'ram_usage_gb': 0.8,
            'disk_usage_gb': 2.5
        }
    }
    
    report = comparator.generate_compliance_report(sample_results)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
