# Cline_SAFE_EnhancedTextProcessing: Gap Analysis & Implementation Guide

## Executive Summary

This document provides a comprehensive gap analysis between your current insurance fraud detection implementation and the AIML (Auto Insurance Multi-modal Learning) framework, focusing specifically on **Enhanced Text Processing** and **SAFE (Semi-Auto Feature Engineering)** components. All recommendations are optimized for Qdrant free tier constraints (1GB RAM, 4GB Disk, 0.5 vCPU).

---

## 1. Current Implementation Analysis

### 1.1 Text Processing (`multitext_classifier.py`)

**Current Approach:**
- Uses `sentence-transformers` (all-MiniLM-L6-v2) for semantic similarity
- 8 classification tasks with keyword-based fallback
- Basic fraud risk indicator extraction
- Memory-efficient but limited sophistication

**Strengths:**
- Lightweight (fits in 1GB RAM)
- Pre-computed embeddings for efficiency
- Handles multiple classification tasks
- Good fallback mechanisms

**Limitations:**
- No domain-specific fine-tuning
- Missing sequence labeling capabilities
- No CRF layer for structured prediction
- Limited contextual understanding

### 1.2 Feature Engineering (`safe_features.py`)

**Current Approach:**
- 33 basic features across 6 categories
- Temporal, amount, frequency, geographic, policy, claimant features
- Simple rule-based feature extraction
- Memory-efficient implementation

**Strengths:**
- Well-structured feature categories
- Covers key risk dimensions
- Resource-conscious design
- Good error handling

**Limitations:**
- Only 33 features vs. AIML's 1,155
- No automated feature interaction generation
- Missing feature selection algorithms
- Limited feature engineering sophistication

---

## 2. AIML Paper Comparison

### 2.1 AIML Text Processing (Superior)

**Architecture:**
- **BERT (Chinese version)** + **CRF** for multi-task learning
- 6 specific classification tasks:
  1. Driving status (5 classes)
  2. Accident type (12 classes)  
  3. Road type (11 classes)
  4. Cause of accident (11 classes)
  5. Vehicle count (4 classes)
  6. Parties involved (5 classes)
- **F1-scores**: 0.93, 0.84, 0.79, 0.85, 0.94
- Joint probability optimization for task correlation

**Performance Impact:**
- Added 45 new features via one-hot encoding
- Improved AUC from 0.8325 → 0.841 (1.02% improvement)
- Significant accuracy in domain-specific tasks

### 2.2 AIML SAFE Algorithm (Superior)

**Architecture:**
- **Feature Classification**: Categorizes original 216 features
- **Feature Derivation**: Creates automated interactions
- **1,155 total features** from 216 original variables
- XGBoost integration for optimal feature selection

**Feature Types Generated:**
- One-hot encoded categoricals
- Feature interactions (combinations)
- Mathematical operations (addition, subtraction)
- Temporal derivatives
- Statistical aggregations

**Performance Impact:**
- Baseline: AUC = 0.8325
- With SAFE + Text + Visual: AUC = 0.9344 (12.24% improvement)

---

## 3. Critical Gaps Identified

### 3.1 Text Processing Gaps

| Gap | Current | AIML | Impact | Priority |
|-----|---------|-------|---------|----------|
| **Domain-specific BERT** | Generic sentence-transformers | BERT fine-tuned on insurance data | Higher accuracy on domain tasks | HIGH |
| **CRF Layer** | None | Conditional Random Fields | Better sequence labeling | HIGH |
| **Multi-task Optimization** | Independent tasks | Joint probability optimization | Task correlation exploitation | MEDIUM |
| **Context Window** | Limited | Full context understanding | Better semantic capture | MEDIUM |
| **Language Support** | English only | Chinese optimized | Geographic applicability | LOW |

### 3.2 SAFE Feature Engineering Gaps

| Gap | Current | AIML | Impact | Priority |
|-----|---------|-------|---------|----------|
| **Feature Scale** | 33 features | 1,155 features | Model capacity | CRITICAL |
| **Auto Interaction** | None | Automated feature combinations | Non-linear relationships | HIGH |
| **Feature Selection** | Manual | Algorithm-driven selection | Optimal feature subset | HIGH |
| **Mathematical Operations** | Basic | Addition, subtraction, multiplication | Rich feature space | MEDIUM |
| **Statistical Features** | Limited | Aggregations, derivatives | Temporal patterns | MEDIUM |

---

## 4. Resource-Constrained Solutions

### 4.1 Memory Optimization Strategy

**Qdrant Free Tier Constraints:**
- **RAM**: 1GB total
- **Disk**: 4GB total
- **CPU**: 0.5 vCPU

**Memory Allocation Plan:**
```
Base System:       200MB
Qdrant Database:   300MB  
Text Models:       200MB
Feature Engine:    150MB
API Framework:     100MB
OS/Overhead:       50MB
Total:            1,000MB
```

### 4.2 Enhanced Text Processing Solutions

#### Solution 1: Lightweight Domain Adaptation
```python
# Memory-efficient domain adaptation
class LightweightBERTAdapter:
    def __init__(self):
        # Use distilled models for memory efficiency
        self.base_model = 'distilbert-base-uncased'  # ~250MB vs BERT ~400MB
        self.insurance_vocab = self._load_insurance_vocabulary()
        
    def _load_insurance_vocabulary(self):
        # Domain-specific vocabulary for better understanding
        return {
            'accident_types': ['collision', 'rollover', 'side_impact'],
            'damage_severity': ['minor', 'moderate', 'severe', 'total'],
            'insurance_terms': ['deductible', 'premium', 'claim', 'coverage']
        }
```

#### Solution 2: Incremental Feature Generation
```python
# Generate features in batches to stay within memory limits
class IncrementalSAFE:
    def __init__(self):
        self.batch_size = 50  # Process 50 features at a time
        self.feature_cache = {}
        
    def generate_features_incremental(self, claim_data):
        # Process features in memory-efficient batches
        temporal_features = self._extract_temporal_batch(claim_data)
        amount_features = self._extract_amount_batch(claim_data)
        # ... other batches
        
        return self._combine_features(temporal_features, amount_features)
```

### 4.3 Enhanced SAFE Implementation

#### Solution 1: Smart Feature Selection
```python
class MemoryEfficientSAFE:
    def __init__(self):
        self.feature_importance_threshold = 0.01
        self.max_features = 200  # Limit for memory efficiency
        
    def generate_smart_features(self, claim_data):
        # Generate only high-impact features
        base_features = self._extract_base_features(claim_data)
        
        # Top 100 feature interactions based on importance
        interactions = self._generate_top_interactions(base_features, limit=100)
        
        # Top 50 mathematical transformations
        transformations = self._generate_transformations(base_features, limit=50)
        
        return self._combine_and_select(base_features, interactions, transformations)
```

#### Solution 2: Lazy Feature Loading
```python
class LazyFeatureGenerator:
    def __init__(self):
        self.feature_generators = {
            'temporal': self._generate_temporal_features,
            'amount': self._generate_amount_features,
            'frequency': self._generate_frequency_features
        }
        
    def get_features_on_demand(self, claim_data, feature_types):
        # Only generate requested features to save memory
        features = []
        for ftype in feature_types:
            if ftype in self.feature_generators:
                features.extend(self.feature_generators[ftype](claim_data))
        return features
```

---

## 5. Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
- [ ] **Upgrade Text Model**: Replace with domain-adapted DistilBERT
- [ ] **Add Feature Interactions**: Implement top 50 interaction features
- [ ] **Memory Optimization**: Implement lazy loading and batching
- [ ] **Basic Validation**: Add performance metrics and AUC tracking

### Phase 2: Core Enhancement (2-3 weeks)  
- [ ] **CRF Implementation**: Add conditional random fields
- [ ] **Advanced SAFE**: Implement automated feature derivation
- [ ] **Feature Selection**: Add algorithm-driven feature selection
- [ ] **Performance Tuning**: Optimize for 1GB RAM constraint

### Phase 3: Production Optimization (1-2 weeks)
- [ ] **Incremental Processing**: Implement batch processing
- [ ] **Caching Strategy**: Add intelligent feature caching
- [ ] **Resource Monitoring**: Add memory/CPU usage tracking
- [ ] **Load Testing**: Validate under Qdrant constraints

---

## 6. Technical Implementation Details

### 6.1 Enhanced Text Processing

#### Memory-Efficient BERT Adapter
```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class MemoryEfficientTextClassifier:
    def __init__(self):
        # Use distilled model for memory efficiency
        model_name = 'distilbert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Insurance domain adaptation layer
        self.domain_adapter = self._create_domain_adapter()
        
        # Classification heads for each task
        self.classification_heads = self._create_classification_heads()
        
    def _create_domain_adapter(self):
        # Lightweight domain adaptation layer
        return torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128)
        )
        
    def _create_classification_heads(self):
        # Multi-task classification heads
        tasks = {
            'driving_status': 5,
            'accident_type': 12,
            'road_type': 11,
            'cause_accident': 11,
            'vehicle_count': 4,
            'parties_involved': 5
        }
        
        heads = {}
        for task, num_classes in tasks.items():
            heads[task] = torch.nn.Linear(128, num_classes)
        
        return heads
        
    def classify_claim(self, text, claim_data):
        # Memory-efficient classification
        with torch.no_grad():
            # Tokenize with max length for memory control
            inputs = self.tokenizer(
                text, 
                max_length=256,  # Limit for memory
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            # Get BERT embeddings
            outputs = self.model(**inputs)
            pooled_output = outputs.pooler_output
            
            # Apply domain adaptation
            domain_features = self.domain_adapter(pooled_output)
            
            # Multi-task classification
            results = {}
            for task, head in self.classification_heads.items():
                logits = head(domain_features)
                probabilities = torch.softmax(logits, dim=-1)
                results[task] = {
                    'prediction': torch.argmax(probabilities, dim=-1).item(),
                    'confidence': torch.max(probabilities).item(),
                    'probabilities': probabilities.squeeze().tolist()
                }
            
            return results
```

### 6.2 Enhanced SAFE Implementation

#### Scalable Feature Engineering
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import itertools

class ScalableSAFE:
    def __init__(self, max_features=200, memory_limit_mb=150):
        self.max_features = max_features
        self.memory_limit_mb = memory_limit_mb
        self.feature_selectors = {}
        self.scaler = StandardScaler()
        
    def generate_comprehensive_features(self, claim_data):
        """Generate AIML-style features within memory constraints"""
        
        # Phase 1: Base features (like current implementation)
        base_features = self._extract_base_features(claim_data)
        
        # Phase 2: Feature interactions (smart selection)
        interaction_features = self._generate_interactions(base_features)
        
        # Phase 3: Mathematical transformations
        transform_features = self._generate_transformations(base_features)
        
        # Phase 4: Temporal features
        temporal_features = self._generate_temporal_derivatives(claim_data)
        
        # Combine all features
        all_features = {
            **base_features,
            **interaction_features, 
            **transform_features,
            **temporal_features
        }
        
        # Feature selection to stay within limits
        selected_features = self._select_top_features(all_features)
        
        return selected_features
        
    def _extract_base_features(self, claim_data):
        """Extract base features (enhanced version of current)"""
        features = {}
        
        # Enhanced temporal features
        features.update(self._enhanced_temporal_features(claim_data))
        
        # Enhanced amount features  
        features.update(self._enhanced_amount_features(claim_data))
        
        # Enhanced frequency features
        features.update(self._enhanced_frequency_features(claim_data))
        
        # Geographic features with more granularity
        features.update(self._enhanced_geographic_features(claim_data))
        
        # Policy features with more depth
        features.update(self._enhanced_policy_features(claim_data))
        
        # Claimant behavioral features
        features.update(self._behavioral_features(claim_data))
        
        return features
        
    def _generate_interactions(self, base_features):
        """Generate feature interactions smartly"""
        interactions = {}
        feature_names = list(base_features.keys())
        
        # Limit interactions to stay within memory
        max_interactions = min(50, len(feature_names) * 5)
        
        # Smart interaction selection based on domain knowledge
        important_pairs = [
            ('amount', 'claim_type'),
            ('time_of_day', 'location_risk'),
            ('customer_history', 'amount'),
            ('policy_age', 'claim_frequency'),
            ('damage_severity', 'amount')
        ]
        
        interaction_count = 0
        for feat1, feat2 in important_pairs:
            if feat1 in base_features and feat2 in base_features:
                if interaction_count >= max_interactions:
                    break
                    
                # Create interaction feature
                interaction_name = f"{feat1}_x_{feat2}"
                interactions[interaction_name] = (
                    base_features[feat1] * base_features[feat2]
                )
                interaction_count += 1
                
        return interactions
        
    def _generate_transformations(self, base_features):
        """Generate mathematical transformations"""
        transformations = {}
        
        # Log transformations for skewed features
        for feat_name, feat_value in base_features.items():
            if feat_value > 0 and 'amount' in feat_name:
                transformations[f"log_{feat_name}"] = np.log1p(feat_value)
                
            # Square transformations for non-linear relationships
            if 'risk' in feat_name or 'severity' in feat_name:
                transformations[f"squared_{feat_name}"] = feat_value ** 2
                
            # Square root transformations
            if feat_value > 0 and ('count' in feat_name or 'frequency' in feat_name):
                transformations[f"sqrt_{feat_name}"] = np.sqrt(feat_value)
                
        return transformations
        
    def _enhanced_temporal_features(self, claim_data):
        """Enhanced temporal feature extraction"""
        features = {}
        
        # Current temporal features (keep existing)
        features.update(self._extract_temporal_features(claim_data))
        
        # Additional temporal features
        claim_date = claim_data.get('date_submitted', '')
        if claim_date:
            try:
                date_obj = pd.to_datetime(claim_date)
                
                # Seasonal features
                features['season'] = date_obj.month % 12 // 3 + 1
                features['is_holiday_season'] = 1 if date_obj.month in [11, 12, 1] else 0
                
                # Business vs weekend patterns
                features['is_business_hour'] = 1 if 9 <= date_obj.hour <= 17 else 0
                features['day_of_week'] = date_obj.dayofweek
                
                # Time-based risk scoring
                features['temporal_risk_score'] = self._calculate_temporal_risk(date_obj)
                
            except:
                pass
                
        return features
        
    def _enhanced_amount_features(self, claim_data):
        """Enhanced amount-based features"""
        features = {}
        amount = float(claim_data.get('amount', 0))
        
        if amount > 0:
            # Current amount features (keep existing)
            features.update(self._extract_amount_features(claim_data))
            
            # Additional amount features
            features['amount_log'] = np.log1p(amount)
            features['amount_sqrt'] = np.sqrt(amount)
            features['amount_squared'] = amount ** 2
            
            # Amount percentiles (based on claim type)
            claim_type = claim_data.get('claim_type', 'auto')
            percentile = self._get_amount_percentile(amount, claim_type)
            features[f'amount_percentile_{claim_type}'] = percentile
            
            # Amount-to-claim-type ratio
            avg_amount = self._get_average_amount_by_type(claim_type)
            if avg_amount > 0:
                features['amount_to_avg_ratio'] = amount / avg_amount
                
        return features
        
    def _calculate_temporal_risk(self, date_obj):
        """Calculate temporal risk score"""
        risk_score = 0.0
        
        # Time of day risk
        if date_obj.hour >= 22 or date_obj.hour <= 5:
            risk_score += 0.3
        elif date_obj.hour >= 18 or date_obj.hour <= 7:
            risk_score += 0.2
            
        # Day of week risk
        if date_obj.dayofweek >= 5:  # Weekend
            risk_score += 0.2
            
        # Seasonal risk
        if date_obj.month in [12, 1, 2]:  # Winter
            risk_score += 0.1
            
        return min(risk_score, 1.0)
        
    def _get_amount_percentile(self, amount, claim_type):
        """Get amount percentile for claim type (simplified)"""
        # In production, this would use historical data
        percentiles = {
            'auto': [1000, 2500, 3500, 5000, 8000],
            'home': [2000, 5000, 8000, 12000, 20000],
            'health': [5000, 15000, 25000, 40000, 75000]
        }
        
        amounts = percentiles.get(claim_type, percentiles['auto'])
        
        for i, threshold in enumerate(amounts):
            if amount <= threshold:
                return (i + 1) / len(amounts)
        
        return 1.0
        
    def _get_average_amount_by_type(self, claim_type):
        """Get average amount by claim type"""
        averages = {
            'auto': 3500,
            'home': 8000, 
            'health': 25000,
            'travel': 1500,
            'life': 50000
        }
        return averages.get(claim_type, 3500)
        
    def _select_top_features(self, all_features):
        """Select top features based on importance and memory constraints"""
        # Convert to numpy array for processing
        feature_names = list(all_features.keys())
        feature_values = list(all_features.values())
        
        # Calculate feature importance (simplified)
        importance_scores = []
        
        for i, (name, value) in enumerate(all_features.items()):
            score = 0.0
            
            # Domain knowledge-based scoring
            if 'amount' in name and 'ratio' in name:
                score += 0.8
            elif 'risk' in name:
                score += 0.7
            elif 'frequency' in name:
                score += 0.6
            elif 'temporal' in name or 'time' in name:
                score += 0.5
            elif 'interaction' in name:
                score += 0.4
            else:
                score += 0.2
                
            importance_scores.append(score)
        
        # Sort by importance and select top features
        feature_importance_pairs = list(zip(feature_names, feature_values, importance_scores))
        feature_importance_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Select top features within memory limit
        selected_features = {}
        memory_usage = 0
        
        for name, value, importance in feature_importance_pairs:
            # Estimate memory usage (simplified)
            feature_memory = len(str(value)) * 2  # Rough estimate
            
            if memory_usage + feature_memory <= self.memory_limit_mb * 1024 * 1024:
                selected_features[name] = value
                memory_usage += feature_memory
            else:
                break
                
        return selected_features
```

### 6.3 Memory Management Utilities

```python
import psutil
import gc

class MemoryManager:
    def __init__(self, limit_mb=100):
        self.limit_mb = limit_mb
        self.limit_bytes = limit_mb * 1024 * 1024
        
    def check_memory_usage(self):
        """Check current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'used_mb': memory_info.rss / (1024 * 1024),
            'available_mb': self.limit_mb - (memory_info.rss / (1024 * 1024)),
            'percentage': (memory_info.rss / self.limit_bytes) * 100
        }
        
    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        gc.collect()
        
    def optimize_for_memory(self):
        """Optimize system for memory usage"""
        # Clear caches
        if hasattr(self, '_cache'):
            self._cache.clear()
            
        # Force garbage collection
        self.force_garbage_collection()
        
        # Return memory status
        return self.check_memory_usage()
        
    def can_allocate(self, required_mb):
        """Check if we can allocate required memory"""
        current_usage = self.check_memory_usage()
        return current_usage['available_mb'] >= required_mb
```

---

## 7. Performance Expectations

### 7.1 Target Improvements

| Metric | Current | Target | AIML Reference |
|--------|---------|--------|----------------|
| **Text Classification F1** | ~0.70 | ~0.85 | 0.79-0.93 |
| **Feature Count** | 33 | 200-300 | 1,155 |
| **AUC Improvement** | Baseline | +3-5% | +12.24% |
| **Memory Usage** | ~200MB | <400MB | N/A |
| **Processing Time** | Fast | <2s | N/A |

### 7.2 Resource Optimization Targets

- **RAM Usage**: <400MB for enhanced system
- **Disk Usage**: <2GB for models and features  
- **CPU Usage**: <80% of 0.5 vCPU during peak
- **Response Time**: <2 seconds for classification
- **Accuracy**: 15-20% improvement over baseline

---

## 8. Validation Strategy

### 8.1 Testing Framework

```python
class PerformanceValidator:
    def __init__(self):
        self.test_claims = self._load_test_data()
        self.baseline_metrics = {}
        self.enhanced_metrics = {}
        
    def validate_text_processing(self, classifier):
        """Validate enhanced text processing"""
        results = {}
        
        for claim in self.test_claims:
            prediction = classifier.classify_claim(
                claim['text'], claim['data']
            )
            results[claim['id']] = prediction
            
        # Calculate metrics
        metrics = self._calculate_classification_metrics(results)
        return metrics
        
    def validate_feature_engineering(self, safe_engine):
        """Validate enhanced SAFE"""
        feature_sets = {}
        
        for claim in self.test_claims:
            features = safe_engine.generate_comprehensive_features(claim['data'])
            feature_sets[claim['id']] = features
            
        # Analyze feature quality
        analysis = self._analyze_features(feature_sets)
        return analysis
        
    def _calculate_classification_metrics(self, results):
        """Calculate classification metrics"""
        # Implement precision, recall, F1 calculation
        pass
        
    def _analyze_features(self, feature_sets):
        """Analyze generated features"""
        # Implement feature quality analysis
        pass
```

### 8.2 Benchmarking Plan

1. **Baseline Measurement**: Current system performance
2. **Incremental Testing**: Test each enhancement separately
3. **Integration Testing**: Combined system performance
4. **Load Testing**: Performance under Qdrant constraints
5. **A/B Testing**: Compare against current system

---

## 9. Implementation Checklist

### Phase 1: Foundation (Week 1-2)
- [ ] **Memory Profiling**: Implement memory usage monitoring
- [ ] **Text Model Upgrade**: Replace with DistilBERT domain adapter
- [ ] **Basic Interactions**: Implement top 20 feature interactions
- [ ] **Feature Selection**: Add basic importance-based selection
- [ ] **Testing Framework**: Set up validation infrastructure

### Phase 2: Enhancement (Week 3-4)
- [ ] **CRF Implementation**: Add conditional random fields
- [ ] **Advanced Interactions**: Implement smart interaction generation
- [ ] **Mathematical Features**: Add transformations and derivatives
- [ ] **Performance Optimization**: Memory and speed improvements
- [ ] **Validation Testing**: Comprehensive performance testing

### Phase 3: Production (Week 5-6)
- [ ] **Load Testing**: Validate under Qdrant constraints
- [ ] **Documentation**: Complete implementation docs
- [ ] **Monitoring**: Add performance monitoring
- [ ] **Deployment**: Production deployment preparation

---

## 10. Risk Mitigation

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|---------|------------|
| **Memory Overflow** | Medium | High | Incremental loading, memory monitoring |
| **Performance Degradation** | Medium | Medium | Lazy loading, caching strategies |
| **Model Compatibility** | Low | Medium | Thorough testing, fallback mechanisms |
| **Data Quality Issues** | Medium | Medium | Input validation, error handling |

### 10.2 Resource Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|---------|------------|
| **Qdrant Limits Exceeded** | Medium | High | Memory optimization, feature limits |
| **CPU Throttling** | High | Medium | Efficient algorithms, batching |
| **Disk Space Issues** | Low | Medium | Model compression, cleanup routines |

---

## 11. Success Metrics

### 11.1 Technical Metrics
- **Memory Usage**: <400MB during operation
- **Response Time**: <2 seconds for claim processing
- **Accuracy**: 15%+ improvement in F1-score
- **Feature Quality**: 200+ high-quality features

### 11.2 Business Metrics
- **Fraud Detection Rate**: 10%+ improvement
- **False Positive Rate**: 5%+ reduction  
- **Processing Efficiency**: 25%+ faster decisions
- **System Reliability**: 99%+ uptime

---

## Conclusion

This gap analysis reveals significant opportunities to enhance your insurance fraud detection system by incorporating proven techniques from the AIML framework. The provided solutions are specifically designed to work within Qdrant's free tier constraints while delivering substantial improvements in accuracy and capability.

The key to success is **incremental implementation** with careful **memory management** and **performance monitoring**. The roadmap provided ensures you can achieve AIML-level performance without exceeding resource limitations.

Next steps should focus on implementing the **quick wins** in Phase 1, validating improvements, and then proceeding with the more advanced enhancements in subsequent phases.

---

**Document Version**: 1.0  
**Created**: November 18, 2025  
**Target Environment**: Qdrant Free Tier (1GB RAM, 4GB Disk, 0.5 vCPU)  
**Focus Areas**: Enhanced Text Processing, SAFE Feature Engineering
