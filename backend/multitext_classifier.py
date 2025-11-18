"""
Multi-Task Text Classifier for Enhanced Insurance Fraud Detection
Based on AIML Paper research - optimized for Qdrant free tier constraints
"""

import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

class MultiTaskTextClassifier:
    """Lightweight multi-task classifier optimized for 1GB RAM constraint"""

    def __init__(self):
        """Initialize the multi-task classifier with minimal memory footprint"""
        self.models = {}
        self.load_lightweight_models()

    def load_lightweight_models(self):
        """Load pre-trained classifiers for each fraud detection task"""
        print("[ENHANCED] Loading multi-task text classifier...")

        # Using existing sentence-transformers with task-specific heads
        try:
            self.base_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[OK] Loaded base sentence-transformers model")
        except Exception as e:
            print(f"[ERROR] Failed to load base model: {e}")
            self.base_model = None

        # Define classification tasks for insurance fraud detection
        self.tasks = {
            'driving_status': ['driving', 'parked', 'stopped', 'unknown', 'passenger'],
            'accident_type': ['collision', 'single_vehicle', 'rollover', 'side_impact', 'rear_end', 'other'],
            'road_type': ['highway', 'urban', 'rural', 'parking', 'intersection', 'residential'],
            'cause_accident': ['negligence', 'weather', 'mechanical', 'medical', 'intentional', 'other'],
            'vehicle_count': ['single', 'two', 'multiple'],
            'party_count': ['single', 'multiple'],
            'injury_severity': ['none', 'minor', 'moderate', 'severe', 'critical'],
            'property_damage': ['none', 'minor', 'moderate', 'major', 'total']
        }

        # Pre-compute category embeddings for efficiency
        self._precompute_category_embeddings()

    def _precompute_category_embeddings(self):
        """Pre-compute embeddings for all categories to save processing time"""
        if not self.base_model:
            self.category_embeddings = {}
            return

        print("[ENHANCED] Pre-computing category embeddings...")
        self.category_embeddings = {}

        for task_name, categories in self.tasks.items():
            try:
                category_embeddings = [
                    self.base_model.encode(cat, normalize_embeddings=True)
                    for cat in categories
                ]
                self.category_embeddings[task_name] = category_embeddings
                print(f"[OK] Pre-computed {len(categories)} embeddings for {task_name}")
            except Exception as e:
                print(f"[ERROR] Failed to pre-compute embeddings for {task_name}: {e}")
                self.category_embeddings[task_name] = []

    def classify_text(self, text: str) -> Dict[str, any]:
        """Extract all classification features from claim text"""
        features = {}

        if not text or not self.base_model:
            return self._default_classifications()

        try:
            # Generate text embedding once
            text_embedding = self.base_model.encode(text, normalize_embeddings=True)

            # Classify each task
            for task_name, categories in self.tasks.items():
                if task_name not in self.category_embeddings:
                    continue

                # Use semantic similarity for classification
                category_embeddings = self.category_embeddings[task_name]

                if not category_embeddings:
                    # Fallback to keyword-based classification
                    features.update(self._keyword_classification(text, task_name, categories))
                    continue

                # Compute similarity scores
                similarities = cosine_similarity(
                    text_embedding.reshape(1, -1),
                    np.array(category_embeddings)
                )[0]

                # Find best match
                best_idx = np.argmax(similarities)
                confidence = float(similarities[best_idx])

                features[f'{task_name}'] = categories[best_idx]
                features[f'{task_name}_confidence'] = confidence

                # Add top-3 predictions with confidence
                top3_idx = np.argsort(similarities)[-3:][::-1]
                features[f'{task_name}_top3'] = [
                    (categories[idx], float(similarities[idx]))
                    for idx in top3_idx
                ]

        except Exception as e:
            print(f"[ERROR] Classification failed: {e}")
            return self._default_classifications()

        return features

    def extract_structured_features(self, claim_text: str) -> List[float]:
        """Convert classifications to numerical features for machine learning"""
        classifications = self.classify_text(claim_text)
        numerical_features = []

        # One-hot encode categorical features (memory efficient)
        for task_name, categories in self.tasks.items():
            prediction = classifications.get(f'{task_name}', 'unknown')

            # Create one-hot encoding
            for cat in categories:
                numerical_features.append(1.0 if prediction == cat else 0.0)

            # Add confidence scores
            confidence = classifications.get(f'{task_name}_confidence', 0.0)
            numerical_features.append(confidence)

            # Add top-3 confidence features (reduced for memory)
            top3 = classifications.get(f'{task_name}_top3', [])
            for i in range(min(2, len(top3))):  # Only top 2 for memory efficiency
                numerical_features.append(top3[i][1])

        # Fill missing features to maintain consistent length
        while len(numerical_features) < 100:  # Target 100 features
            numerical_features.append(0.0)

        return numerical_features[:100]  # Limit to 100 features for memory efficiency

    def _keyword_classification(self, text: str, task_name: str, categories: List[str]) -> Dict[str, any]:
        """Fallback keyword-based classification"""
        text_lower = text.lower()
        features = {}

        # Define keyword mappings for each task
        keyword_mappings = {
            'driving_status': {
                'driving': ['driving', 'operating', 'behind wheel'],
                'parked': ['parked', 'stationary', 'stopped'],
                'stopped': ['stopped', 'halted', 'immobile'],
                'passenger': ['passenger', 'rider', 'in car']
            },
            'accident_type': {
                'collision': ['collision', 'crash', 'hit', 'smash'],
                'single_vehicle': ['single', 'alone', 'no other'],
                'rollover': ['rollover', 'flipped', 'overturned'],
                'side_impact': ['side', 't-bone', 'broadside'],
                'rear_end': ['rear', 'back', 'behind']
            },
            'injury_severity': {
                'none': ['no injury', 'unharmed', 'safe'],
                'minor': ['minor', 'slight', 'small'],
                'moderate': ['moderate', 'medium', 'some'],
                'severe': ['severe', 'serious', 'major'],
                'critical': ['critical', 'life-threatening', 'emergency']
            },
            'property_damage': {
                'none': ['no damage', 'intact', 'undamaged'],
                'minor': ['minor', 'small', 'light'],
                'moderate': ['moderate', 'some', 'partial'],
                'major': ['major', 'significant', 'extensive'],
                'total': ['total', 'complete', 'write-off']
            }
        }

        if task_name in keyword_mappings:
            keyword_map = keyword_mappings[task_name]
            scores = {}

            for category, keywords in keyword_map.items():
                if category in categories:
                    score = sum(1 for keyword in keywords if keyword in text_lower)
                    scores[category] = score / len(keywords) if keywords else 0

            if scores:
                best_category = max(scores, key=scores.get)
                best_score = scores[best_category]
                features[f'{task_name}'] = best_category
                features[f'{task_name}_confidence'] = min(best_score * 2, 1.0)  # Boost keyword scores
            else:
                features[f'{task_name}'] = 'unknown'
                features[f'{task_name}_confidence'] = 0.0
        else:
            features[f'{task_name}'] = 'unknown'
            features[f'{task_name}_confidence'] = 0.0

        return features

    def _default_classifications(self) -> Dict[str, any]:
        """Return default classifications when model fails"""
        defaults = {}
        for task_name, categories in self.tasks.items():
            defaults[f'{task_name}'] = 'unknown'
            defaults[f'{task_name}_confidence'] = 0.0
        return defaults

    def extract_fraud_risk_indicators(self, claim_text: str) -> Dict[str, any]:
        """Extract specific fraud risk indicators from text"""
        text_lower = claim_text.lower()

        fraud_keywords = {
            'suspicious_timing': ['after hours', 'late night', 'unusual time', 'odd time'],
            'vague_description': ['someone', 'something', 'somehow', 'unclear', 'vague'],
            'hasty_settlement': ['quick settlement', 'immediate payment', 'fast cash', 'urgent money'],
            'excessive_damage': ['total loss', 'write-off', 'completely destroyed', 'nothing left'],
            'multiple_claims': ['another claim', 'previous incident', 'again', 'similar before'],
            'witness_absence': ['no witnesses', 'alone', 'no one saw', 'isolated'],
            'delayed_reporting': ['days later', 'weeks later', 'delayed', 'took time to report']
        }

        risk_scores = {}
        total_risk_score = 0

        for indicator, keywords in fraud_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            normalized_score = min(score / len(keywords), 1.0)
            risk_scores[indicator] = normalized_score
            total_risk_score += normalized_score

        return {
            'risk_indicators': risk_scores,
            'total_risk_score': min(total_risk_score / len(fraud_keywords), 1.0),
            'high_risk_count': sum(1 for score in risk_scores.values() if score > 0.3)
        }

    def get_memory_usage(self) -> Dict[str, any]:
        """Get memory usage information for monitoring"""
        import sys

        # Estimate memory usage
        model_memory = 0
        if self.base_model:
            model_memory = 150  # MB for sentence-transformers

        embedding_memory = 0
        for embeddings in self.category_embeddings.values():
            embedding_memory += sys.getsizeof(embeddings) / (1024 * 1024)  # Convert to MB

        total_memory = model_memory + embedding_memory

        return {
            'total_memory_mb': total_memory,
            'model_memory_mb': model_memory,
            'embedding_memory_mb': embedding_memory,
            'memory_efficiency': 'low' if total_memory < 50 else 'medium' if total_memory < 100 else 'high'
        }

# Global instance for reuse across requests
_multitext_classifier = None

def get_multitext_classifier() -> MultiTaskTextClassifier:
    """Get or create singleton instance"""
    global _multitext_classifier
    if _multitext_classifier is None:
        _multitext_classifier = MultiTaskTextClassifier()
    return _multitext_classifier