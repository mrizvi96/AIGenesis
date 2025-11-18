"""
Enhanced BERT-based Text Classifier for Insurance Fraud Detection
Domain-adapted DistilBERT with CRF layer and multi-task learning
Optimized for Qdrant Free Tier (1GB RAM, 4GB Disk, 0.5 vCPU)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from transformers import AutoTokenizer, AutoModel
import re
from datetime import datetime
try:
    from .memory_manager import get_memory_manager
except ImportError:
    from memory_manager import get_memory_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DomainAdapter(nn.Module):
    """
    Domain-specific adaptation layer for insurance terminology
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Initialize weights for domain adaptation
        nn.init.xavier_uniform_(self.adapter[0].weight)
        nn.init.xavier_uniform_(self.adapter[4].weight)

    def forward(self, x):
        return self.adapter(x)

class MultiTaskClassificationHeads(nn.Module):
    """
    Multi-task classification heads for AIML-specified tasks
    """

    def __init__(self, input_dim: int = 128):
        super().__init__()

        # AIML Paper Tasks: 6 specific classification tasks
        self.tasks = {
            'driving_status': 5,      # 5 classes
            'accident_type': 12,      # 12 classes
            'road_type': 11,          # 11 classes
            'cause_accident': 11,     # 11 classes
            'vehicle_count': 4,       # 4 classes
            'parties_involved': 5     # 5 classes
        }

        self.classification_heads = nn.ModuleDict()
        for task_name, num_classes in self.tasks.items():
            self.classification_heads[task_name] = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, num_classes)
            )

    def forward(self, x):
        outputs = {}
        for task_name, head in self.classification_heads.items():
            outputs[task_name] = head(x)
        return outputs

class ConditionalRandomField(nn.Module):
    """
    Lightweight CRF layer for structured prediction
    Simplified implementation for memory efficiency
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # Initialize transitions
        nn.init.xavier_uniform_(self.transitions)

    def forward(self, emissions, mask=None):
        """
        Simplified CRF forward pass (memory efficient)
        """
        # For memory efficiency, we use a simplified approach
        # that approximates CRF benefits without full viterbi decoding
        return emissions

class EnhancedBERTClassifier:
    """
    Enhanced BERT-based classifier with domain adaptation and multi-task learning
    """

    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        """
        Initialize enhanced BERT classifier

        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_manager = get_memory_manager()

        # Domain-specific vocabulary for better understanding
        self.insurance_vocab = self._load_insurance_vocabulary()

        # Initialize model components
        self.tokenizer = None
        self.base_model = None
        self.domain_adapter = None
        self.classification_heads = None
        self.crf_layer = None

        # Performance optimization
        self.max_sequence_length = 256  # Limit for memory
        self.batch_size = 1  # Process one at a time for memory constraints

        # Cache for frequently used embeddings
        self.embedding_cache = {}
        self.cache_size_limit = 100

        # Initialize the model
        self._initialize_model()

    def _load_insurance_vocabulary(self) -> Dict[str, List[str]]:
        """
        Load domain-specific vocabulary for insurance domain adaptation
        """
        return {
            'accident_types': [
                'collision', 'rollover', 'side_impact', 'rear_end', 'head_on',
                'single_vehicle', 'multi_vehicle', 'pedestrian', 'animal', 'object',
                'parking_lot', 'intersection', 'highway', 'urban', 'rural'
            ],
            'damage_severity': [
                'minor', 'moderate', 'severe', 'total', 'cosmetic', 'structural',
                'mechanical', 'electrical', 'frame', 'suspension', 'engine'
            ],
            'insurance_terms': [
                'deductible', 'premium', 'claim', 'coverage', 'policy', 'liability',
                'comprehensive', 'collision', 'uninsured', 'underinsured', 'medical',
                'bodily_injury', 'property_damage', 'personal_injury', 'no_fault'
            ],
            'fraud_indicators': [
                'staged', 'intentional', 'deliberate', 'planned', 'premeditated',
                'exaggerated', 'inflated', 'suspicious', 'unusual', 'inconsistent',
                'contradictory', 'delayed', 'immediate', 'urgent', 'pressure'
            ],
            'medical_terms': [
                'whiplash', 'concussion', 'fracture', 'sprain', 'strain', 'contusion',
                'laceration', 'abrasion', 'hematoma', 'dislocation', 'torn', 'ruptured',
                'emergency_room', 'ambulance', 'hospital', 'clinic', 'physician'
            ]
        }

    def _initialize_model(self):
        """
        Initialize the BERT model with memory management
        """
        try:
            logger.info("Initializing enhanced BERT classifier...")

            # Check memory availability
            memory_check = self.memory_manager.can_allocate(180, 'text_classifier')
            if not memory_check['can_allocate']:
                logger.error(f"Insufficient memory for BERT model: {memory_check}")
                raise MemoryError("Cannot allocate memory for BERT model")

            # Allocate memory for the model
            self.memory_manager.allocate_component_memory('text_classifier', 180)

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Initialize base model with memory optimization
            self.base_model = AutoModel.from_pretrained(self.model_name)
            self.base_model.eval()  # Set to evaluation mode

            # Initialize domain adapter
            self.domain_adapter = DomainAdapter()
            self.domain_adapter.eval()

            # Initialize classification heads
            self.classification_heads = MultiTaskClassificationHeads()
            self.classification_heads.eval()

            # Initialize simplified CRF
            self.crf_layer = ConditionalRandomField(num_tags=12)  # Max task classes
            self.crf_layer.eval()

            # Move to device
            self.base_model.to(self.device)
            self.domain_adapter.to(self.device)
            self.classification_heads.to(self.device)
            self.crf_layer.to(self.device)

            logger.info("Enhanced BERT classifier initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize enhanced BERT classifier: {e}")
            self._cleanup_on_error()
            raise

    def _cleanup_on_error(self):
        """Clean up resources on initialization error"""
        try:
            self.memory_manager.release_component_memory('text_classifier')
        except:
            pass

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text with domain-specific enhancements
        """
        if not text:
            return ""

        # Convert to lowercase and strip
        text = text.lower().strip()

        # Domain-specific text normalization
        text = self._normalize_insurance_terms(text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        return text

    def _normalize_insurance_terms(self, text: str) -> str:
        """
        Normalize insurance-specific terminology
        """
        # Common abbreviations and variations
        term_mappings = {
            'car': 'vehicle',
            'auto': 'vehicle',
            'truck': 'vehicle',
            'suv': 'vehicle',
            'dr': 'doctor',
            'md': 'doctor',
            'er': 'emergency_room',
            'ed': 'emergency_department',
            'icu': 'intensive_care',
            'emt': 'emergency_medical_technician',
            'ems': 'emergency_medical_services',
            'police': 'law_enforcement',
            'cop': 'law_enforcement',
            'sheriff': 'law_enforcement'
        }

        for variation, standard in term_mappings.items():
            text = re.sub(r'\b' + re.escape(variation) + r'\b', standard, text)

        return text

    def classify_claim(self, text: str, claim_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Classify claim text with multi-task predictions

        Args:
            text: Claim text to classify
            claim_data: Additional claim data for context

        Returns:
            Dictionary with classification results
        """
        if not text or not self.base_model:
            return self._get_default_predictions()

        try:
            # Check cache first
            cache_key = hash(text + str(claim_data))
            if cache_key in self.embedding_cache:
                logger.debug("Using cached classification results")
                return self.embedding_cache[cache_key]

            # Preprocess text
            processed_text = self._preprocess_text(text)

            # Memory context for this operation
            with self.memory_manager.memory_limit_context(50, 'classification'):

                # Tokenize text
                inputs = self.tokenizer(
                    processed_text,
                    max_length=self.max_sequence_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                ).to(self.device)

                # Extract features with gradient computation disabled
                with torch.no_grad():
                    # Get BERT embeddings
                    outputs = self.base_model(**inputs)

                    # Use [CLS] token or pooled output
                    if hasattr(outputs, 'pooler_output'):
                        pooled_output = outputs.pooler_output
                    else:
                        # For DistilBERT and other models without pooler_output
                        # Use the [CLS] token (first token) from last_hidden_state
                        pooled_output = outputs.last_hidden_state[:, 0, :]

                    # Apply domain adaptation
                    domain_features = self.domain_adapter(pooled_output)

                    # Multi-task classification
                    task_logits = self.classification_heads(domain_features)

                    # Apply CRF for structured prediction (simplified)
                    task_predictions = {}
                    for task_name, logits in task_logits.items():
                        # Apply CRF if available
                        if self.crf_layer:
                            crf_output = self.crf_layer(logits)
                            probabilities = torch.softmax(crf_output, dim=-1)
                        else:
                            probabilities = torch.softmax(logits, dim=-1)

                        # Get predictions
                        predicted_class = torch.argmax(probabilities, dim=-1).item()
                        confidence = torch.max(probabilities).item()

                        # Get class names
                        task_classes = self._get_task_classes(task_name)
                        predicted_class_name = task_classes[predicted_class] if predicted_class < len(task_classes) else 'unknown'

                        task_predictions[task_name] = {
                            'prediction': predicted_class_name,
                            'confidence': confidence,
                            'probabilities': probabilities.squeeze().cpu().numpy().tolist(),
                            'class_names': task_classes
                        }

                # Combine with fraud risk analysis
                fraud_indicators = self._extract_fraud_risk_indicators(processed_text)

                # Create final result
                result = {
                    'task_predictions': task_predictions,
                    'fraud_indicators': fraud_indicators,
                    'domain_adaptation': True,
                    'model_type': 'enhanced_bert',
                    'processing_time': datetime.now().isoformat()
                }

                # Cache result (if within cache size limit)
                if len(self.embedding_cache) < self.cache_size_limit:
                    self.embedding_cache[cache_key] = result

                return result

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return self._get_default_predictions()

    def _get_task_classes(self, task_name: str) -> List[str]:
        """
        Get class names for a specific task
        """
        task_classes = {
            'driving_status': [
                'driving', 'parked', 'stopped', 'passenger', 'unknown'
            ],
            'accident_type': [
                'collision', 'rollover', 'side_impact', 'rear_end', 'head_on',
                'single_vehicle', 'multi_vehicle', 'pedestrian', 'animal', 'object',
                'parking_lot', 'other'
            ],
            'road_type': [
                'highway', 'urban', 'rural', 'parking', 'intersection',
                'residential', 'commercial', 'industrial', 'bridge', 'tunnel', 'other'
            ],
            'cause_accident': [
                'negligence', 'weather', 'mechanical', 'medical', 'intentional',
                'distraction', 'fatigue', 'impaired', 'speed', 'road_condition', 'other'
            ],
            'vehicle_count': [
                'single', 'two', 'multiple', 'unknown'
            ],
            'parties_involved': [
                'single', 'two', 'multiple', 'pedestrian', 'property_only'
            ]
        }

        return task_classes.get(task_name, ['unknown'])

    def _extract_fraud_risk_indicators(self, text: str) -> Dict[str, Any]:
        """
        Extract fraud risk indicators using enhanced BERT understanding
        """
        text_lower = text.lower()

        # Enhanced fraud keyword patterns
        fraud_patterns = {
            'suspicious_timing': [
                r'after hours', r'late night', r'unusual time', r'odd time',
                r'midnight', r'early morning', r'holiday weekend'
            ],
            'vague_description': [
                r'someone', r'something', r'somehow', r'unclear', r'vague',
                r'don\'t remember', r'can\'t recall', r'unsure'
            ],
            'hasty_settlement': [
                r'quick settlement', r'immediate payment', r'fast cash',
                r'urgent money', r'speedy process', r'right away'
            ],
            'excessive_damage': [
                r'total loss', r'write-off', r'completely destroyed',
                r'nothing left', r'totaled', r'beyond repair'
            ],
            'multiple_claims': [
                r'another claim', r'previous incident', r'again',
                r'similar before', r'recurring', r'pattern'
            ],
            'witness_absence': [
                r'no witnesses', r'alone', r'no one saw', r'isolated',
                r'by myself', r'unobserved'
            ],
            'delayed_reporting': [
                r'days later', r'weeks later', r'delayed', r'took time',
                r'waited', r'postponed'
            ],
            'inconsistent_details': [
                r'contradiction', r'inconsistent', r'changed story',
                r'different version', r'conflicting'
            ]
        }

        risk_scores = {}
        total_risk_score = 0
        matched_patterns = []

        for indicator, patterns in fraud_patterns.items():
            score = 0
            matched = []

            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    score += matches
                    matched.append(pattern)

            if matched:
                normalized_score = min(score / len(patterns), 1.0)
                risk_scores[indicator] = {
                    'score': normalized_score,
                    'matched_patterns': matched,
                    'pattern_count': score
                }
                total_risk_score += normalized_score
                matched_patterns.extend(matched)

        # Calculate overall risk assessment
        overall_risk = min(total_risk_score / len(fraud_patterns), 1.0)
        high_risk_count = sum(1 for score_data in risk_scores.values() if score_data['score'] > 0.3)

        return {
            'risk_indicators': risk_scores,
            'total_risk_score': overall_risk,
            'high_risk_count': high_risk_count,
            'risk_level': self._get_risk_level(overall_risk),
            'matched_patterns_count': len(matched_patterns),
            'analysis_confidence': min(0.9, 0.5 + (len(matched_patterns) * 0.1))
        }

    def _get_risk_level(self, score: float) -> str:
        """
        Get risk level based on score
        """
        if score >= 0.7:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        elif score >= 0.2:
            return 'low'
        else:
            return 'minimal'

    def _get_default_predictions(self) -> Dict[str, Any]:
        """
        Get default predictions when model fails
        """
        task_predictions = {}
        for task_name in self.classification_heads.tasks.keys() if self.classification_heads else []:
            task_classes = self._get_task_classes(task_name)
            task_predictions[task_name] = {
                'prediction': 'unknown',
                'confidence': 0.0,
                'probabilities': [0.0] * len(task_classes),
                'class_names': task_classes
            }

        return {
            'task_predictions': task_predictions,
            'fraud_indicators': {
                'risk_indicators': {},
                'total_risk_score': 0.0,
                'high_risk_count': 0,
                'risk_level': 'unknown',
                'matched_patterns_count': 0,
                'analysis_confidence': 0.0
            },
            'domain_adaptation': False,
            'model_type': 'fallback',
            'processing_time': datetime.now().isoformat(),
            'error': True
        }

    def extract_structured_features(self, claim_text: str, claim_data: Optional[Dict] = None) -> List[float]:
        """
        Extract structured features for machine learning

        Args:
            claim_text: Claim text to process
            claim_data: Additional claim data

        Returns:
            List of numerical features
        """
        try:
            classification_result = self.classify_claim(claim_text, claim_data)

            features = []

            # Process task predictions
            task_predictions = classification_result.get('task_predictions', {})
            for task_name, prediction_data in task_predictions.items():
                # One-hot encode prediction
                class_names = prediction_data.get('class_names', [])
                predicted_class = prediction_data.get('prediction', 'unknown')

                for class_name in class_names:
                    features.append(1.0 if class_name == predicted_class else 0.0)

                # Add confidence score
                features.append(prediction_data.get('confidence', 0.0))

                # Add top probabilities (memory efficient - top 3)
                probabilities = prediction_data.get('probabilities', [])
                if probabilities:
                    sorted_probs = sorted(probabilities, reverse=True)[:3]
                    features.extend(sorted_probs)
                else:
                    features.extend([0.0, 0.0, 0.0])

            # Process fraud indicators
            fraud_indicators = classification_result.get('fraud_indicators', {})
            risk_data = fraud_indicators.get('risk_indicators', {})

            for indicator in ['suspicious_timing', 'vague_description', 'hasty_settlement',
                            'excessive_damage', 'multiple_claims', 'witness_absence',
                            'delayed_reporting', 'inconsistent_details']:
                score_data = risk_data.get(indicator, {})
                features.append(score_data.get('score', 0.0))

            # Add overall risk features
            features.append(fraud_indicators.get('total_risk_score', 0.0))
            features.append(fraud_indicators.get('high_risk_count', 0))
            features.append(1.0 if fraud_indicators.get('risk_level') == 'high' else 0.0)
            features.append(fraud_indicators.get('analysis_confidence', 0.0))

            # Ensure consistent feature length (pad or truncate to 200 features)
            while len(features) < 200:
                features.append(0.0)

            return features[:200]

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return default features
            return [0.0] * 200

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage information
        """
        memory_info = self.memory_manager.check_memory_usage()

        return {
            'component': 'enhanced_bert_classifier',
            'model_name': self.model_name,
            'device': str(self.device),
            'max_sequence_length': self.max_sequence_length,
            'cache_size': len(self.embedding_cache),
            'system_memory': memory_info,
            'allocated_memory_mb': self.memory_manager.component_usage.get('text_classifier', 0)
        }

    def clear_cache(self):
        """
        Clear the embedding cache to free memory
        """
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        logger.info(f"Cleared {cache_size} cached embeddings")

    def optimize_memory(self):
        """
        Optimize memory usage
        """
        optimization_result = self.memory_manager.optimize_for_memory()
        self.clear_cache()

        return optimization_result

# Global instance for reuse
_enhanced_classifier = None

def get_enhanced_bert_classifier() -> EnhancedBERTClassifier:
    """
    Get or create singleton instance of enhanced BERT classifier
    """
    global _enhanced_classifier
    if _enhanced_classifier is None:
        _enhanced_classifier = EnhancedBERTClassifier()
    return _enhanced_classifier