"""
Enhanced Recommender with Multi-Task Classification, SAFE Features, and Inconsistency Detection
Optimized for Qdrant free tier constraints (1GB RAM, 4GB Disk)
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import json
import math

# Import our new components
from multitext_classifier import get_multitext_classifier
from safe_features import get_safe_features
from inconsistency_detector import get_inconsistency_detector

class EnhancedClaimsRecommenderAdvanced:
    """Advanced recommendation engine with multi-modal fraud detection"""

    def __init__(self, qdrant_manager):
        """Initialize the enhanced recommender"""
        print("[ENHANCED] Loading Advanced Fraud Detection Engine...")
        self.qdrant = qdrant_manager
        self.load_enhanced_models()

    def load_enhanced_models(self):
        """Load all enhanced detection models"""
        print("[ENHANCED] Initializing advanced detection components...")

        # Initialize enhanced components
        self.multitext_classifier = get_multitext_classifier()
        self.safe_features = get_safe_features()
        self.inconsistency_detector = get_inconsistency_detector()

        # Load medical coding data
        self.medical_codes = self._load_medical_codes()
        self.provider_networks = self._load_provider_networks()

        # Fraud detection patterns
        self.fraud_patterns = self._load_fraud_patterns()

        print("[OK] Advanced Fraud Detection Engine initialized")

    def recommend_outcome(self, claim_data: Dict[str, Any],
                          include_detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive claim recommendation with advanced fraud detection

        Args:
            claim_data: Dictionary containing claim information
            include_detailed_analysis: Whether to include detailed analysis

        Returns:
            Enhanced recommendation with fraud detection
        """
        try:
            # Basic claim processing
            basic_result = self._process_claim_basic(claim_data)

            # Enhanced fraud detection
            fraud_analysis = self._perform_fraud_analysis(claim_data)

            # Medical analysis (if health claim)
            medical_analysis = self._perform_medical_analysis(claim_data)

            # Risk assessment
            risk_assessment = self._perform_risk_assessment(claim_data, fraud_analysis)

            # Generate final recommendation
            recommendation = self._generate_recommendation(
                basic_result, fraud_analysis, medical_analysis, risk_assessment
            )

            result = {
                'claim_id': claim_data.get('claim_id', 'unknown'),
                'recommendation': recommendation['action'],
                'confidence': recommendation['confidence'],
                'justification': recommendation['justification'],
                'processing_time': recommendation['processing_time'],
                'timestamp': datetime.now().isoformat(),
                'basic_analysis': basic_result,
                'fraud_analysis': fraud_analysis,
                'medical_analysis': medical_analysis,
                'risk_assessment': risk_assessment,
                'detailed_analysis': include_detailed_analysis
            }

            return result

        except Exception as e:
            print(f"[ERROR] Advanced recommendation failed: {e}")
            # Fallback to basic recommendation
            return self._fallback_recommendation(claim_data)

    def _process_claim_basic(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process claim using basic similarity search"""
        try:
            # Generate embedding for claim text
            claim_text = claim_data.get('description', '') + ' ' + \
                        claim_data.get('customer_id', '') + ' ' + \
                        claim_data.get('claim_type', '')

            if not claim_text.strip():
                claim_text = "insurance claim"

            # Use existing enhanced embedder
            from enhanced_embeddings import EnhancedMultimodalEmbedder
            embedder = EnhancedMultimodalEmbedder()
            embedding = embedder.embed_text(claim_text)

            # Search similar claims
            collection_name = 'text_claims'
            similar_claims = self.qdrant.search_similar(
                embedding, collection_name=collection_name, limit=5
            )

            # Analyze similar claims
            outcome_distribution = self._analyze_outcome_distribution(similar_claims)

            return {
                'similar_claims_found': len(similar_claims),
                'outcome_distribution': outcome_distribution,
                'text_embedding_dimension': len(embedding),
                'collection_used': collection_name
            }

        except Exception as e:
            print(f"[ERROR] Basic processing failed: {e}")
            return {
                'similar_claims_found': 0,
                'outcome_distribution': {},
                'error': str(e)
            }

    def _perform_fraud_analysis(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive fraud analysis"""
        try:
            analysis_start = datetime.now()

            # Multi-task text classification
            text_analysis = self.multitext_classifier.classify_text(claim_data.get('description', ''))

            # Extract fraud risk indicators
            risk_indicators = self.multitext_classifier.extract_fraud_risk_indicators(
                claim_data.get('description', '')
            )

            # Generate engineered features
            engineered_features = self.safe_features.generate_risk_factors(claim_data)

            # Detect inconsistencies
            inconsistencies = self.inconsistency_detector.detect_inconsistencies(claim_data)

            # Apply fraud pattern matching
            pattern_matches = self._match_fraud_patterns(claim_data, text_analysis)

            # Calculate overall fraud probability
            fraud_probability = self._calculate_fraud_probability(
                text_analysis, risk_indicators, engineered_features,
                inconsistencies, pattern_matches
            )

            processing_time = (datetime.now() - analysis_start).total_seconds()

            return {
                'fraud_probability': fraud_probability,
                'risk_level': self._get_fraud_risk_level(fraud_probability),
                'text_classification': text_analysis,
                'risk_indicators': risk_indicators,
                'engineered_features_count': len(engineered_features),
                'inconsistencies': inconsistencies,
                'pattern_matches': pattern_matches,
                'processing_time_seconds': processing_time,
                'memory_usage': self._get_current_memory_usage()
            }

        except Exception as e:
            print(f"[ERROR] Fraud analysis failed: {e}")
            return {
                'fraud_probability': 0.1,
                'risk_level': 'low',
                'error': str(e)
            }

    def _perform_medical_analysis(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform medical analysis for health insurance claims"""
        try:
            claim_type = claim_data.get('claim_type', '').lower()

            if claim_type not in ['health', 'medical', 'life']:
                return {'applicable': False, 'reason': f'Not a medical claim (type: {claim_type})'}

            description = claim_data.get('description', '').lower()

            # Extract medical entities
            medical_entities = self._extract_medical_entities(description)

            # Match ICD-10 codes
            icd10_matches = self._match_icd10_codes(medical_entities)

            # Match CPT codes
            cpt_matches = self._match_cpt_codes(medical_entities)

            # Check medical necessity
            necessity_score = self._assess_medical_necessity(description, icd10_matches, cpt_matches)

            # Estimate settlement range
            settlement_estimate = self._estimate_medical_settlement(claim_type, medical_entities, icd10_matches)

            # Provider network check
            network_status = self._check_provider_network(claim_data)

            return {
                'applicable': True,
                'medical_entities': medical_entities,
                'icd10_matches': icd10_matches,
                'cpt_matches': cpt_matches,
                'necessity_score': necessity_score,
                'settlement_estimate': settlement_estimate,
                'provider_network_status': network_status,
                'treatment_duration_estimate': self._estimate_treatment_duration(icd10_matches)
            }

        except Exception as e:
            print(f"[ERROR] Medical analysis failed: {e}")
            return {
                'applicable': False,
                'error': str(e)
            }

    def _perform_risk_assessment(self, claim_data: Dict[str, Any],
                               fraud_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        try:
            base_amount = float(claim_data.get('amount', 0))
            claim_type = claim_data.get('claim_type', '')

            risk_factors = {
                'amount_risk': self._assess_amount_risk(base_amount, claim_type),
                'complexity_risk': self._assess_complexity_risk(claim_data),
                'customer_risk': self._assess_customer_risk(claim_data),
                'timing_risk': self._assess_timing_risk(claim_data),
                'documentation_risk': self._assess_documentation_risk(claim_data)
            }

            # Combine with fraud analysis
            fraud_risk = fraud_analysis.get('fraud_probability', 0.0)

            # Calculate overall risk score
            risk_scores = list(risk_factors.values()) + [fraud_risk]
            overall_risk_score = float(np.mean(risk_scores))

            # Risk level determination
            risk_level = self._determine_risk_level(overall_risk_score)

            # Recommended actions
            recommended_actions = self._generate_risk_actions(risk_level, risk_factors, fraud_analysis)

            return {
                'overall_risk_score': overall_risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'recommended_actions': recommended_actions,
                'requires_investigation': bool(overall_risk_score > 0.6),
                'requires_specialist_review': bool(fraud_risk > 0.5)
            }

        except Exception as e:
            print(f"[ERROR] Risk assessment failed: {e}")
            return {
                'overall_risk_score': 0.3,
                'risk_level': 'medium',
                'error': str(e)
            }

    def _generate_recommendation(self, basic_result: Dict[str, Any],
                                fraud_analysis: Dict[str, Any],
                                medical_analysis: Dict[str, Any],
                                risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendation"""
        try:
            fraud_prob = fraud_analysis.get('fraud_probability', 0.0)
            risk_level = risk_assessment.get('risk_level', 'medium')

            # Decision logic
            if fraud_prob > 0.8:
                action = 'REJECT - High Fraud Probability'
                confidence = 0.95
                justification = f"Very high fraud risk detected (score: {fraud_prob:.2f})"

            elif fraud_prob > 0.6:
                action = 'INVESTIGATE - Suspicious Patterns'
                confidence = 0.85
                justification = f"Medium-high fraud risk (score: {fraud_prob:.2f})"

            elif fraud_prob > 0.3:
                action = 'REVIEW_WITH_CAUTION'
                confidence = 0.75
                justification = f"Low-medium fraud risk (score: {fraud_prob:.2f})"

            else:
                action = 'APPROVE - Low Fraud Risk'
                confidence = 0.90
                justification = f"Low fraud risk (score: {fraud_prob:.2f})"

            # Add medical and risk considerations
            if medical_analysis.get('applicable', False):
                action += f" (Medical: {medical_analysis.get('necessity_score', 0.5):.1f})"

            if risk_assessment.get('requires_investigation', False):
                action = f"INVESTIGATE - {action}"
                confidence *= 0.9  # Reduce confidence for complex cases

            return {
                'action': action,
                'confidence': confidence,
                'justification': justification,
                'processing_time': 0.5  # Include processing time estimation
            }

        except Exception as e:
            print(f"[ERROR] Recommendation generation failed: {e}")
            return {
                'action': 'REVIEW - Error',
                'confidence': 0.1,
                'justification': 'Error in analysis',
                'processing_time': 0.1
            }

    def _calculate_fraud_probability(self, text_analysis: Dict[str, Any],
                                   risk_indicators: Dict[str, Any],
                                   engineered_features: List[float],
                                   inconsistencies: Dict[str, Any],
                                   pattern_matches: List[str]) -> float:
        """Calculate comprehensive fraud probability"""
        try:
            # Text classification fraud indicators
            text_risk = self._extract_text_fraud_risk(text_analysis)

            # Risk indicators
            indicator_risk = risk_indicators.get('total_risk_score', 0.0)

            # Engineered features risk
            feature_risk = self._assess_feature_risk(engineered_features)

            # Inconsistency risk
            inconsistency_risk = inconsistencies.get('inconsistency_score', 0.0)

            # Pattern match risk
            pattern_risk = min(len(pattern_matches) / 5.0, 1.0) * 0.7

            # Weighted combination
            fraud_probability = (
                text_risk * 0.3 +
                indicator_risk * 0.25 +
                feature_risk * 0.2 +
                inconsistency_risk * 0.15 +
                pattern_risk * 0.1
            )

            return min(fraud_probability, 1.0)

        except Exception as e:
            print(f"[ERROR] Fraud probability calculation failed: {e}")
            return 0.1

    def _extract_text_fraud_risk(self, text_analysis: Dict[str, Any]) -> float:
        """Extract fraud risk from text classification"""
        try:
            risk_score = 0.0

            # Suspicious claim characteristics
            suspicious_driving = text_analysis.get('driving_status', 'unknown')
            if suspicious_driving in ['unknown', 'passenger']:
                risk_score += 0.2

            suspicious_accident = text_analysis.get('accident_type', 'other')
            if suspicious_accident in ['other']:
                risk_score += 0.15

            injury_severity = text_analysis.get('injury_severity', 'none')
            if injury_severity in ['severe', 'critical']:
                risk_score += 0.1

            # Check confidence levels
            for task in self.multitext_classifier.tasks.keys():
                confidence = text_analysis.get(f'{task}_confidence', 0.0)
                if confidence < 0.3:  # Low confidence might indicate fraud
                    risk_score += 0.05

            return min(risk_score, 1.0)

        except Exception as e:
            print(f"[ERROR] Text fraud risk extraction failed: {e}")
            return 0.0

    def _assess_feature_risk(self, features: List[float]) -> float:
        """Assess risk from engineered features"""
        try:
            if not features:
                return 0.0

            # Calculate statistical risk indicators
            feature_array = np.array(features)

            # High variance can indicate anomalies
            if len(feature_array) > 1:
                variance = float(np.var(feature_array))
                variance_risk = min(variance / 0.25, 1.0)  # Normalize
            else:
                variance_risk = 0.0

            # Number of non-zero features (sparse features might be suspicious)
            non_zero_count = int(np.count_nonzero(feature_array))
            sparsity_risk = 1.0 - (non_zero_count / len(feature_array)) if len(feature_array) > 0 else 0.0

            # Combine risks
            feature_risk = (variance_risk * 0.6 + sparsity_risk * 0.4)

            return min(feature_risk, 1.0)

        except Exception as e:
            print(f"[ERROR] Feature risk assessment failed: {e}")
            return 0.0

    def _match_fraud_patterns(self, claim_data: Dict[str, Any],
                             text_analysis: Dict[str, Any]) -> List[str]:
        """Match claim against known fraud patterns"""
        try:
            matches = []

            description = claim_data.get('description', '').lower()
            amount = float(claim_data.get('amount', 0))

            # Pattern 1: Round amounts
            if amount > 0 and amount % 1000 == 0:
                matches.append('round_amount_pattern')

            # Pattern 2: Vague descriptions
            vague_indicators = ['someone', 'something', 'somehow', 'unclear']
            if any(indicator in description for indicator in vague_indicators):
                matches.append('vague_description_pattern')

            # Pattern 3: Quick settlement mention
            if 'quick settlement' in description or 'fast payment' in description:
                matches.append('quick_settlement_pattern')

            # Pattern 4: Suspicious timing
            if 'late night' in description or 'after hours' in description:
                matches.append('suspicious_timing_pattern')

            return matches

        except Exception as e:
            print(f"[ERROR] Pattern matching failed: {e}")
            return []

    def _load_medical_codes(self) -> Dict[str, Any]:
        """Load ICD-10 and CPT coding data"""
        return {
            'icd10': {
                # Cardiac conditions
                'I21.9': 'Acute myocardial infarction, unspecified',
                'I46.9': 'Cardiac arrest, unspecified',
                'I20.9': 'Angina pectoris, unspecified',
                'I50.9': 'Heart failure, unspecified',
                # Respiratory conditions
                'J44.9': 'Chronic obstructive pulmonary disease, unspecified',
                'J45.909': 'Unspecified asthma, uncomplicated',
                # Injuries
                'S72.0': 'Fracture of neck of femur',
                'S82.8': 'Other fractures of lower leg',
                'S06.0': 'Intracranial injury'
            },
            'cpt': {
                # Cardiology procedures
                '92950': 'Cardioversion',
                '93010': 'Cardiopulmonary resuscitation',
                '33224': 'Cardiac catheterization',
                '93510': 'Electrocardiogram',
                # Orthopedic procedures
                '27236': 'Hip replacement',
                '27130': 'Open reduction of fracture',
                '29505': 'Spinal fusion',
                # General procedures
                '99214': 'Office visit',
                '99215': 'Initial hospital care'
            }
        }

    def _load_provider_networks(self) -> Dict[str, List[str]]:
        """Load provider network information"""
        return {
            'in_network': [
                'General Hospital', 'City Medical Center',
                'Community Health Center', 'Regional Medical Center'
            ],
            'out_of_network': [
                'Private Clinic', 'Specialty Hospital',
                'Out-of-State Provider', 'Concierge Medical'
            ]
        }

    def _load_fraud_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load known fraud patterns"""
        return {
            'staged_accident': {
                'indicators': ['perfect damage', 'no witnesses', 'immediate settlement'],
                'risk_score': 0.8
            },
            'inflated_damage': {
                'indicators': ['excessive amount', 'minor damage', 'no photos'],
                'risk_score': 0.6
            },
            'phantom_claim': {
                'indicators': ['no real accident', 'fake documentation'],
                'risk_score': 0.9
            },
            'repeat_offender': {
                'indicators': ['multiple similar claims', 'pattern recognition'],
                'risk_score': 0.7
            }
        }

    def _fallback_recommendation(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback recommendation when enhanced analysis fails"""
        return {
            'recommendation': 'APPROVE - Fallback Analysis',
            'confidence': 0.6,
            'justification': 'Using fallback analysis due to processing error',
            'processing_time': 0.3,
            'fallback_used': True
        }

    # Helper methods for medical analysis
    def _extract_medical_entities(self, text: str) -> List[str]:
        """Extract medical entities from text"""
        medical_keywords = [
            'chest pain', 'heart attack', 'stroke', 'fracture', 'injury',
            'surgery', 'emergency', 'hospital', 'doctor', 'nurse',
            'treatment', 'medication', 'diagnosis', 'symptom'
        ]

        entities = []
        text_lower = text.lower()
        for keyword in medical_keywords:
            if keyword in text_lower:
                entities.append(keyword)

        return list(set(entities))

    def _match_icd10_codes(self, entities: List[str]) -> List[Dict[str, str]]:
        """Match entities to ICD-10 codes"""
        matches = []
        icd10_codes = self.medical_codes['icd10']

        for code, description in icd10.items():
            # Simplified matching
            for entity in entities:
                if any(word in description.lower() for word in entity.split()):
                    matches.append({'code': code, 'description': description, 'entity': entity})
                    break

        return matches[:5]  # Limit to top 5 matches

    def _match_cpt_codes(self, entities: List[str]) -> List[Dict[str, str]]:
        """Match entities to CPT codes"""
        matches = []
        cpt_codes = self.medical_codes['cpt']

        for code, description in cpt_codes.items():
            for entity in entities:
                if any(word in description.lower() for word in entity.split()):
                    matches.append({'code': code, 'description': description, 'entity': entity})
                    break

        return matches[:5]  # Limit to top 5 matches

    def _assess_medical_necessity(self, description: str, icd10_matches: List, cpt_matches: List) -> float:
        """Assess medical necessity of treatments"""
        necessity_score = 0.5  # Default

        # High necessity conditions
        high_necessity_terms = ['emergency', 'critical', 'acute', 'severe', 'life-threatening']
        for term in high_necessity_terms:
            if term in description.lower():
                necessity_score = 0.9
                break

        # Check for necessary procedures
        if icd10_matches:
            emergency_codes = ['I21.9', 'I46.9', 'J44.9']  # Emergency ICD-10 codes
            if any(match['code'] in emergency_codes for match in icd10_matches):
                necessity_score = 0.85

        return necessity_score

    def _estimate_medical_settlement(self, claim_type: str, entities: List[str], icd10_matches: List) -> Dict[str, float]:
        """Estimate medical settlement range"""
        base_amounts = {'health': 25000, 'medical': 15000, 'life': 100000}
        base_amount = base_amounts.get(claim_type, 15000)

        # Adjust based on severity
        severity_multiplier = 1.0
        if 'emergency' in ' '.join(entities).lower() or 'critical' in ' '.join(entities).lower():
            severity_multiplier = 2.0
        elif 'severe' in ' '.join(entities).lower() or 'major' in ' '.join(entities).lower():
            severity_multiplier = 1.5

        estimated_amount = base_amount * severity_multiplier

        return {
            'estimated_range': {
                'low': estimated_amount * 0.7,
                'high': estimated_amount * 1.3
            },
            'base_amount': base_amount,
            'severity_multiplier': severity_multiplier
        }

    def _check_provider_network(self, claim_data: Dict[str, Any]) -> Dict[str, str]:
        """Check if provider is in-network"""
        provider = claim_data.get('provider', '').lower()

        in_network = self.provider_networks.get('in_network', [])
        out_network = self.provider_networks.get('out_of_network', [])

        if any(network in provider for network in in_network):
            return {'status': 'in_network', 'type': 'standard'}
        elif any(network in provider for network in out_network):
            return {'status': 'out_of_network', 'type': 'higher_cost'}
        else:
            return {'status': 'unknown', 'type': 'standard'}

    def _estimate_treatment_duration(self, icd10_matches: List) -> str:
        """Estimate treatment duration based on ICD-10 codes"""
        if not icd10_matches:
            return "unknown"

        # Simplified duration mapping
        emergency_codes = ['I21.9', 'I46.9']
        surgical_codes = ['S72.0', '27236']
        chronic_codes = ['J44.9', 'I50.9']

        for match in icd10_matches:
            code = match['code']
            if code in emergency_codes:
                return "emergency_care (3-7 days)"
            elif code in surgical_codes:
                return "surgical_recovery (4-12 weeks)"
            elif code in chronic_codes:
                return "chronic_management (ongoing)"

        return "standard_treatment (1-4 weeks)"

    def _get_fraud_risk_level(self, probability: float) -> str:
        """Get fraud risk level from probability"""
        if probability >= 0.8:
            return 'critical'
        elif probability >= 0.6:
            return 'high'
        elif probability >= 0.3:
            return 'medium'
        else:
            return 'low'

    # Risk assessment helper methods
    def _assess_amount_risk(self, amount: float, claim_type: str) -> float:
        """Assess risk based on claim amount"""
        if amount <= 0:
            return 0.0

        avg_amounts = {
            'auto': 3500, 'home': 8000, 'health': 25000,
            'travel': 1500, 'life': 50000
        }

        avg_amount = avg_amounts.get(claim_type, 5000)
        ratio = amount / avg_amount

        if ratio > 5:
            return 0.8
        elif ratio > 3:
            return 0.6
        elif ratio > 2:
            return 0.4
        else:
            return 0.2

    def _assess_complexity_risk(self, claim_data: Dict[str, Any]) -> float:
        """Assess risk based on claim complexity"""
        description = claim_data.get('description', '')
        complexity_score = min(len(description) / 1000.0, 1.0)

        # Very short descriptions can be suspicious
        if len(description) < 50:
            return 0.6

        return complexity_score * 0.3

    def _assess_customer_risk(self, claim_data: Dict[str, Any]) -> float:
        """Assess risk based on customer profile"""
        # Simplified customer risk assessment
        return 0.1  # Default low risk

    def _assess_timing_risk(self, claim_data: Dict[str, Any]) -> float:
        """Assess risk based on timing"""
        return 0.1  # Default low risk

    def _assess_documentation_risk(self, claim_data: Dict[str, Any]) -> float:
        """Assess risk based on documentation"""
        return 0.1  # Default low risk

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score"""
        if risk_score >= 0.7:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'

    def _generate_risk_actions(self, risk_level: str, risk_factors: Dict[str, Any],
                            fraud_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommended actions based on risk level"""
        actions = []

        if risk_level == 'high':
            actions.extend([
                'Investigate fraud indicators',
                'Require documentation verification',
                'Consider special investigator'
            ])

        if fraud_analysis.get('fraud_probability', 0) > 0.5:
            actions.append('Fraud investigation required')

        if risk_factors.get('amount_risk', 0) > 0.6:
            actions.append('Verify damage assessment')

        return actions

    def _get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage for monitoring"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return {
                'embedding_classifier_mb': self.multitext_classifier.get_memory_usage()['total_memory_mb'],
                'safe_features_mb': 20,  # Estimated
                'inconsistency_detector_mb': self.inconsistency_detector.get_memory_usage()['memory_usage_mb'],
                'total_mb': memory_mb
            }
        except:
            return {
                'embedding_classifier_mb': 0,
                'safe_features_mb': 20,
                'inconsistency_detector_mb': 0,
                'total_mb': 0
            }

    def _analyze_outcome_distribution(self, similar_claims: List) -> Dict[str, Any]:
        """Analyze outcomes of similar claims"""
        if not similar_claims:
            return {}

        outcomes = [claim.get('outcome', 'unknown') for claim in similar_claims]
        outcome_counts = {}
        for outcome in outcomes:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        total = len(similar_claims)
        outcome_distribution = {k: v/total for k, v in outcome_counts.items()}

        return outcome_distribution

# Enhanced factory function
def create_enhanced_recommender(qdrant_manager):
    """Create enhanced recommender with all advanced features"""
    return EnhancedClaimsRecommenderAdvanced(qdrant_manager)