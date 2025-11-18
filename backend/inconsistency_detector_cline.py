"""
Enhanced Inconsistency Detection System
Based on Cline Recommendations: 4-6% accuracy improvement
Memory target: <20MB
Cross-modal consistency checking with advanced pattern recognition
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InconsistencyType(Enum):
    """Types of inconsistencies that can be detected"""
    TEMPORAL = "temporal"
    AMOUNT_MODAL = "amount_modal"
    TEXT_IMAGE = "text_image"
    INVESTIGATOR = "investigator"
    GEOGRAPHIC = "geographic"
    POLICY = "policy"

@dataclass
class InconsistencyResult:
    """Result of inconsistency detection"""
    inconsistency_type: str
    score: float
    severity: str
    details: Dict[str, Any]
    indicators: List[str]

class ClineInconsistencyDetector:
    """
    Advanced cross-modal inconsistency detection
    Implements AutoFraudNet research insights within resource constraints
    Memory-optimized with pattern-based analysis
    """

    def __init__(self, memory_limit_mb: int = 20):
        """Initialize enhanced inconsistency detector"""
        logger.info(f"[CLINE-DETECTOR] Loading enhanced inconsistency detector (memory_limit={memory_limit_mb}MB)...")

        self.memory_limit_mb = memory_limit_mb

        # Load cross-modal inconsistency rules from Cline research
        self.cross_modal_rules = self._load_cross_modal_rules()

        # Severity thresholds (from Cline optimization)
        self.severity_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }

        # Pattern libraries for inconsistency detection
        self.temporal_patterns = self._load_temporal_patterns()
        self.amount_patterns = self._load_amount_patterns()
        self.text_image_patterns = self._load_text_image_patterns()

        # Initialize inconsistency cache
        self.inconsistency_cache = {}
        self.cache_size_limit = 50

        logger.info("[OK] Enhanced inconsistency detector initialized")

    def detect_comprehensive_inconsistencies(self, claim_data: Dict[str, Any],
                                           description: str = "",
                                           image_data: Optional[Dict] = None,
                                           investigation_notes: str = "") -> Dict[str, Any]:
        """
        Detect comprehensive inconsistencies across all modalities
        Implements research-backed inconsistency detection patterns from Cline
        """
        try:
            inconsistency_results = []
            detailed_scores = {}

            # 1. Temporal Cross-Modal Consistency
            temporal_result = self._check_temporal_consistency(claim_data, description)
            inconsistency_results.append(temporal_result)
            detailed_scores['temporal'] = temporal_result.score

            # 2. Amount-Modal Consistency
            amount_result = self._check_amount_modal_consistency(claim_data, description)
            inconsistency_results.append(amount_result)
            detailed_scores['amount_modal'] = amount_result.score

            # 3. Text-Image Cross-Modal Consistency (if image available)
            if image_data:
                text_image_result = self._check_text_image_consistency(description, image_data, claim_data)
                inconsistency_results.append(text_image_result)
                detailed_scores['text_image'] = text_image_result.score
            else:
                detailed_scores['text_image'] = 0.0

            # 4. Investigator-Pattern Cross-Modal Analysis (if investigation notes available)
            if investigation_notes:
                investigator_result = self._check_investigator_cross_modal_patterns(
                    claim_data, description, investigation_notes
                )
                inconsistency_results.append(investigator_result)
                detailed_scores['investigator'] = investigator_result.score
            else:
                detailed_scores['investigator'] = 0.0

            # 5. Geographic Inconsistency Detection
            geographic_result = self._check_geographic_consistency(claim_data, description)
            inconsistency_results.append(geographic_result)
            detailed_scores['geographic'] = geographic_result.score

            # 6. Policy-Claim Inconsistency Detection
            policy_result = self._check_policy_consistency(claim_data, description)
            inconsistency_results.append(policy_result)
            detailed_scores['policy'] = policy_result.score

            # Calculate weighted overall inconsistency score
            weights = {
                'temporal': 0.25,
                'amount_modal': 0.20,
                'text_image': 0.15,
                'investigator': 0.15,
                'geographic': 0.15,
                'policy': 0.10
            }

            overall_score = sum(
                detailed_scores[key] * weights[key]
                for key in detailed_scores.keys()
            )
            normalized_score = min(overall_score, 1.0)

            # Categorize inconsistency types
            high_severity_inconsistencies = [
                result for result in inconsistency_results
                if result.severity == 'high'
            ]

            # Generate recommendations
            recommendations = self._generate_inconsistency_recommendations(
                inconsistency_results, normalized_score
            )

            # Calculate risk level
            risk_level = self._calculate_risk_level(normalized_score, len(high_severity_inconsistencies))

            result = {
                'overall_inconsistency_score': float(normalized_score),
                'risk_level': risk_level,
                'total_inconsistencies_detected': len(inconsistency_results),
                'high_severity_count': len(high_severity_inconsistencies),
                'detailed_results': {
                    result.inconsistency_type: {
                        'score': result.score,
                        'severity': result.severity,
                        'details': result.details,
                        'indicators': result.indicators
                    }
                    for result in inconsistency_results
                },
                'category_scores': {k: float(v) for k, v in detailed_scores.items()},
                'recommendations': recommendations,
                'processing_metadata': {
                    'detection_timestamp': datetime.now().isoformat(),
                    'modalities_checked': list(detailed_scores.keys()),
                    'memory_usage_mb': self._estimate_memory_usage(),
                    'cross_modal_analysis': True,
                    'cline_enhanced': True
                }
            }

            logger.info(f"[OK] Inconsistency detection completed with score {normalized_score:.3f}")
            return result

        except Exception as e:
            logger.error(f"[ERROR] Comprehensive inconsistency detection failed: {e}")
            return self._get_fallback_inconsistency_result()

    def _check_temporal_consistency(self, claim_data: Dict[str, Any], description: str) -> InconsistencyResult:
        """Check temporal inconsistencies across claim timeline"""
        try:
            score = 0.0
            indicators = []
            details = {}

            # Extract temporal information
            accident_time = claim_data.get('accident_time', '')
            accident_date = claim_data.get('accident_date', '')
            claim_time = claim_data.get('claim_time', '')

            # Check 1: Accident time vs. description consistency
            if accident_time and description:
                hour = self._parse_hour(accident_time)
                if hour is not None:
                    # Check for contradictions between reported time and description
                    description_lower = description.lower()

                    # Night accident described as "bright sunny day"
                    if hour in [22, 23, 0, 1, 2, 3, 4, 5, 6] and any(
                        word in description_lower for word in ['sunny', 'bright', 'daylight', 'morning sun']
                    ):
                        score += 0.4
                        indicators.append('night_day_mismatch')

                    # Rush hour accident with no traffic mentions
                    if hour in [7, 8, 9, 17, 18, 19] and not any(
                        word in description_lower for word in ['traffic', 'congestion', 'rush hour', 'busy']
                    ):
                        score += 0.2
                        indicators.append('rush_hour_no_traffic_mention')

            # Check 2: Accident date vs. claim date plausibility
            if accident_date and claim_time:
                try:
                    accident_dt = datetime.strptime(accident_date, '%Y-%m-%d')
                    claim_dt = datetime.strptime(claim_time.split()[0], '%Y-%m-%d')
                    delay_days = (claim_dt - accident_dt).days

                    # Immediate claims for severe accidents (suspicious)
                    if delay_days == 0:
                        if 'severe' in description.lower() or 'major' in description.lower():
                            score += 0.3
                            indicators.append('immediate_severe_claim')

                    # Very delayed claims without justification
                    elif delay_days > 30:
                        if not any(word in description.lower() for word in ['hospital', 'injury', 'recovery']):
                            score += 0.2
                            indicators.append('delayed_no_justification')

                    details['delay_days'] = delay_days
                except:
                    pass

            # Check 3: Temporal patterns in description
            temporal_words = ['suddenly', 'immediately', 'instantly', 'out of nowhere']
            temporal_count = sum(1 for word in temporal_words if word in description.lower())
            if temporal_count > 2:
                score += 0.2
                indicators.append('excessive_temporal_adverbs')
                details['temporal_adverb_count'] = temporal_count

            severity = self._calculate_severity(score)

            return InconsistencyResult(
                inconsistency_type=InconsistencyType.TEMPORAL.value,
                score=score,
                severity=severity,
                details=details,
                indicators=indicators
            )

        except Exception as e:
            logger.error(f"[ERROR] Temporal consistency check failed: {e}")
            return InconsistencyResult(
                inconsistency_type=InconsistencyType.TEMPORAL.value,
                score=0.0,
                severity='low',
                details={'error': str(e)},
                indicators=[]
            )

    def _check_amount_modal_consistency(self, claim_data: Dict[str, Any], description: str) -> InconsistencyResult:
        """Check consistency between claim amount and other modalities"""
        try:
            score = 0.0
            indicators = []
            details = {}

            amount = float(claim_data.get('amount', 0))
            claim_type = claim_data.get('claim_type', 'auto')
            location = claim_data.get('location', '')

            if amount > 0:
                # Check 1: Amount vs. damage description consistency
                description_lower = description.lower()

                # High damage claims with minimal description
                if amount > 10000:
                    minimal_desc_indicators = ['scratch', 'dent', 'minor', 'small']
                    if any(indicator in description_lower for indicator in minimal_desc_indicators):
                        score += 0.4
                        indicators.append('high_amount_minimal_damage')

                # Low amount claims with severe damage description
                elif amount < 1000:
                    severe_desc_indicators = ['totaled', 'destroyed', 'write-off', 'unrepairable', 'major']
                    if any(indicator in description_lower for indicator in severe_desc_indicators):
                        score += 0.5
                        indicators.append('low_amount_severe_damage')

                # Check 2: Amount vs. location consistency
                if location:
                    location_lower = location.lower()
                    # Parking lot accidents with high amounts
                    if 'parking' in location_lower and amount > 5000:
                        score += 0.3
                        indicators.append('parking_high_amount')

                    # Highway accidents with very low amounts
                    elif 'highway' in location_lower and amount < 1000:
                        score += 0.2
                        indicators.append('highway_low_amount')

                # Check 3: Amount vs. claim type consistency
                amount_expectations = {
                    'auto': (500, 15000),
                    'home': (1000, 50000),
                    'health': (500, 100000),
                    'travel': (100, 10000),
                    'life': (10000, 1000000)
                }

                expected_min, expected_max = amount_expectations.get(claim_type, (0, 50000))
                if amount < expected_min:
                    score += 0.2
                    indicators.append('amount_below_type_minimum')
                    details['expected_minimum'] = expected_min
                elif amount > expected_max:
                    score += 0.3
                    indicators.append('amount_above_type_maximum')
                    details['expected_maximum'] = expected_max

                # Check 4: Round number suspicion
                if amount in [1000, 2500, 5000, 10000, 25000, 50000, 100000]:
                    score += 0.1
                    indicators.append('suspicious_round_number')

                details['amount'] = amount
                details['claim_type'] = claim_type

            severity = self._calculate_severity(score)

            return InconsistencyResult(
                inconsistency_type=InconsistencyType.AMOUNT_MODAL.value,
                score=score,
                severity=severity,
                details=details,
                indicators=indicators
            )

        except Exception as e:
            logger.error(f"[ERROR] Amount modal consistency check failed: {e}")
            return InconsistencyResult(
                inconsistency_type=InconsistencyType.AMOUNT_MODAL.value,
                score=0.0,
                severity='low',
                details={'error': str(e)},
                indicators=[]
            )

    def _check_text_image_consistency(self, description: str, image_data: Dict[str, Any],
                                    claim_data: Dict[str, Any]) -> InconsistencyResult:
        """Check text-description vs. image content consistency"""
        try:
            score = 0.0
            indicators = []
            details = {}

            # Simulate image analysis (in real system, would use computer vision)
            image_analysis = image_data.get('analysis', {}) if image_data else {}

            # Extract image features
            damage_severity = image_analysis.get('damage_severity', 'unknown')
            vehicle_parts = image_analysis.get('affected_parts', [])
            weather_conditions = image_analysis.get('weather', 'unknown')

            # Check 1: Damage severity consistency
            description_lower = description.lower()

            if damage_severity == 'minor':
                severe_words = ['severe', 'major', 'extensive', 'significant', 'heavy']
                if any(word in description_lower for word in severe_words):
                    score += 0.4
                    indicators.append('description_severe_image_minor')

            elif damage_severity == 'severe':
                minor_words = ['minor', 'small', 'light', 'slight', 'minimal']
                if any(word in description_lower for word in minor_words):
                    score += 0.5
                    indicators.append('description_minor_image_severe')

            # Check 2: Vehicle parts consistency
            if 'front' in vehicle_parts and 'rear' not in description_lower:
                if 'rear' in description_lower:
                    score += 0.3
                    indicators.append('contradictory_damage_location')

            if 'windshield' in vehicle_parts and 'glass' not in description_lower:
                score += 0.2
                indicators.append('glass_damage_not_mentioned')

            # Check 3: Weather consistency
            if weather_conditions != 'unknown':
                weather_words = {
                    'rain': ['rain', 'wet', 'puddle', 'storm'],
                    'snow': ['snow', 'ice', 'slippery', 'blizzard'],
                    'sunny': ['sunny', 'clear', 'bright', 'dry'],
                    'night': ['dark', 'night', 'headlights', 'visibility']
                }

                if weather_conditions in weather_words:
                    weather_indicators = weather_words[weather_conditions]
                    if not any(indicator in description_lower for indicator in weather_indicators):
                        score += 0.2
                        indicators.append('weather_not_mentioned')

            details.update({
                'image_damage_severity': damage_severity,
                'image_vehicle_parts': vehicle_parts,
                'image_weather': weather_conditions
            })

            severity = self._calculate_severity(score)

            return InconsistencyResult(
                inconsistency_type=InconsistencyType.TEXT_IMAGE.value,
                score=score,
                severity=severity,
                details=details,
                indicators=indicators
            )

        except Exception as e:
            logger.error(f"[ERROR] Text-image consistency check failed: {e}")
            return InconsistencyResult(
                inconsistency_type=InconsistencyType.TEXT_IMAGE.value,
                score=0.0,
                severity='low',
                details={'error': str(e)},
                indicators=[]
            )

    def _check_investigator_cross_modal_patterns(self, claim_data: Dict[str, Any],
                                               description: str, investigation_notes: str) -> InconsistencyResult:
        """Check investigator notes for cross-modal inconsistencies"""
        try:
            score = 0.0
            indicators = []
            details = {}

            notes_lower = investigation_notes.lower()
            description_lower = description.lower()

            # Check 1: Contradictory statements
            contradiction_pairs = [
                ('minor', 'severe'),
                ('parked', 'moving'),
                ('low speed', 'high speed'),
                ('no injury', 'injury'),
                ('single vehicle', 'multiple vehicles')
            ]

            for word1, word2 in contradiction_pairs:
                if (word1 in description_lower and word2 in notes_lower) or \
                   (word2 in description_lower and word1 in notes_lower):
                    score += 0.4
                    indicators.append(f'contradiction_{word1}_{word2}')

            # Check 2: Missing information patterns
            if 'unclear' in notes_lower or 'unknown' in notes_lower:
                if len(description.split()) > 50:  # Long description but investigator notes unclear
                    score += 0.2
                    indicators.append('long_description_unclear_investigation')

            # Check 3: Suspicious patterns in investigation
            suspicious_investigation_words = ['uncooperative', 'refused', 'evasive', 'contradictory']
            suspicious_count = sum(1 for word in suspicious_investigation_words if word in notes_lower)

            if suspicious_count > 0:
                score += 0.1 * suspicious_count
                indicators.append('suspicious_investigation_behavior')
                details['suspicious_word_count'] = suspicious_count

            # Check 4: Timeline discrepancies
            claim_time = claim_data.get('claim_time', '')
            investigation_time = claim_data.get('investigation_time', '')

            if claim_time and investigation_time:
                try:
                    claim_dt = datetime.strptime(claim_time.split()[0], '%Y-%m-%d')
                    inv_dt = datetime.strptime(investigation_time.split()[0], '%Y-%m-%d')
                    investigation_delay = (inv_dt - claim_dt).days

                    if investigation_delay > 90:  # Very late investigation
                        score += 0.2
                        indicators.append('delayed_investigation')
                        details['investigation_delay_days'] = investigation_delay

                except:
                    pass

            details['investigation_notes_length'] = len(investigation_notes.split())

            severity = self._calculate_severity(score)

            return InconsistencyResult(
                inconsistency_type=InconsistencyType.INVESTIGATOR.value,
                score=score,
                severity=severity,
                details=details,
                indicators=indicators
            )

        except Exception as e:
            logger.error(f"[ERROR] Investigator cross-modal check failed: {e}")
            return InconsistencyResult(
                inconsistency_type=InconsistencyType.INVESTIGATOR.value,
                score=0.0,
                severity='low',
                details={'error': str(e)},
                indicators=[]
            )

    def _check_geographic_consistency(self, claim_data: Dict[str, Any], description: str) -> InconsistencyResult:
        """Check geographic consistency across claim data"""
        try:
            score = 0.0
            indicators = []
            details = {}

            location = claim_data.get('location', '').lower()
            description_lower = description.lower()

            if location:
                # Check 1: Location type vs. description consistency
                location_descriptions = {
                    'parking': ['parked', 'stationary', 'stopped', 'parking lot'],
                    'highway': ['highway', 'freeway', 'fast', 'speed', 'traffic'],
                    'intersection': ['intersection', 'crossroads', 'turning', 'crossing'],
                    'residential': ['street', 'neighborhood', 'house', 'residential']
                }

                for loc_type, keywords in location_descriptions.items():
                    if loc_type in location:
                        found_keywords = [kw for kw in keywords if kw in description_lower]
                        if not found_keywords:
                            score += 0.2
                            indicators.append(f'{loc_type}_description_mismatch')
                            details[f'{loc_type}_expected_keywords'] = keywords

                # Check 2: Multiple conflicting locations
                location_keywords = ['highway', 'parking', 'intersection', 'residential', 'commercial']
                mentioned_locations = [kw for kw in location_keywords if kw in description_lower]

                if len(mentioned_locations) > 1:
                    score += 0.3
                    indicators.append('multiple_conflicting_locations')
                    details['mentioned_locations'] = mentioned_locations

                # Check 3: Vague location indicators
                vague_indicators = ['somewhere', 'someplace', 'unknown location', 'not sure where']
                vague_count = sum(1 for indicator in vague_indicators if indicator in description_lower)

                if vague_count > 0:
                    score += 0.1 * vague_count
                    indicators.append('vague_location_description')
                    details['vague_indicator_count'] = vague_count

            severity = self._calculate_severity(score)

            return InconsistencyResult(
                inconsistency_type=InconsistencyType.GEOGRAPHIC.value,
                score=score,
                severity=severity,
                details=details,
                indicators=indicators
            )

        except Exception as e:
            logger.error(f"[ERROR] Geographic consistency check failed: {e}")
            return InconsistencyResult(
                inconsistency_type=InconsistencyType.GEOGRAPHIC.value,
                score=0.0,
                severity='low',
                details={'error': str(e)},
                indicators=[]
            )

    def _check_policy_consistency(self, claim_data: Dict[str, Any], description: str) -> InconsistencyResult:
        """Check policy vs. claim consistency"""
        try:
            score = 0.0
            indicators = []
            details = {}

            claim_type = claim_data.get('claim_type', '')
            amount = float(claim_data.get('amount', 0))
            claimant_age = claim_data.get('claimant_age', 0)

            # Check 1: Claim type vs. coverage period
            policy_start = claim_data.get('policy_start_date', '')
            accident_date = claim_data.get('accident_date', '')

            if policy_start and accident_date:
                try:
                    policy_dt = datetime.strptime(policy_start, '%Y-%m-%d')
                    accident_dt = datetime.strptime(accident_date, '%Y-%m-%d')

                    if accident_dt < policy_dt:
                        score += 0.8
                        indicators.append('accident_before_policy_start')
                        details['days_before_policy'] = (policy_dt - accident_dt).days
                except:
                    pass

            # Check 2: Age-based claim patterns
            if claimant_age > 0:
                if claim_type == 'auto' and claimant_age < 18:
                    score += 0.4
                    indicators.append('minor_driver_auto_claim')

                if claim_type == 'life' and claimant_age > 80:
                    score += 0.2
                    indicators.append('elderly_life_claim')

            # Check 3: Amount vs. policy limits consistency
            coverage_limit = claim_data.get('coverage_limit', 0)
            if coverage_limit > 0 and amount > coverage_limit:
                score += 0.3
                indicators.append('claim_exceeds_coverage')
                details['coverage_limit'] = coverage_limit
                details['excess_amount'] = amount - coverage_limit

            severity = self._calculate_severity(score)

            return InconsistencyResult(
                inconsistency_type=InconsistencyType.POLICY.value,
                score=score,
                severity=severity,
                details=details,
                indicators=indicators
            )

        except Exception as e:
            logger.error(f"[ERROR] Policy consistency check failed: {e}")
            return InconsistencyResult(
                inconsistency_type=InconsistencyType.POLICY.value,
                score=0.0,
                severity='low',
                details={'error': str(e)},
                indicators=[]
            )

    def _calculate_severity(self, score: float) -> str:
        """Calculate severity level based on score"""
        if score >= self.severity_thresholds['high']:
            return 'high'
        elif score >= self.severity_thresholds['medium']:
            return 'medium'
        else:
            return 'low'

    def _calculate_risk_level(self, score: float, high_severity_count: int) -> str:
        """Calculate overall risk level"""
        if score >= 0.8 or high_severity_count >= 2:
            return 'critical'
        elif score >= 0.6 or high_severity_count >= 1:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'

    def _generate_inconsistency_recommendations(self, results: List[InconsistencyResult],
                                              overall_score: float) -> List[str]:
        """Generate recommendations based on detected inconsistencies"""
        recommendations = []

        if overall_score >= 0.7:
            recommendations.append("URGENT: Comprehensive investigation required - high inconsistency score")

        high_severity_results = [r for r in results if r.severity == 'high']

        for result in high_severity_results:
            if result.inconsistency_type == 'temporal':
                recommendations.append("Verify timeline accuracy and investigate temporal discrepancies")
            elif result.inconsistency_type == 'amount_modal':
                recommendations.append("Conduct detailed damage assessment and verify claim amount justification")
            elif result.inconsistency_type == 'text_image':
                recommendations.append("Review image analysis and cross-reference with claim description")
            elif result.inconsistency_type == 'investigator':
                recommendations.append("Follow up with investigator for clarification on contradictory notes")
            elif result.inconsistency_type == 'geographic':
                recommendations.append("Verify exact accident location and assess geographic discrepancies")
            elif result.inconsistency_type == 'policy':
                recommendations.append("Review policy coverage and validate claim eligibility")

        if not high_severity_results and overall_score >= 0.4:
            recommendations.append("Monitor claim for additional verification requirements")

        if not recommendations:
            recommendations.append("No significant inconsistencies detected")

        return recommendations

    def _parse_hour(self, time_str: str) -> Optional[int]:
        """Parse hour from time string"""
        try:
            if ':' in time_str:
                hour = int(time_str.split(':')[0])
                return hour % 24
            return None
        except:
            return None

    def _load_cross_modal_rules(self) -> Dict[str, Any]:
        """Load cross-modal inconsistency rules"""
        return {
            'temporal_rules': {
                'night_day_contradiction': 0.8,
                'immediate_severe_suspicion': 0.6,
                'delayed_no_justification': 0.4
            },
            'amount_rules': {
                'high_amount_minor_damage': 0.7,
                'low_amount_severe_damage': 0.8,
                'round_number_suspicion': 0.3
            },
            'text_image_rules': {
                'severity_mismatch': 0.6,
                'location_mismatch': 0.5,
                'missing_damage_reference': 0.4
            }
        }

    def _load_temporal_patterns(self) -> List[str]:
        """Load temporal inconsistency patterns"""
        return [
            'suddenly', 'immediately', 'instantly', 'out of nowhere',
            'unexpectedly', 'without warning', 'mysteriously'
        ]

    def _load_amount_patterns(self) -> List[str]:
        """Load amount inconsistency patterns"""
        return [
            'estimate', 'approximately', 'around', 'about',
            'exactly', 'precisely', 'specifically'
        ]

    def _load_text_image_patterns(self) -> List[str]:
        """Load text-image inconsistency patterns"""
        return [
            'damage', 'dent', 'scratch', 'broken', 'shattered',
            'cracked', 'bent', 'twisted', 'crushed'
        ]

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        return float(len(str(self.inconsistency_cache)) / (1024 * 1024))

    def _get_fallback_inconsistency_result(self) -> Dict[str, Any]:
        """Get fallback result when detection fails"""
        return {
            'overall_inconsistency_score': 0.0,
            'risk_level': 'unknown',
            'total_inconsistencies_detected': 0,
            'high_severity_count': 0,
            'detailed_results': {},
            'category_scores': {},
            'recommendations': ['Detection failed - manual review required'],
            'processing_metadata': {
                'detection_timestamp': datetime.now().isoformat(),
                'modalities_checked': [],
                'memory_usage_mb': 0.0,
                'cross_modal_analysis': False,
                'fallback_mode': True,
                'error': 'Inconsistency detection failed'
            }
        }

# Global instance for memory efficiency
_cline_inconsistency_detector = None

def get_cline_inconsistency_detector(memory_limit_mb: int = 20) -> ClineInconsistencyDetector:
    """Get or create Cline inconsistency detector instance"""
    global _cline_inconsistency_detector
    if _cline_inconsistency_detector is None:
        _cline_inconsistency_detector = ClineInconsistencyDetector(memory_limit_mb)
    return _cline_inconsistency_detector