"""
Inconsistency Detection System for Insurance Fraud Detection
Lightweight implementation optimized for Qdrant free tier constraints
"""

import re
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
import json

class InconsistencyDetector:
    """Detect inconsistencies across modalities and data sources"""

    def __init__(self):
        """Initialize the inconsistency detector"""
        print("[ENHANCED] Loading inconsistency detection system...")
        self.inconsistency_rules = self._load_inconsistency_rules()
        self.severity_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }

    def _load_inconsistency_rules(self) -> Dict[str, Any]:
        """Load predefined inconsistency detection rules"""
        return {
            'text_image_mismatch': {
                'description': 'Text description does not match image evidence',
                'severity': 0.8,
                'checks': [
                    'damage_mentions_vs_detection',
                    'severity_mismatch',
                    'object_presence_mismatch'
                ]
            },
            'timeline_impossible': {
                'description': 'Claim timeline is logically impossible',
                'severity': 1.0,
                'checks': [
                    'claim_before_accident',
                    'excessive_delay',
                    'temporal_order_violation'
                ]
            },
            'amount_excessive': {
                'description': 'Claim amount is unreasonable for the described damage',
                'severity': 0.6,
                'checks': [
                    'amount_vs_damage_severity',
                    'amount_vs_claim_type',
                    'amount_vs_location'
                ]
            },
            'geographic_implausible': {
                'description': 'Claim location is geographically inconsistent',
                'severity': 0.4,
                'checks': [
                    'location_vs_accident_type',
                    'distance_impossibility',
                    'location_vs_weather'
                ]
            },
            'investigator_suspicious': {
                'description': 'Investigator or adjuster shows suspicious patterns',
                'severity': 0.7,
                'checks': [
                    'pattern_frequency',
                    'settlement_speed',
                    'documentation_completeness'
                ]
            },
            'witness_inconsistency': {
                'description': 'Witness statements are inconsistent',
                'severity': 0.5,
                'checks': [
                    'multiple_witness_disagreement',
                    'witness_unavailability',
                    'witness_timing_mismatch'
                ]
            }
        }

    def detect_inconsistencies(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect inconsistencies across all data sources"""
        inconsistencies = []
        inconsistency_score = 0.0

        try:
            # Perform all consistency checks
            checks_performed = 0

            # Text-Image consistency
            text_image_score = self._check_text_image_consistency(claim_data)
            if text_image_score > 0.3:
                inconsistencies.append('text_image_mismatch')
                inconsistency_score += text_image_score * 0.8
            checks_performed += 1

            # Temporal consistency
            temporal_score = self._check_temporal_consistency(claim_data)
            if temporal_score > 0.3:
                inconsistencies.append('timeline_impossible')
                inconsistency_score += temporal_score
            checks_performed += 1

            # Amount reasonableness
            amount_score = self._check_amount_reasonableness(claim_data)
            if amount_score > 0.3:
                inconsistencies.append('amount_excessive')
                inconsistency_score += amount_score * 0.6
            checks_performed += 1

            # Geographic plausibility
            geo_score = self._check_geographic_plausibility(claim_data)
            if geo_score > 0.3:
                inconsistencies.append('geographic_implausible')
                inconsistency_score += geo_score * 0.4
            checks_performed += 1

            # Investigator patterns
            inv_score = self._check_investigator_patterns(claim_data)
            if inv_score > 0.3:
                inconsistencies.append('investigator_suspicious')
                inconsistency_score += inv_score * 0.7
            checks_performed += 1

            # Witness consistency
            witness_score = self._check_witness_consistency(claim_data)
            if witness_score > 0.3:
                inconsistencies.append('witness_inconsistency')
                inconsistency_score += witness_score * 0.5
            checks_performed += 1

            # Calculate normalized inconsistency score
            if checks_performed > 0:
                normalized_score = min(inconsistency_score / checks_performed, 1.0)
            else:
                normalized_score = 0.0

            return {
                'inconsistencies': inconsistencies,
                'inconsistency_score': normalized_score,
                'risk_level': self._calculate_risk_level(normalized_score),
                'checks_performed': checks_performed,
                'detailed_scores': {
                    'text_image': text_image_score,
                    'temporal': temporal_score,
                    'amount': amount_score,
                    'geographic': geo_score,
                    'investigator': inv_score,
                    'witness': witness_score
                }
            }

        except Exception as e:
            print(f"[ERROR] Inconsistency detection failed: {e}")
            return {
                'inconsistencies': ['detection_error'],
                'inconsistency_score': 0.1,  # Small score for error
                'risk_level': 'low',
                'error': str(e)
            }

    def _check_text_image_consistency(self, claim_data: Dict[str, Any]) -> float:
        """Check if text description matches image evidence"""
        try:
            description = claim_data.get('description', '').lower()
            image_analysis = claim_data.get('image_analysis', {})

            # Check damage mentions vs. detected damage
            text_mentions_damage = self._count_damage_mentions(description)
            image_shows_damage = image_analysis.get('damage_detected', False)
            image_damage_score = image_analysis.get('damage_score', 0.0)

            inconsistency_score = 0.0

            # Major inconsistency: text says no damage but images show damage
            if text_mentions_damage == 0 and image_shows_damage and image_damage_score > 0.5:
                inconsistency_score += 0.8

            # Minor inconsistency: text mentions damage but no damage detected
            if text_mentions_damage > 0 and not image_shows_damage and image_damage_score < 0.2:
                inconsistency_score += 0.4

            # Severity mismatch
            text_severity = self._estimate_text_severity(description)
            image_severity = image_analysis.get('damage_severity', 0)

            severity_diff = abs(text_severity - image_severity)
            if severity_diff > 0.6:  # Large severity mismatch
                inconsistency_score += 0.3

            return min(inconsistency_score, 1.0)

        except:
            return 0.0

    def _check_temporal_consistency(self, claim_data: Dict[str, Any]) -> float:
        """Check if accident timeline is plausible"""
        try:
            accident_time = claim_data.get('accident_time', '')
            claim_time = claim_data.get('claim_submitted_time', '')
            report_time = claim_data.get('police_report_time', '')

            inconsistency_score = 0.0

            # Check if claim submitted before accident
            if accident_time and claim_time:
                try:
                    accident_dt = self._parse_datetime(accident_time)
                    claim_dt = self._parse_datetime(claim_time)

                    if claim_dt and accident_dt and claim_dt < accident_dt:
                        inconsistency_score += 1.0  # Major inconsistency
                except:
                    pass

            # Check for excessive delay
            if accident_time and claim_time:
                try:
                    accident_dt = self._parse_datetime(accident_time)
                    claim_dt = self._parse_datetime(claim_time)

                    if claim_dt and accident_dt:
                        days_passed = (claim_dt - accident_dt).days
                        if days_passed > 30:  # More than 30 days
                            delay_score = min(days_passed / 180.0, 1.0)  # Scale to 0-1
                            inconsistency_score += delay_score * 0.6
                except:
                    pass

            # Check police report timing
            if accident_time and report_time:
                try:
                    accident_dt = self._parse_datetime(accident_time)
                    report_dt = self._parse_datetime(report_time)

                    if report_dt and accident_dt:
                        hours_passed = (report_dt - accident_dt).total_seconds() / 3600
                        if hours_passed < 1:  # Police arrived too quickly (suspicious)
                            inconsistency_score += 0.3
                        elif hours_passed > 72:  # Police report filed too late
                            delay_score = min((hours_passed - 72) / 168.0, 1.0)
                            inconsistency_score += delay_score * 0.4
                except:
                    pass

            return min(inconsistency_score, 1.0)

        except:
            return 0.0

    def _check_amount_reasonableness(self, claim_data: Dict[str, Any]) -> float:
        """Check if claim amount is reasonable"""
        try:
            amount = float(claim_data.get('amount', 0))
            claim_type = claim_data.get('claim_type', 'auto')
            description = claim_data.get('description', '').lower()

            if amount <= 0:
                return 0.0

            # Average amounts by claim type
            avg_amounts = {
                'auto': 3500,
                'home': 8000,
                'health': 25000,
                'travel': 1500,
                'life': 50000
            }

            avg_amount = avg_amounts.get(claim_type, 5000)

            # Check for excessive amounts
            if amount > avg_amount * 5:  # More than 5x average
                return 1.0
            elif amount > avg_amount * 3:  # More than 3x average
                return 0.7
            elif amount < avg_amount * 0.1:  # Less than 10% of average
                return 0.3

            # Check amount vs. described severity
            severity_indicators = {
                'minor': ['scratch', 'dent', 'small', 'light', 'minimal'],
                'moderate': ['moderate', 'medium', 'some', 'partial'],
                'severe': ['severe', 'major', 'extensive', 'significant', 'serious'],
                'catastrophic': ['total', 'write-off', 'destroyed', 'ruined', 'lost']
            }

            detected_severity = 'minor'
            severity_score = 0

            for severity, indicators in severity_indicators.items():
                if any(indicator in description for indicator in indicators):
                    detected_severity = severity
                    break

            # Severity scoring
            severity_scores = {'minor': 0.2, 'moderate': 0.5, 'severe': 0.8, 'catastrophic': 1.0}
            severity_score = severity_scores.get(detected_severity, 0.2)

            # Calculate expected amount range based on severity
            expected_min = avg_amount * severity_score * 0.5
            expected_max = avg_amount * severity_score * 2.0

            if amount < expected_min or amount > expected_max:
                # Amount is outside expected range
                deviation = min(abs(amount - avg_amount) / avg_amount, 2.0)
                return deviation * 0.5

            return 0.0

        except:
            return 0.0

    def _check_geographic_plausibility(self, claim_data: Dict[str, Any]) -> float:
        """Check if location is geographically plausible"""
        try:
            location = claim_data.get('location', '').lower()
            claim_type = claim_data.get('claim_type', '')
            description = claim_data.get('description', '').lower()

            if not location:
                return 0.0

            inconsistency_score = 0.0

            # Check for impossible locations
            impossible_locations = ['moon', 'mars', 'space', 'underwater car', 'flying car']
            if any(impossible in location for impossible in impossible_locations):
                return 1.0

            # Check location consistency with claim type
            location_inconsistencies = {
                'auto': ['airplane', 'boat', 'train', 'submarine'],
                'home': ['car', 'vehicle', 'highway', 'road'],
                'health': ['vehicle', 'property', 'building'],
                'travel': ['home', 'workplace', 'local']
            }

            if claim_type in location_inconsistencies:
                for inconsistent in location_inconsistencies[claim_type]:
                    if inconsistent in location:
                        inconsistency_score += 0.6
                        break

            # Check for suspicious location patterns
            suspicious_patterns = [
                'intersection of main and unknown',
                '123 fake street',
                'unspecified location',
                'nowhere'
            ]

            if any(pattern in location for pattern in suspicious_patterns):
                inconsistency_score += 0.4

            return min(inconsistency_score, 1.0)

        except:
            return 0.0

    def _check_investigator_patterns(self, claim_data: Dict[str, Any]) -> float:
        """Check for suspicious investigator patterns"""
        try:
            investigator_id = claim_data.get('investigator_id', '')
            settlement_time = claim_data.get('settlement_time', '')
            claim_time = claim_data.get('claim_submitted_time', '')

            inconsistency_score = 0.0

            # Check for rapid settlement patterns
            if settlement_time and claim_time:
                try:
                    settlement_dt = self._parse_datetime(settlement_time)
                    claim_dt = self._parse_datetime(claim_time)

                    if settlement_dt and claim_dt:
                        settlement_hours = (settlement_dt - claim_dt).total_seconds() / 3600

                        if settlement_hours < 24:  # Settlement in less than 24 hours
                            inconsistency_score += 0.7
                        elif settlement_hours < 72:  # Settlement in less than 3 days
                            inconsistency_score += 0.4
                except:
                    pass

            # Check for high-volume investigators (simplified)
            if investigator_id:
                # Use hash to simulate investigator workload
                investigator_hash = hash(investigator_id) % 100
                if investigator_hash < 10:  # High volume investigator
                    inconsistency_score += 0.3

            # Check documentation completeness
            documentation = claim_data.get('documentation', {})
            if isinstance(documentation, dict):
                doc_completeness = len(documentation) / 10.0  # Normalize
                if doc_completeness < 0.3:  # Poor documentation
                    inconsistency_score += 0.2

            return min(inconsistency_score, 1.0)

        except:
            return 0.0

    def _check_witness_consistency(self, claim_data: Dict[str, Any]) -> float:
        """Check witness statement consistency"""
        try:
            witnesses = claim_data.get('witnesses', [])
            witness_count = len(witnesses) if isinstance(witnesses, list) else 0

            if witness_count == 0:
                # No witnesses - check if this is suspicious
                claim_type = claim_data.get('claim_type', '')
                claim_amount = float(claim_data.get('amount', 0))

                # High value claims with no witnesses are suspicious
                if claim_amount > 10000:
                    if claim_type in ['auto', 'home']:
                        return 0.6
                    elif claim_type == 'health':
                        return 0.2  # Less suspicious for health claims

                return 0.0

            elif witness_count == 1:
                # Single witness - check for reliability
                witness = witnesses[0] if witnesses else {}
                witness_type = witness.get('type', '').lower()

                # Immediate family member as single witness is suspicious
                if witness_type in ['family', 'relative', 'spouse', 'parent', 'child']:
                    return 0.4
                # Friend as single witness is moderately suspicious
                elif witness_type in ['friend', 'acquaintance']:
                    return 0.2

            else:
                # Multiple witnesses - check for consistency
                if witness_count > 10:  # Unusually high number of witnesses
                    return 0.3

                # Simplified consistency check
                testimony_consistency = self._calculate_witness_consistency(witnesses)
                if testimony_consistency < 0.5:
                    return 0.6

            return 0.0

        except:
            return 0.0

    def _calculate_witness_consistency(self, witnesses: List[Dict]) -> float:
        """Calculate consistency between multiple witness statements"""
        try:
            if not witnesses or len(witnesses) < 2:
                return 1.0

            # Extract key information from witness statements
            witness_statements = []
            for witness in witnesses:
                statement = witness.get('statement', '').lower()
                witness_statements.append(statement)

            # Check for major inconsistencies
            consistency_score = 1.0
            total_checks = 0
            passed_checks = 0

            # Check for common elements
            common_elements = set()
            for statement in witness_statements:
                # Simple word-based consistency check
                words = set(statement.split())
                if not common_elements:
                    common_elements = words
                else:
                    common_elements &= words

                total_checks += 1

            # High overlap suggests consistency
            if common_elements:
                avg_overlap = len(common_elements) / len(witness_statements[0].split())
                if avg_overlap > 0.3:  # 30% overlap threshold
                    passed_checks += 1

            return passed_checks / total_checks if total_checks > 0 else 1.0

        except:
            return 0.5

    def _count_damage_mentions(self, text: str) -> int:
        """Count damage-related keywords in text"""
        damage_keywords = [
            'damage', 'damaged', 'dent', 'scratch', 'broken', 'crack', 'smash',
            'hit', 'collision', 'accident', 'crashed', 'wreck', 'destroyed',
            'injured', 'injury', 'hurt', 'pain', 'medical'
        ]

        return sum(1 for keyword in damage_keywords if keyword in text)

    def _estimate_text_severity(self, text: str) -> float:
        """Estimate damage severity from text description"""
        severity_keywords = {
            'catastrophic': ['total', 'write-off', 'destroyed', 'ruined', 'lost', 'completely'],
            'severe': ['severe', 'major', 'extensive', 'significant', 'serious', 'critical'],
            'moderate': ['moderate', 'medium', 'some', 'partial', 'considerable'],
            'minor': ['minor', 'small', 'slight', 'minimal', 'superficial']
        }

        text_lower = text.lower()
        max_severity = 0.0

        for severity, keywords in severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                severity_scores = {
                    'catastrophic': 1.0,
                    'severe': 0.8,
                    'moderate': 0.5,
                    'minor': 0.2
                }
                max_severity = max(max_severity, severity_scores.get(severity, 0.0))

        return max_severity

    def _parse_datetime(self, datetime_str: str) -> datetime:
        """Parse various datetime formats"""
        if not datetime_str:
            return None

        # Common formats
        formats = [
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%m/%d/%Y %H:%M',
            '%m/%d/%Y',
            '%d/%m/%Y %H:%M',
            '%d/%m/%Y'
        ]

        for fmt in formats:
            try:
                return datetime.strptime(datetime_str.split('.')[0], fmt)
            except ValueError:
                continue

        return None

    def _calculate_risk_level(self, inconsistency_score: float) -> str:
        """Calculate risk level based on inconsistency score"""
        if inconsistency_score >= self.severity_thresholds['high']:
            return 'critical'
        elif inconsistency_score >= self.severity_thresholds['medium']:
            return 'high'
        elif inconsistency_score >= self.severity_thresholds['low']:
            return 'medium'
        else:
            return 'low'

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information for monitoring"""
        import sys

        # Estimate memory usage of the detector
        rules_size = sys.getsizeof(self.inconsistency_rules) / (1024 * 1024)
        total_memory = rules_size + 1  # Add small overhead

        return {
            'memory_usage_mb': total_memory,
            'rules_loaded': len(self.inconsistency_rules),
            'memory_efficiency': 'very_low' if total_memory < 1 else 'low'
        }

# Global instance
_inconsistency_detector = None

def get_inconsistency_detector() -> InconsistencyDetector:
    """Get or create singleton instance"""
    global _inconsistency_detector
    if _inconsistency_detector is None:
        _inconsistency_detector = InconsistencyDetector()
    return _inconsistency_detector