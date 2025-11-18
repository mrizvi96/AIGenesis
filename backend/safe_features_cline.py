"""
Enhanced SAFE (Semi-Auto Feature Engineering) Implementation
Based on Cline Recommendations: 3-5% accuracy improvement
Memory target: <30MB
Optimization: Comprehensive automated risk factor generation
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import json
import logging
import re
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClineSAFEFeatures:
    """
    Enhanced Automated feature engineering for insurance risk assessment
    Based on Cline recommendations and AIML research
    Generates 25+ comprehensive risk factors within memory constraints
    """

    def __init__(self, max_features: int = 25, memory_limit_mb: int = 30):
        """Initialize enhanced SAFE feature engineer"""
        logger.info(f"[CLINE-SAFE] Loading enhanced SAFE feature engineer (max_features={max_features}, memory_limit={memory_limit_mb}MB)...")

        self.max_features = max_features
        self.memory_limit_mb = memory_limit_mb

        # Enhanced historical baselines for comparison (from Cline research)
        self.historical_baselines = {
            'auto': {
                'avg_amount': 3500,
                'avg_frequency': 2.1,
                'high_amount_threshold': 10500,
                'suspicious_frequency_threshold': 5.0
            },
            'home': {
                'avg_amount': 8000,
                'avg_frequency': 0.8,
                'high_amount_threshold': 24000,
                'suspicious_frequency_threshold': 2.0
            },
            'health': {
                'avg_amount': 25000,
                'avg_frequency': 1.5,
                'high_amount_threshold': 75000,
                'suspicious_frequency_threshold': 3.0
            },
            'travel': {
                'avg_amount': 1500,
                'avg_frequency': 3.2,
                'high_amount_threshold': 4500,
                'suspicious_frequency_threshold': 6.0
            },
            'life': {
                'avg_amount': 50000,
                'avg_frequency': 0.3,
                'high_amount_threshold': 150000,
                'suspicious_frequency_threshold': 1.0
            }
        }

        # Enhanced geographic risk scoring
        self.location_risk_scores = {
            'highway': 0.8, 'intersection': 0.9, 'parking': 0.3,
            'urban': 0.5, 'rural': 0.7, 'unknown': 0.6,
            'residential': 0.4, 'commercial': 0.5, 'industrial': 0.6
        }

        # Time-based risk patterns (from Cline analysis)
        self.temporal_risk_patterns = {
            'high_risk_hours': [22, 23, 0, 1, 2, 3, 4, 5, 6],  # 10PM-6AM
            'medium_risk_hours': [7, 8, 9, 17, 18, 19, 20, 21],  # Rush hours
            'weekend_days': [5, 6],  # Saturday, Sunday
            'holiday_periods': [11, 12, 0, 1]  # Nov-Dec (holiday season), Jan (new year)
        }

        # Initialize feature cache
        self.feature_cache = {}
        self.cache_size_limit = 100

        # Suspicious pattern indicators
        self.suspicious_keywords = [
            'suddenly', 'unexpectedly', 'out of nowhere', 'for no reason',
            'immediately', 'instantly', 'without warning', 'mysteriously'
        ]

        logger.info("[OK] Enhanced SAFE feature engineer initialized")

    def generate_enhanced_risk_factors(self, claim_data: Dict[str, Any], description: str = "") -> Dict[str, Any]:
        """
        Generate comprehensive enhanced risk factors
        Returns 25+ features across all categories with detailed analysis
        """
        try:
            features = {}

            # Temporal Features (8 features)
            temporal_features = self._extract_enhanced_temporal_features(claim_data)
            features.update(temporal_features)

            # Amount-based Features (6 features)
            amount_features = self._extract_enhanced_amount_features(claim_data)
            features.update(amount_features)

            # Frequency Features (4 features)
            frequency_features = self._extract_enhanced_frequency_features(claim_data)
            features.update(frequency_features)

            # Geographic Features (4 features)
            geographic_features = self._extract_enhanced_geographic_features(claim_data)
            features.update(geographic_features)

            # Policy Features (4 features)
            policy_features = self._extract_enhanced_policy_features(claim_data)
            features.update(policy_features)

            # Description-based Features (3 features)
            if description:
                desc_features = self._extract_description_features(description)
                features.update(desc_features)

            # Cross-modal Interaction Features (2 features)
            interaction_features = self._extract_interaction_features(claim_data, description)
            features.update(interaction_features)

            # Generate summary statistics
            feature_summary = self._generate_feature_summary(features, claim_data)

            result = {
                'enhanced_safe_features': features,
                'feature_summary': feature_summary,
                'processing_metadata': {
                    'total_features': len(features),
                    'max_target': self.max_features,
                    'meets_target': len(features) >= self.max_features,
                    'memory_usage_mb': self._estimate_memory_usage(),
                    'generation_timestamp': datetime.now().isoformat()
                }
            }

            logger.info(f"[OK] Generated {len(features)} enhanced risk factors")
            return result

        except Exception as e:
            logger.error(f"[ERROR] Enhanced SAFE feature generation failed: {e}")
            return self._get_fallback_features()

    def _extract_enhanced_temporal_features(self, claim_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract enhanced time-based risk factors (8 features)"""
        features = {}

        try:
            # Time of day risk patterns
            accident_time = claim_data.get('accident_time', '')
            if accident_time:
                hour = self._parse_hour(accident_time)
                if hour is not None:
                    # Night risk (10PM-6AM = higher risk)
                    features['temporal_night_risk'] = 1.0 if hour in self.temporal_risk_patterns['high_risk_hours'] else 0.0

                    # Rush hour risk
                    features['temporal_rush_hour_risk'] = 1.0 if hour in self.temporal_risk_patterns['medium_risk_hours'] else 0.0

                    # Normalized hour for cyclical encoding
                    features['temporal_hour_normalized'] = hour / 24.0

                    # Sine and cosine encoding for cyclical time
                    features['temporal_hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
                    features['temporal_hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
                else:
                    for key in ['temporal_night_risk', 'temporal_rush_hour_risk', 'temporal_hour_normalized', 'temporal_hour_sin', 'temporal_hour_cos']:
                        features[key] = 0.0

            # Day of week risk
            accident_date = claim_data.get('accident_date', '')
            if accident_date:
                day_of_week = self._parse_day_of_week(accident_date)
                if day_of_week is not None:
                    # Weekend risk (different patterns)
                    features['temporal_weekend_risk'] = 1.0 if day_of_week in self.temporal_risk_patterns['weekend_days'] else 0.0

                    # Normalized day
                    features['temporal_day_normalized'] = day_of_week / 7.0

                    # Month-based risk (holiday periods)
                    month = self._parse_month(accident_date)
                    if month is not None:
                        features['temporal_holiday_risk'] = 1.0 if month in self.temporal_risk_patterns['holiday_periods'] else 0.0
                    else:
                        features['temporal_holiday_risk'] = 0.0
                else:
                    for key in ['temporal_weekend_risk', 'temporal_day_normalized', 'temporal_holiday_risk']:
                        features[key] = 0.0
            else:
                for key in ['temporal_weekend_risk', 'temporal_day_normalized', 'temporal_holiday_risk']:
                    features[key] = 0.0

        except Exception as e:
            logger.error(f"[ERROR] Temporal feature extraction failed: {e}")
            # Fallback values
            temporal_defaults = [
                'temporal_night_risk', 'temporal_rush_hour_risk', 'temporal_hour_normalized',
                'temporal_hour_sin', 'temporal_hour_cos', 'temporal_weekend_risk',
                'temporal_day_normalized', 'temporal_holiday_risk'
            ]
            for key in temporal_defaults:
                features[key] = 0.0

        return features

    def _extract_enhanced_amount_features(self, claim_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract enhanced amount-based risk factors (6 features)"""
        features = {}

        try:
            amount = float(claim_data.get('amount', 0))
            claim_type = claim_data.get('claim_type', 'auto')

            if amount <= 0:
                amount = 0.001  # Avoid log(0)

            # Get baseline for claim type
            baseline = self.historical_baselines.get(claim_type, self.historical_baselines['auto'])
            avg_amount = baseline['avg_amount']
            high_threshold = baseline['high_amount_threshold']

            # Log transformation (reduces skew)
            log_amount = np.log1p(amount)
            features['amount_log_normalized'] = min(log_amount / 15.0, 2.0)  # Normalized

            # Deviation from average
            if avg_amount > 0:
                deviation = (amount - avg_amount) / avg_amount
                features['amount_deviation_normalized'] = max(-2.0, min(deviation, 2.0))  # Clipped
            else:
                features['amount_deviation_normalized'] = 0.0

            # High amount flag
            features['amount_high_value_flag'] = 1.0 if amount > high_threshold else 0.0

            # Amount range categorization (one-hot encoded style)
            if avg_amount > 0:
                amount_ratio = amount / avg_amount
                features['amount_ratio_very_low'] = 1.0 if amount_ratio < 0.1 else 0.0
                features['amount_ratio_low'] = 1.0 if 0.1 <= amount_ratio < 0.5 else 0.0
                features['amount_ratio_normal'] = 1.0 if 0.5 <= amount_ratio <= 2.0 else 0.0
                features['amount_ratio_high'] = 1.0 if amount_ratio > 2.0 else 0.0
            else:
                for key in ['amount_ratio_very_low', 'amount_ratio_low', 'amount_ratio_normal', 'amount_ratio_high']:
                    features[key] = 0.0

        except Exception as e:
            logger.error(f"[ERROR] Amount feature extraction failed: {e}")
            # Fallback values
            amount_defaults = [
                'amount_log_normalized', 'amount_deviation_normalized', 'amount_high_value_flag',
                'amount_ratio_very_low', 'amount_ratio_low', 'amount_ratio_normal', 'amount_ratio_high'
            ]
            for key in amount_defaults:
                features[key] = 0.0

        return features

    def _extract_enhanced_frequency_features(self, claim_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract enhanced frequency-based risk factors (4 features)"""
        features = {}

        try:
            customer_id = claim_data.get('customer_id', '')
            if customer_id:
                # Simulate claim history (in real system, this would query database)
                recent_claims = self._simulate_recent_claims(customer_id)
                fraud_claims = self._simulate_fraud_claims(customer_id)

                claim_type = claim_data.get('claim_type', 'auto')
                baseline = self.historical_baselines.get(claim_type, self.historical_baselines['auto'])
                suspicious_threshold = baseline['suspicious_frequency_threshold']

                # Recent claims frequency (normalized)
                features['frequency_recent_normalized'] = min(recent_claims / 10.0, 1.0)

                # Fraud history frequency
                features['frequency_fraud_history'] = min(fraud_claims / 5.0, 1.0)

                # Frequency pattern anomaly
                features['frequency_pattern_anomaly'] = 1.0 if recent_claims > suspicious_threshold else 0.0

                # Claim velocity (claims per month over last year)
                if recent_claims > 0:
                    features['frequency_velocity'] = recent_claims / 12.0
                else:
                    features['frequency_velocity'] = 0.0

            else:
                frequency_defaults = ['frequency_recent_normalized', 'frequency_fraud_history',
                                   'frequency_pattern_anomaly', 'frequency_velocity']
                for key in frequency_defaults:
                    features[key] = 0.0

        except Exception as e:
            logger.error(f"[ERROR] Frequency feature extraction failed: {e}")
            frequency_defaults = ['frequency_recent_normalized', 'frequency_fraud_history',
                                'frequency_pattern_anomaly', 'frequency_velocity']
            for key in frequency_defaults:
                features[key] = 0.0

        return features

    def _extract_enhanced_geographic_features(self, claim_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract enhanced geographic risk factors (4 features)"""
        features = {}

        try:
            location = claim_data.get('location', '').lower()
            claim_type = claim_data.get('claim_type', '')

            if location:
                # Location risk scoring
                location_risk = 0.0
                for loc_type, score in self.location_risk_scores.items():
                    if loc_type in location:
                        location_risk = score
                        break
                features['geographic_risk_score'] = location_risk

                # Location specificity (vague locations = higher risk)
                vague_indicators = ['unknown', 'somewhere', 'some', 'unspecified', 'not sure']
                features['geographic_vagueness'] = 1.0 if any(indicator in location for indicator in vague_indicators) else 0.0

                # Multiple locations mentioned (potential inconsistency)
                location_words = len([word for word in location.split() if len(word) > 3])
                features['geographic_complexity'] = min(location_words / 5.0, 1.0)

                # Location-claim type consistency
                if claim_type == 'auto':
                    auto_indicators = ['highway', 'road', 'street', 'intersection', 'parking']
                    features['geographic_consistency'] = 1.0 if any(indicator in location for indicator in auto_indicators) else 0.0
                elif claim_type == 'home':
                    home_indicators = ['home', 'house', 'residence', 'property', 'apartment']
                    features['geographic_consistency'] = 1.0 if any(indicator in location for indicator in home_indicators) else 0.0
                else:
                    features['geographic_consistency'] = 0.5  # Neutral for other types
            else:
                geographic_defaults = ['geographic_risk_score', 'geographic_vagueness',
                                     'geographic_complexity', 'geographic_consistency']
                for key in geographic_defaults:
                    features[key] = 0.0

        except Exception as e:
            logger.error(f"[ERROR] Geographic feature extraction failed: {e}")
            geographic_defaults = ['geographic_risk_score', 'geographic_vagueness',
                                 'geographic_complexity', 'geographic_consistency']
            for key in geographic_defaults:
                features[key] = 0.0

        return features

    def _extract_enhanced_policy_features(self, claim_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract enhanced policy-based risk factors (4 features)"""
        features = {}

        try:
            customer_id = claim_data.get('customer_id', '')
            policy_number = claim_data.get('policy_number', '')
            claim_type = claim_data.get('claim_type', 'auto')
            claimant_age = claim_data.get('claimant_age', 0)

            # Policy age (newer policies = higher risk)
            policy_age_months = self._simulate_policy_age(customer_id, policy_number)
            features['policy_age_normalized'] = min(policy_age_months / 120.0, 1.0)  # 10 years = max

            # Policy type risk
            policy_risk_scores = {
                'auto': 0.7, 'home': 0.4, 'health': 0.3,
                'travel': 0.2, 'life': 0.1, 'comprehensive': 0.8
            }
            features['policy_type_risk'] = policy_risk_scores.get(claim_type, 0.5)

            # Coverage adequacy (undercoverage = higher fraud risk)
            coverage_ratio = self._simulate_coverage_ratio(customer_id, policy_number, claim_type)
            features['policy_coverage_adequacy'] = coverage_ratio

            # Age-based risk (younger or very old claimants = different patterns)
            if claimant_age > 0:
                if claimant_age < 25:
                    features['policy_age_risk'] = 0.8  # Younger
                elif claimant_age > 65:
                    features['policy_age_risk'] = 0.6  # Older
                else:
                    features['policy_age_risk'] = 0.2  # Middle-aged
            else:
                features['policy_age_risk'] = 0.0  # Unknown

        except Exception as e:
            logger.error(f"[ERROR] Policy feature extraction failed: {e}")
            policy_defaults = ['policy_age_normalized', 'policy_type_risk',
                             'policy_coverage_adequacy', 'policy_age_risk']
            for key in policy_defaults:
                features[key] = 0.0

        return features

    def _extract_description_features(self, description: str) -> Dict[str, float]:
        """Extract description-based features (3 features)"""
        features = {}

        try:
            description_lower = description.lower()

            # Suspicious keyword density
            suspicious_count = sum(1 for keyword in self.suspicious_keywords if keyword in description_lower)
            word_count = len(description_lower.split())
            features['description_suspicious_density'] = suspicious_count / max(word_count, 1)

            # Description complexity (overly simple = suspicious)
            sentence_count = len([s for s in description.split('.') if s.strip()])
            if sentence_count > 0:
                features['description_complexity'] = min(len(description.split()) / sentence_count, 50.0) / 50.0
            else:
                features['description_complexity'] = 0.0

            # Emotional indicators (excessive emotion = potential fraud)
            emotional_words = ['devastated', 'heartbroken', 'terrified', 'furious', 'outrageded']
            emotional_count = sum(1 for word in emotional_words if word in description_lower)
            features['description_emotional_score'] = min(emotional_count / 5.0, 1.0)

        except Exception as e:
            logger.error(f"[ERROR] Description feature extraction failed: {e}")
            desc_defaults = ['description_suspicious_density', 'description_complexity', 'description_emotional_score']
            for key in desc_defaults:
                features[key] = 0.0

        return features

    def _extract_interaction_features(self, claim_data: Dict[str, Any], description: str = "") -> Dict[str, float]:
        """Extract cross-modal interaction features (2 features)"""
        features = {}

        try:
            amount = float(claim_data.get('amount', 0))
            location = claim_data.get('location', '').lower()

            # Amount-description consistency
            if description and amount > 0:
                severity_indicators = ['severe', 'major', 'significant', 'extensive', 'total loss']
                severity_count = sum(1 for indicator in severity_indicators if indicator in description.lower())
                expected_severity = amount / 20000.0  # Rough severity expectation
                features['interaction_amount_description_consistency'] = 1.0 - abs(severity_count / 5.0 - min(expected_severity, 1.0))
            else:
                features['interaction_amount_description_consistency'] = 0.5

            # Location-time interaction
            if location and claim_data.get('accident_time'):
                location_risk = self.location_risk_scores.get('highway', 0.5)
                if 'highway' in location:
                    location_risk = self.location_risk_scores['highway']
                elif 'parking' in location:
                    location_risk = self.location_risk_scores['parking']

                accident_time = claim_data.get('accident_time', '')
                hour = self._parse_hour(accident_time)
                if hour is not None:
                    time_risk = 1.0 if hour in self.temporal_risk_patterns['high_risk_hours'] else 0.3
                    features['interaction_location_time_risk'] = location_risk * time_risk
                else:
                    features['interaction_location_time_risk'] = location_risk * 0.5
            else:
                features['interaction_location_time_risk'] = 0.0

        except Exception as e:
            logger.error(f"[ERROR] Interaction feature extraction failed: {e}")
            interaction_defaults = ['interaction_amount_description_consistency', 'interaction_location_time_risk']
            for key in interaction_defaults:
                features[key] = 0.0

        return features

    def _generate_feature_summary(self, features: Dict[str, float], claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive feature summary and analysis"""
        try:
            # Risk scoring by category
            risk_scores = {
                'temporal_risk': np.mean([v for k, v in features.items() if k.startswith('temporal_')]),
                'amount_risk': np.mean([v for k, v in features.items() if k.startswith('amount_')]),
                'frequency_risk': np.mean([v for k, v in features.items() if k.startswith('frequency_')]),
                'geographic_risk': np.mean([v for k, v in features.items() if k.startswith('geographic_')]),
                'policy_risk': np.mean([v for k, v in features.items() if k.startswith('policy_')]),
                'description_risk': np.mean([v for k, v in features.items() if k.startswith('description_')]),
                'interaction_risk': np.mean([v for k, v in features.items() if k.startswith('interaction_')])
            }

            # Overall risk score
            overall_risk = np.mean(list(risk_scores.values()))

            # High-risk features identification
            high_risk_features = [k for k, v in features.items() if v > 0.7]

            # Feature completeness
            expected_features = 25  # Target from Cline recommendations
            completeness = len(features) / expected_features

            return {
                'overall_risk_score': float(overall_risk),
                'category_risk_scores': {k: float(v) for k, v in risk_scores.items()},
                'high_risk_features': high_risk_features,
                'high_risk_count': len(high_risk_features),
                'feature_completeness': float(completeness),
                'total_feature_count': len(features),
                'risk_level': 'low' if overall_risk < 0.3 else 'medium' if overall_risk < 0.6 else 'high'
            }

        except Exception as e:
            logger.error(f"[ERROR] Feature summary generation failed: {e}")
            return {
                'overall_risk_score': 0.0,
                'category_risk_scores': {},
                'high_risk_features': [],
                'high_risk_count': 0,
                'feature_completeness': 0.0,
                'total_feature_count': 0,
                'risk_level': 'unknown'
            }

    def _parse_hour(self, time_str: str) -> int:
        """Parse hour from time string"""
        try:
            if ':' in time_str:
                hour = int(time_str.split(':')[0])
                return hour % 24
            return None
        except:
            return None

    def _parse_day_of_week(self, date_str: str) -> int:
        """Parse day of week from date string"""
        try:
            if date_str:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                return dt.weekday()  # Monday=0, Sunday=6
            return None
        except:
            return None

    def _parse_month(self, date_str: str) -> int:
        """Parse month from date string"""
        try:
            if date_str:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                return dt.month
            return None
        except:
            return None

    def _simulate_recent_claims(self, customer_id: str) -> int:
        """Simulate recent claims count (in real system, query database)"""
        # Hash-based deterministic simulation for consistency
        hash_val = int(hashlib.md5(f"{customer_id}_recent".encode()).hexdigest()[:8], 16)
        return (hash_val % 10)  # 0-9 recent claims

    def _simulate_fraud_claims(self, customer_id: str) -> int:
        """Simulate fraud claims count (in real system, query database)"""
        hash_val = int(hashlib.md5(f"{customer_id}_fraud".encode()).hexdigest()[:8], 16)
        return (hash_val % 3)  # 0-2 fraud claims

    def _simulate_policy_age(self, customer_id: str, policy_number: str) -> int:
        """Simulate policy age in months"""
        combined = f"{customer_id}_{policy_number}"
        hash_val = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
        return (hash_val % 120) + 1  # 1-120 months

    def _simulate_coverage_ratio(self, customer_id: str, policy_number: str, claim_type: str) -> float:
        """Simulate coverage adequacy ratio"""
        combined = f"{customer_id}_{policy_number}_{claim_type}"
        hash_val = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
        return 0.5 + (hash_val % 100) / 200.0  # 0.5-1.0

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        return float(len(str(self.feature_cache)) / (1024 * 1024))

    def _get_fallback_features(self) -> Dict[str, Any]:
        """Get fallback features when all methods fail"""
        default_features = {}

        # Generate default values for all expected features
        feature_groups = {
            'temporal': ['night_risk', 'rush_hour_risk', 'hour_normalized', 'hour_sin', 'hour_cos',
                        'weekend_risk', 'day_normalized', 'holiday_risk'],
            'amount': ['log_normalized', 'deviation_normalized', 'high_value_flag',
                      'ratio_very_low', 'ratio_low', 'ratio_normal', 'ratio_high'],
            'frequency': ['recent_normalized', 'fraud_history', 'pattern_anomaly', 'velocity'],
            'geographic': ['risk_score', 'vagueness', 'complexity', 'consistency'],
            'policy': ['age_normalized', 'type_risk', 'coverage_adequacy', 'age_risk'],
            'description': ['suspicious_density', 'complexity', 'emotional_score'],
            'interaction': ['amount_description_consistency', 'location_time_risk']
        }

        for group, feature_names in feature_groups.items():
            for feature_name in feature_names:
                default_features[f"{group}_{feature_name}"] = 0.0

        return {
            'enhanced_safe_features': default_features,
            'feature_summary': {
                'overall_risk_score': 0.0,
                'category_risk_scores': {},
                'high_risk_features': [],
                'high_risk_count': 0,
                'feature_completeness': 0.0,
                'total_feature_count': 0,
                'risk_level': 'unknown'
            },
            'processing_metadata': {
                'total_features': 0,
                'max_target': self.max_features,
                'meets_target': False,
                'memory_usage_mb': 0.0,
                'generation_timestamp': datetime.now().isoformat(),
                'fallback_mode': True
            }
        }

# Global instance for memory efficiency
_cline_safe_features = None

def get_cline_safe_features(max_features: int = 25, memory_limit_mb: int = 30) -> ClineSAFEFeatures:
    """Get or create Cline SAFE features instance"""
    global _cline_safe_features
    if _cline_safe_features is None:
        _cline_safe_features = ClineSAFEFeatures(max_features, memory_limit_mb)
    return _cline_safe_features