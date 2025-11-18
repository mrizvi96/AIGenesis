"""
Semi-Auto Feature Engineering (SAFE) for Insurance Fraud Detection
Lightweight implementation optimized for Qdrant free tier constraints
"""

import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
import math
import re
import json

class SemiAutoFeatureEngineering:
    """Generate automated risk features while maintaining memory efficiency"""

    def __init__(self):
        """Initialize SAFE with minimal memory footprint"""
        print("[ENHANCED] Loading SAFE feature engineering...")
        self.feature_cache = {}
        self.historical_data = self._load_historical_data()
        self.risk_profiles = self._load_risk_profiles()

    def _load_historical_data(self) -> Dict[str, Any]:
        """Load or initialize historical data for comparison"""
        # In production, this would load from database
        # For demo, we use reasonable defaults
        return {
            'avg_amounts': {
                'auto': 3500,
                'home': 8000,
                'health': 25000,
                'travel': 1500,
                'life': 50000
            },
            'location_risks': {
                'high_risk': ['downtown', 'airport', 'shopping_center'],
                'medium_risk': ['residential', 'suburban'],
                'low_risk': ['rural', 'parking_lot']
            },
            'time_risks': {
                'high_risk_hours': [22, 23, 0, 1, 2, 3, 4, 5],  # Late night
                'medium_risk_hours': [6, 7, 8, 17, 18, 19, 20, 21],
                'low_risk_hours': [9, 10, 11, 12, 13, 14, 15, 16]  # Business hours
            }
        }

    def _load_risk_profiles(self) -> Dict[str, Any]:
        """Load predefined risk assessment profiles"""
        return {
            'high_amount_multiplier': 3.0,
            'medium_amount_multiplier': 1.5,
            'frequent_claimer_threshold': 3,  # claims per year
            'new_policy_risk_months': 6,  # months for new policy
            'suspicious_age_range': (18, 25)  # young driver risk
        }

    def generate_risk_factors(self, claim_data: Dict[str, Any]) -> List[float]:
        """Generate all automated risk features efficiently"""
        features = []

        try:
            # Temporal Features (8 features)
            features.extend(self._extract_temporal_features(claim_data))

            # Amount-based Features (6 features)
            features.extend(self._extract_amount_features(claim_data))

            # Frequency Features (6 features)
            features.extend(self._extract_frequency_features(claim_data))

            # Geographic Features (4 features)
            features.extend(self._extract_geographic_features(claim_data))

            # Policy Features (4 features)
            features.extend(self._extract_policy_features(claim_data))

            # Claimant Profile Features (3 features)
            features.extend(self._extract_claimant_features(claim_data))

            # Cross-modal Consistency Features (2 features)
            features.extend(self._extract_consistency_features(claim_data))

        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            # Return default features
            return [0.0] * 33  # Total expected features

        return features

    def _extract_temporal_features(self, claim_data: Dict[str, Any]) -> List[float]:
        """Extract time-based risk factors (8 features)"""
        features = []

        try:
            # Time of day risk (4 features)
            claim_time = claim_data.get('accident_time', claim_data.get('time_of_day', ''))
            if claim_time and ':' in claim_time:
                hour = int(claim_time.split(':')[0])

                # Risk by time of day
                if hour in self.historical_data['time_risks']['high_risk_hours']:
                    features.extend([1.0, 0.0, 0.0])  # High risk
                elif hour in self.historical_data['time_risks']['medium_risk_hours']:
                    features.extend([0.0, 1.0, 0.0])  # Medium risk
                else:
                    features.extend([0.0, 0.0, 1.0])  # Low risk

                # Normalized hour (24-hour scale)
                features.append(hour / 24.0)
            else:
                features.extend([0.0, 0.0, 1.0, 0.5])  # Default to low risk, midday

            # Day of week risk (4 features)
            claim_date = claim_data.get('accident_date', claim_data.get('date_submitted', ''))
            if claim_date:
                try:
                    # Handle various date formats
                    if '-' in claim_date:
                        date_obj = datetime.strptime(claim_date.split()[0], '%Y-%m-%d')
                    else:
                        # Try common formats
                        date_obj = datetime.strptime(claim_date, '%m/%d/%Y')

                    day_of_week = date_obj.weekday()

                    # Weekend risk (Fri-Sun)
                    weekend_risk = 1.0 if day_of_week >= 5 else 0.0
                    features.append(weekend_risk)

                    # Normalized day (0-1)
                    features.append(day_of_week / 7.0)

                    # Holiday risk (simplified)
                    holiday_risk = self._is_holiday_risk(date_obj)
                    features.append(holiday_risk)

                    # Season risk (winter = higher risk)
                    month = date_obj.month
                    winter_risk = 1.0 if month in [12, 1, 2] else 0.0
                    features.append(winter_risk)
                except:
                    features.extend([0.0, 0.5, 0.0, 0.0])  # Default values
            else:
                features.extend([0.0, 0.5, 0.0, 0.0])  # Default values

        except:
            features.extend([0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.0, 0.0])  # All defaults

        return features

    def _extract_amount_features(self, claim_data: Dict[str, Any]) -> List[float]:
        """Extract amount-based risk factors (6 features)"""
        features = []

        try:
            claim_amount = float(claim_data.get('amount', 0))

            if claim_amount > 0:
                # Log transformation (helps with skewed distributions)
                log_amount = math.log1p(claim_amount)
                normalized_log = min(log_amount / 15.0, 1.0)  # Normalize to 0-1
                features.append(normalized_log)

                # Deviation from average for claim type
                claim_type = claim_data.get('claim_type', 'auto')
                avg_amount = self.historical_data['avg_amounts'].get(claim_type, 5000)

                if avg_amount > 0:
                    deviation = (claim_amount - avg_amount) / avg_amount
                    normalized_deviation = max(-2.0, min(2.0, deviation)) / 2.0  # Normalize to -1 to 1
                    features.append(normalized_deviation)
                else:
                    features.append(0.0)

                # High amount flags (3 features)
                high_multiplier = self.risk_profiles['high_amount_multiplier']
                medium_multiplier = self.risk_profiles['medium_amount_multiplier']

                very_high_flag = 1.0 if claim_amount > avg_amount * high_multiplier else 0.0
                high_flag = 1.0 if claim_amount > avg_amount * medium_multiplier else 0.0
                round_amount_flag = 1.0 if claim_amount % 1000 == 0 else 0.0  # Round numbers often suspicious

                features.extend([very_high_flag, high_flag, round_amount_flag])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        except:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # All defaults

        return features

    def _extract_frequency_features(self, claim_data: Dict[str, Any]) -> List[float]:
        """Extract frequency-based risk factors (6 features)"""
        features = []

        try:
            customer_id = claim_data.get('customer_id', '')

            if customer_id:
                # Simulated claim history (in production, query database)
                customer_id_hash = hash(customer_id) % 1000  # Simple hash for demo

                # Recent claims (last 12 months)
                recent_claims = (customer_id_hash % 10) / 10.0  # Simulated 0-1
                features.append(min(recent_claims, 1.0))

                # Fraud history
                fraud_claims = (customer_id_hash % 5) / 5.0  # Simulated 0-1
                features.append(min(fraud_claims, 1.0))

                # Time since last claim
                days_since_last = (customer_id_hash % 365)
                recent_claim_risk = max(0.0, (365 - days_since_last) / 365.0)
                features.append(recent_claim_risk)

                # Claim frequency rate (claims per year)
                claim_frequency = recent_claims / 1.0  # Per year
                normalized_frequency = min(claim_frequency / 5.0, 1.0)  # Normalize
                features.append(normalized_frequency)

                # Pattern regularity (check for suspicious timing)
                pattern_risk = self._calculate_pattern_risk(customer_id_hash)
                features.append(pattern_risk)

                # Policy tenure risk
                tenure_days = (customer_id_hash % 1825)  # 0-5 years
                short_tenure_risk = 1.0 if tenure_days < 180 else 0.0  # Less than 6 months
                features.append(short_tenure_risk)
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # No customer data

        except:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # All defaults

        return features

    def _extract_geographic_features(self, claim_data: Dict[str, Any]) -> List[float]:
        """Extract geographic risk factors (4 features)"""
        features = []

        try:
            location = claim_data.get('location', '').lower()

            if location:
                # Location risk assessment
                location_lower = location.lower()
                risk_score = 0.0

                # Check against predefined risk areas
                for area_type in ['high_risk', 'medium_risk', 'low_risk']:
                    risk_areas = self.historical_data['location_risks'].get(area_type, [])
                    if any(risk_area in location_lower for risk_area in risk_areas):
                        if area_type == 'high_risk':
                            risk_score = 1.0
                        elif area_type == 'medium_risk':
                            risk_score = 0.5
                        else:
                            risk_score = 0.2
                        break

                features.append(risk_score)

                # Distance from home (simplified check)
                far_from_home = 1.0 if 'away' in location_lower or 'traveling' in location_lower else 0.0
                features.append(far_from_home)

                # International flag
                international = 1.0 if any(country in location_lower for country in
                                 ['abroad', 'overseas', 'international', 'foreign']) else 0.0
                features.append(international)

                # Urban vs rural (simplified)
                urban_indicators = ['city', 'downtown', 'urban', 'metropolitan']
                rural_indicators = ['rural', 'country', 'farm', 'countryside']

                if any(indicator in location_lower for indicator in urban_indicators):
                    features.append(1.0)  # Urban
                elif any(indicator in location_lower for indicator in rural_indicators):
                    features.append(0.0)  # Rural
                else:
                    features.append(0.5)  # Unknown
            else:
                features.extend([0.0, 0.0, 0.0, 0.5])  # Default to medium risk

        except:
            features.extend([0.0, 0.0, 0.0, 0.5])  # Defaults

        return features

    def _extract_policy_features(self, claim_data: Dict[str, Any]) -> List[float]:
        """Extract policy-based risk factors (4 features)"""
        features = []

        try:
            # Policy number analysis
            policy_number = claim_data.get('policy_number', '')
            if policy_number:
                # Policy length (unusually long might be suspicious)
                policy_length = len(policy_number)
                unusual_length = 1.0 if policy_length > 15 else 0.0
                features.append(unusual_length)

                # Policy age (simplified)
                policy_date_str = claim_data.get('policy_start_date', '')
                if policy_date_str:
                    try:
                        policy_date = datetime.strptime(policy_date_str.split()[0], '%Y-%m-%d')
                        days_old = (datetime.now() - policy_date).days
                        new_policy_risk = 1.0 if days_old < self.risk_profiles['new_policy_risk_months'] * 30 else 0.0
                        features.append(new_policy_risk)

                        # Coverage adequacy (simplified)
                        coverage_adequate = 1.0 if days_old > 365 else 0.5
                        features.append(coverage_adequate)
                    except:
                        features.extend([0.0, 0.5])  # Defaults
                else:
                    features.extend([0.0, 0.5])  # No policy date

                # Policy type risk (simplified categorization)
                policy_type_risk = 0.3  # Default medium risk
                features.append(policy_type_risk)
            else:
                features.extend([0.0, 0.0, 0.5, 0.3])  # No policy number

        except:
            features.extend([0.0, 0.0, 0.5, 0.3])  # Defaults

        return features

    def _extract_claimant_features(self, claim_data: Dict[str, Any]) -> List[float]:
        """Extract claimant profile risk factors (3 features)"""
        features = []

        try:
            # Claimant age risk
            age = claim_data.get('claimant_age', 30)
            age_range = self.risk_profiles['suspicious_age_range']
            young_driver_risk = 1.0 if age_range[0] <= age <= age_range[1] else 0.0
            features.append(young_driver_risk)

            # Description length (very short or very long can be suspicious)
            description = claim_data.get('description', '')
            desc_length = len(description) if description else 0

            if desc_length < 50:
                desc_risk = 1.0  # Too short
            elif desc_length > 2000:
                desc_risk = 0.8  # Too long
            else:
                desc_risk = 0.0  # Normal
            features.append(desc_risk)

            # Detail level (simplified word count analysis)
            word_count = len(description.split()) if description else 0
            detail_score = min(word_count / 100.0, 1.0)  # Normalize to 0-1
            features.append(detail_score)

        except:
            features.extend([0.0, 0.0, 0.0])  # Defaults

        return features

    def _extract_consistency_features(self, claim_data: Dict[str, Any]) -> List[float]:
        """Extract cross-modal consistency features (2 features)"""
        features = []

        try:
            # Text vs amount consistency
            description = claim_data.get('description', '').lower()
            amount = float(claim_data.get('amount', 0))

            # Check for inconsistency between description severity and amount
            severe_keywords = ['severe', 'critical', 'emergency', 'life-threatening', 'major']
            minor_keywords = ['minor', 'small', 'slight', 'scratch', 'dent']

            severity_score = 0
            if any(keyword in description for keyword in severe_keywords):
                severity_score = 1.0
            elif any(keyword in description for keyword in minor_keywords):
                severity_score = 0.2

            # Amount severity (normalized)
            claim_type = claim_data.get('claim_type', 'auto')
            avg_amount = self.historical_data['avg_amounts'].get(claim_type, 5000)
            amount_severity = min(amount / (avg_amount * 2), 1.0)

            # Consistency score (0 = consistent, 1 = inconsistent)
            consistency_diff = abs(severity_score - amount_severity)
            features.append(consistency_diff)

            # Time reporting consistency
            accident_time = claim_data.get('accident_time', '')
            reported_time = claim_data.get('time_reported', '')

            if accident_time and reported_time:
                # Simplified time difference check
                features.append(0.0)  # Assume consistent for now
            else:
                features.append(0.1)  # Small risk for missing time data

        except:
            features.extend([0.0, 0.0])  # Defaults

        return features

    def _is_holiday_risk(self, date_obj: datetime) -> float:
        """Check if date falls on or near holidays (simplified)"""
        # Simplified holiday risk for major US holidays
        month, day = date_obj.month, date_obj.day

        holidays = [
            (1, 1),   # New Year
            (7, 4),   # Independence Day
            (12, 25), # Christmas
        ]

        for holiday_month, holiday_day in holidays:
            if month == holiday_month and abs(day - holiday_day) <= 2:
                return 1.0

        return 0.0

    def _calculate_pattern_risk(self, customer_hash: int) -> float:
        """Calculate pattern regularity risk from customer hash"""
        # Simple pattern detection using customer hash
        # In production, this would analyze actual claim timestamps
        risk_indicators = 0

        # Check for regular intervals (simplified)
        if customer_hash % 7 == 0:  # Patterns every 7 days
            risk_indicators += 0.3

        if customer_hash % 30 == 0:  # Patterns every 30 days
            risk_indicators += 0.5

        if customer_hash % 90 == 0:  # Patterns every 90 days
            risk_indicators += 0.2

        return min(risk_indicators, 1.0)

    def get_feature_importance(self) -> Dict[str, List[str]]:
        """Get feature importance information for monitoring"""
        return {
            'temporal_features': [
                'time_risk_category', 'hour_normalized',
                'weekend_risk', 'day_normalized', 'holiday_risk', 'winter_risk'
            ],
            'amount_features': [
                'log_amount', 'deviation_from_average',
                'very_high_amount', 'high_amount', 'round_amount'
            ],
            'frequency_features': [
                'recent_claims_rate', 'fraud_history',
                'time_since_last_claim', 'claim_frequency', 'pattern_risk'
            ],
            'geographic_features': [
                'location_risk', 'far_from_home',
                'international_flag', 'urban_rural'
            ]
        }

# Global instance for reuse
_safe_features_instance = None

def get_safe_features() -> SemiAutoFeatureEngineering:
    """Get or create singleton instance"""
    global _safe_features_instance
    if _safe_features_instance is None:
        _safe_features_instance = SemiAutoFeatureEngineering()
    return _safe_features_instance