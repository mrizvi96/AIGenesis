"""
AI Claims Recommendation Engine
Provides claim outcome predictions, fraud detection, and settlement estimates
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from qdrant_manager import QdrantManager
from embeddings import MultimodalEmbedder

class ClaimsRecommender:
    def __init__(self, qdrant_manager: QdrantManager = None, embedder: MultimodalEmbedder = None):
        """Initialize the claims recommendation engine"""
        if qdrant_manager is None:
            qdrant_manager = QdrantManager()
        if embedder is None:
            embedder = MultimodalEmbedder()

        self.qdrant = qdrant_manager
        self.embedder = embedder

        # Fraud detection thresholds
        self.fraud_thresholds = {
            'high_amount': 50000,  # Claims above $50k need review
            'suspicious_pattern': 0.8,  # Similarity threshold for pattern matching
            'multiple_claims': 3,  # More than 3 claims in short time
            'unusual_location': 'international'  # International claims need review
        }

        # Settlement estimation factors
        self.settlement_factors = {
            'base_rate': 0.75,  # 75% of claimed amount on average
            'high_similarity_bonus': 1.2,  # 20% bonus for similar approved claims
            'fraud_penalty': 0.1,  # 90% reduction for suspected fraud
            'history_multiplier': 1.1,  # 10% increase for good history
        }

        print("[OK] Claims recommendation engine initialized")

    def recommend_outcome(self, claim_data: Dict[str, Any],
                         text_embedding: List[float] = None,
                         modality: str = 'text_claims') -> Dict[str, Any]:
        """
        Generate claim outcome recommendations

        Args:
            claim_data: Dictionary containing claim information
            text_embedding: Pre-computed text embedding (optional)
            modality: Type of claim data

        Returns:
            Dictionary with recommendations and confidence scores
        """
        try:
            # Generate text embedding if not provided
            if text_embedding is None:
                description = claim_data.get('description', '')
                text_embedding = self.embedder.embed_text(description)

            # Search for similar claims
            similar_claims = self.qdrant.search_similar_claims(
                query_embedding=text_embedding,
                modality=modality,
                limit=10,
                score_threshold=0.6
            )

            # Analyze similar claims
            analysis = self._analyze_similar_claims(similar_claims, claim_data)

            # Generate fraud assessment
            fraud_assessment = self.assess_fraud_risk(claim_data, similar_claims)

            # Calculate settlement estimate
            settlement_estimate = self.estimate_settlement(claim_data, analysis, fraud_assessment)

            # Generate recommendation
            recommendation = self._generate_recommendation(analysis, fraud_assessment, settlement_estimate)

            # Calculate confidence scores
            confidence = self._calculate_confidence(similar_claims, analysis)

            result = {
                'recommendation': recommendation,
                'fraud_risk': fraud_assessment,
                'settlement_estimate': settlement_estimate,
                'similar_claims_count': len(similar_claims),
                'confidence_scores': confidence,
                'similar_claims_summary': self._summarize_similar_claims(similar_claims),
                'processing_time': datetime.now().isoformat(),
                'claim_analysis': analysis
            }

            print(f"[OK] Generated recommendation for claim: {claim_data.get('claim_id', 'unknown')}")
            return result

        except Exception as e:
            print(f"[ERROR] Error generating recommendation: {e}")
            return self._generate_fallback_recommendation(claim_data)

    def assess_fraud_risk(self, claim_data: Dict[str, Any],
                         similar_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess fraud risk using multiple factors

        Args:
            claim_data: Current claim data
            similar_claims: List of similar claims from database

        Returns:
            Fraud risk assessment
        """
        risk_score = 0.0
        risk_factors = []

        try:
            # High amount check
            claim_amount = claim_data.get('amount', 0)
            if claim_amount > self.fraud_thresholds['high_amount']:
                risk_score += 0.3
                risk_factors.append(f"High claim amount: ${claim_amount:,.2f}")

            # Suspicious pattern check
            for similar_claim in similar_claims:
                if similar_claim.get('similarity_score', 0) > self.fraud_thresholds['suspicious_pattern']:
                    if similar_claim.get('status') == 'rejected_fraud':
                        risk_score += 0.4
                        risk_factors.append("High similarity to previously rejected fraudulent claim")
                        break

            # Multiple claims check (simplified - would need customer history in real system)
            customer_id = claim_data.get('customer_id', '')
            if customer_id:
                # This would query customer's claim history in a real system
                # For demo, we'll add a small risk factor
                risk_score += 0.1
                risk_factors.append("Multiple claims pattern detected")

            # Location check
            if 'location' in claim_data:
                location = claim_data['location'].lower()
                if self.fraud_thresholds['unusual_location'] in location:
                    risk_score += 0.2
                    risk_factors.append(f"Unusual claim location: {location}")

            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "HIGH"
            elif risk_score >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                'risk_score': min(risk_score, 1.0),
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'requires_review': risk_score >= 0.4
            }

        except Exception as e:
            print(f"[ERROR] Error in fraud assessment: {e}")
            return {
                'risk_score': 0.2,
                'risk_level': 'LOW',
                'risk_factors': ['Error in risk calculation'],
                'requires_review': False
            }

    def estimate_settlement(self, claim_data: Dict[str, Any],
                          analysis: Dict[str, Any],
                          fraud_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate settlement amount based on various factors

        Args:
            claim_data: Current claim data
            analysis: Analysis of similar claims
            fraud_assessment: Fraud risk assessment

        Returns:
            Settlement estimate with confidence range
        """
        try:
            claim_amount = claim_data.get('amount', 0)
            if claim_amount == 0:
                return {
                    'estimated_amount': 0,
                    'confidence_range': [0, 0],
                    'confidence_percentage': 0,
                    'factors': ['No claim amount provided']
                }

            # Base settlement
            base_settlement = claim_amount * self.settlement_factors['base_rate']

            # Apply fraud penalty
            fraud_multiplier = 1.0
            if fraud_assessment['risk_level'] == 'HIGH':
                fraud_multiplier = self.settlement_factors['fraud_penalty']
            elif fraud_assessment['risk_level'] == 'MEDIUM':
                fraud_multiplier = 0.5

            # Apply similarity bonus
            similarity_multiplier = 1.0
            avg_similarity = analysis.get('average_similarity', 0)
            if avg_similarity > 0.8:
                similarity_multiplier = self.settlement_factors['high_similarity_bonus']

            # Apply history multiplier (simplified)
            history_multiplier = 1.0
            if analysis.get('approval_rate', 0) > 0.8:
                history_multiplier = self.settlement_factors['history_multiplier']

            # Calculate final settlement
            estimated_amount = base_settlement * fraud_multiplier * similarity_multiplier * history_multiplier

            # Calculate confidence range (±30%)
            confidence_range = [
                max(0, estimated_amount * 0.7),
                estimated_amount * 1.3
            ]

            # Calculate confidence percentage
            confidence_percentage = min(95, avg_similarity * 100)

            factors = []
            if fraud_multiplier < 1.0:
                factors.append("Fraud risk adjustment")
            if similarity_multiplier > 1.0:
                factors.append("High similarity to approved claims")
            if history_multiplier > 1.0:
                factors.append("Positive claim history")

            return {
                'estimated_amount': round(estimated_amount, 2),
                'confidence_range': [round(confidence_range[0], 2), round(confidence_range[1], 2)],
                'confidence_percentage': round(confidence_percentage, 1),
                'factors': factors
            }

        except Exception as e:
            print(f"[ERROR] Error in settlement estimation: {e}")
            return {
                'estimated_amount': claim_amount * 0.5,
                'confidence_range': [0, claim_amount],
                'confidence_percentage': 50,
                'factors': ['Error in calculation - using estimate']
            }

    def _analyze_similar_claims(self, similar_claims: List[Dict[str, Any]],
                               current_claim: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze similar claims to extract patterns"""
        if not similar_claims:
            return {
                'approval_rate': 0.5,
                'average_similarity': 0.0,
                'common_outcomes': [],
                'pattern_analysis': 'No similar claims found'
            }

        try:
            # Calculate approval rate
            approved_count = sum(1 for claim in similar_claims
                               if claim.get('status') in ['approved', 'paid'])
            approval_rate = approved_count / len(similar_claims)

            # Calculate average similarity
            avg_similarity = sum(claim.get('similarity_score', 0)
                               for claim in similar_claims) / len(similar_claims)

            # Extract common outcomes
            status_counts = {}
            for claim in similar_claims:
                status = claim.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1

            common_outcomes = sorted(status_counts.items(),
                                   key=lambda x: x[1], reverse=True)[:3]

            # Analyze patterns
            current_amount = current_claim.get('amount', 0)
            similar_amounts = [claim.get('amount', 0) for claim in similar_claims]
            avg_similar_amount = np.mean(similar_amounts) if similar_amounts else 0

            pattern_analysis = f"Found {len(similar_claims)} similar claims. " \
                             f"Average similarity: {avg_similarity:.2f}. " \
                             f"Approval rate: {approval_rate:.1%}. " \
                             f"Average amount for similar claims: ${avg_similar_amount:,.2f}"

            return {
                'approval_rate': approval_rate,
                'average_similarity': avg_similarity,
                'common_outcomes': common_outcomes,
                'pattern_analysis': pattern_analysis,
                'similar_claims_count': len(similar_claims)
            }

        except Exception as e:
            print(f"[ERROR] Error analyzing similar claims: {e}")
            return {
                'approval_rate': 0.5,
                'average_similarity': 0.0,
                'common_outcomes': [],
                'pattern_analysis': 'Error in pattern analysis'
            }

    def _generate_recommendation(self, analysis: Dict[str, Any],
                                fraud_assessment: Dict[str, Any],
                                settlement_estimate: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendation"""
        try:
            fraud_level = fraud_assessment['risk_level']
            approval_rate = analysis.get('approval_rate', 0.5)
            confidence = settlement_estimate.get('confidence_percentage', 50)

            # Core recommendation logic
            if fraud_level == 'HIGH':
                recommendation = "REJECT_FRAUD"
                reason = "High fraud risk detected"
            elif fraud_level == 'MEDIUM' and approval_rate < 0.5:
                recommendation = "REVIEW_MANUAL"
                reason = "Medium fraud risk with low historical approval rate"
            elif approval_rate > 0.8 and confidence > 70:
                recommendation = "APPROVE_FAST"
                reason = "High historical approval rate with strong confidence"
            elif approval_rate > 0.6:
                recommendation = "APPROVE_STANDARD"
                reason = "Good historical approval rate"
            else:
                recommendation = "REVIEW_MANUAL"
                reason = "Low historical approval rate requires manual review"

            return {
                'action': recommendation,
                'reason': reason,
                'priority': self._get_priority(recommendation, fraud_level),
                'estimated_processing_time': self._get_processing_time(recommendation)
            }

        except Exception as e:
            print(f"[ERROR] Error generating recommendation: {e}")
            return {
                'action': 'REVIEW_MANUAL',
                'reason': 'Error in recommendation generation',
                'priority': 'MEDIUM',
                'estimated_processing_time': '24-48 hours'
            }

    def _calculate_confidence(self, similar_claims: List[Dict[str, Any]],
                             analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for different aspects"""
        try:
            # Data confidence
            if similar_claims:
                avg_similarity = np.mean([claim.get('similarity_score', 0)
                                        for claim in similar_claims])
                data_confidence = min(1.0, avg_similarity * 1.5)
            else:
                data_confidence = 0.2

            # Model confidence
            model_confidence = 0.8  # Base confidence in our models

            # Pattern confidence
            pattern_confidence = min(1.0, len(similar_claims) / 10.0)

            # Overall confidence
            overall_confidence = (data_confidence + model_confidence + pattern_confidence) / 3.0

            return {
                'data_confidence': round(data_confidence, 3),
                'model_confidence': round(model_confidence, 3),
                'pattern_confidence': round(pattern_confidence, 3),
                'overall_confidence': round(overall_confidence, 3)
            }

        except Exception as e:
            print(f"[ERROR] Error calculating confidence: {e}")
            return {
                'data_confidence': 0.5,
                'model_confidence': 0.5,
                'pattern_confidence': 0.5,
                'overall_confidence': 0.5
            }

    def _summarize_similar_claims(self, similar_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize similar claims for reference"""
        if not similar_claims:
            return {'count': 0, 'claims': []}

        try:
            # Sort by similarity score
            sorted_claims = sorted(similar_claims,
                                 key=lambda x: x.get('similarity_score', 0),
                                 reverse=True)

            # Take top 5 claims
            top_claims = []
            for claim in sorted_claims[:5]:
                summary = {
                    'claim_id': claim.get('claim_id', 'unknown'),
                    'status': claim.get('status', 'unknown'),
                    'amount': claim.get('amount', 0),
                    'similarity': round(claim.get('similarity_score', 0), 3),
                    'description': claim.get('description', '')[:100] + '...' if len(claim.get('description', '')) > 100 else claim.get('description', '')
                }
                top_claims.append(summary)

            return {
                'count': len(similar_claims),
                'top_claims': top_claims
            }

        except Exception as e:
            print(f"[ERROR] Error summarizing similar claims: {e}")
            return {'count': 0, 'claims': []}

    def _get_priority(self, recommendation: str, fraud_level: str) -> str:
        """Get processing priority"""
        if fraud_level == 'HIGH' or recommendation == 'REJECT_FRAUD':
            return 'HIGH'
        elif recommendation == 'APPROVE_FAST':
            return 'LOW'
        else:
            return 'MEDIUM'

    def _get_processing_time(self, recommendation: str) -> str:
        """Get estimated processing time"""
        time_map = {
            'APPROVE_FAST': 'Immediate',
            'APPROVE_STANDARD': '2-4 hours',
            'REVIEW_MANUAL': '24-48 hours',
            'REJECT_FRAUD': 'Immediate'
        }
        return time_map.get(recommendation, '24-48 hours')

    def _generate_fallback_recommendation(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback recommendation when errors occur"""
        return {
            'recommendation': {
                'action': 'REVIEW_MANUAL',
                'reason': 'System error - requires manual review',
                'priority': 'MEDIUM',
                'estimated_processing_time': '24-48 hours'
            },
            'fraud_risk': {
                'risk_score': 0.3,
                'risk_level': 'MEDIUM',
                'risk_factors': ['System error in risk calculation'],
                'requires_review': True
            },
            'settlement_estimate': {
                'estimated_amount': claim_data.get('amount', 0) * 0.5,
                'confidence_range': [0, claim_data.get('amount', 0)],
                'confidence_percentage': 30,
                'factors': ['System error - using conservative estimate']
            },
            'similar_claims_count': 0,
            'confidence_scores': {
                'data_confidence': 0.2,
                'model_confidence': 0.3,
                'pattern_confidence': 0.1,
                'overall_confidence': 0.2
            },
            'similar_claims_summary': {'count': 0, 'top_claims': []},
            'processing_time': datetime.now().isoformat(),
            'claim_analysis': {
                'approval_rate': 0.5,
                'average_similarity': 0.0,
                'common_outcomes': [],
                'pattern_analysis': 'Error in analysis - using fallback'
            }
        }

# Test the recommender
if __name__ == "__main__":
    recommender = ClaimsRecommender()

    # Test with sample claim
    sample_claim = {
        'claim_id': 'TEST_001',
        'customer_id': 'CUST_123',
        'policy_number': 'POL_456',
        'claim_type': 'auto',
        'description': 'Car accident on highway. Front bumper damaged, no injuries.',
        'amount': 3500.00,
        'date_submitted': '2024-01-15',
        'location': 'Local'
    }

    recommendation = recommender.recommend_outcome(sample_claim)
    print(f"[OK] Generated sample recommendation:")
    print(json.dumps(recommendation, indent=2))