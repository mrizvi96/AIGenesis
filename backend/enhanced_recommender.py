"""
Enhanced AI Claims Recommendation Engine for Health Insurance
Specialized for health insurance with medical coding, provider verification, and advanced analytics
"""

import os
import json
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from qdrant_manager import QdrantManager
from enhanced_embeddings import EnhancedMultimodalEmbedder

class EnhancedClaimsRecommender:
    def __init__(self, qdrant_manager: QdrantManager = None, embedder: EnhancedMultimodalEmbedder = None):
        """Initialize the enhanced health insurance claims recommendation engine"""
        if qdrant_manager is None:
            qdrant_manager = QdrantManager()
        if embedder is None:
            embedder = EnhancedMultimodalEmbedder()

        self.qdrant = qdrant_manager
        self.embedder = embedder

        # Health insurance specific configurations
        self.medical_codes = self._load_medical_codes()
        self.provider_networks = self._load_provider_networks()
        self.policy_coverages = self._load_policy_coverages()

        # Enhanced fraud detection for health insurance
        self.health_fraud_patterns = {
            'upcoding': ['excessive billing', 'unbundling', 'duplicate charges'],
            'unnecessary_services': ['experimental treatment', 'cosmetic procedures', 'investigational'],
            'provider_fraud': ['phantom billing', 'kickbacks', 'referral fraud'],
            'patient_fraud': ['false diagnosis', 'exaggerated symptoms', 'staged accidents']
        }

        # Health insurance settlement factors
        self.health_settlement_factors = {
            'covered_services': 0.85,  # 85% for covered services
            'in_network_bonus': 1.1,   # 10% bonus for in-network
            'medical_necessity': 1.2,  # 20% for medically necessary
            'preventive_care': 1.15,    # 15% for preventive care
            'emergency_care': 1.25,     # 25% for emergency care
            'out_of_network_penalty': 0.7,  # 30% penalty for out-of-network
            'experimental_penalty': 0.3   # 70% penalty for experimental
        }

        print("[OK] Enhanced Health Insurance Claims Recommendation Engine initialized")

    def _load_medical_codes(self) -> Dict[str, Any]:
        """Load ICD-10 and CPT medical codes"""
        return {
            'icd10_diagnoses': {
                'chest_pain': ['R07.9', 'R07.8', 'R07.4'],
                'heart_attack': ['I21.9', 'I21.0', 'I21.1'],
                'diabetes': ['E11.9', 'E11.8', 'E11.0'],
                'hypertension': ['I10', 'I11.9', 'I12.9'],
                'broken_bone': ['S72.0', 'S52.5', 'S42.0'],
                'covid19': ['U07.1', 'J12.82', 'J44.0']
            },
            'cpt_procedures': {
                'ecg': ['93000', '93005', '93010'],
                'blood_test': ['36415', '36416', '82306'],
                'x_ray': ['71045', '71046', '71047'],
                'mri': ['70551', '70552', '70553'],
                'surgery': ['49000', '49120', '49180'],
                'vaccination': ['90471', '90472', '90686']
            },
            'hcpcs_supplies': {
                'crutches': ['E0110', 'E0113', 'E0115'],
                'wheelchair': ['E1030', 'E1031', 'E1032'],
                'oxygen': ['E0424', 'E0431', 'E0433'],
                'diabetic_supplies': ['A4253', 'A4252', 'A4251']
            }
        }

    def _load_provider_networks(self) -> Dict[str, Any]:
        """Load provider network information"""
        return {
            'in_network_providers': [
                'GENERAL_HOSPITAL', 'MEDICAL_CENTER', 'URGENT_CARE',
                'PRIMARY_CARE_PHYSICIAN', 'SPECIALIST_CLINIC'
            ],
            'out_of_network_alerts': [
                'INTERNATIONAL_HOSPITAL', 'EXPERIMENTAL_CLINIC',
                'ALTERNATIVE_MEDICINE', 'OUT_OF_STATE_PROVIDER'
            ],
            'high_cost_providers': [
                'SPECIALTY_HOSPITAL', 'RESEARCH_CENTER', 'UNIVERSITY_HOSPITAL'
            ]
        }

    def _load_policy_coverages(self) -> Dict[str, Any]:
        """Load insurance policy coverage rules"""
        return {
            ' preventive_care': {
                'coverage': 1.0,  # 100% covered
                'services': ['annual_physical', 'vaccination', 'screening', 'counseling']
            },
            'emergency_care': {
                'coverage': 0.9,  # 90% covered
                'services': ['er_visit', 'ambulance', 'emergency_surgery']
            },
            'specialist_care': {
                'coverage': 0.8,  # 80% covered
                'services': ['specialist_visit', 'specialist_procedure']
            },
            'hospitalization': {
                'coverage': 0.85,  # 85% covered
                'services': ['inpatient_care', 'surgery', 'medication']
            },
            'mental_health': {
                'coverage': 0.75,  # 75% covered
                'services': ['therapy', 'psychiatry', 'counseling']
            },
            'prescription_drugs': {
                'coverage': 0.7,  # 70% covered
                'services': ['medication', 'prescription']
            }
        }

    def recommend_outcome(self, claim_data: Dict[str, Any],
                         text_embedding: List[float] = None,
                         modality: str = 'text_claims') -> Dict[str, Any]:
        """
        Generate enhanced health insurance claim outcome recommendations

        Args:
            claim_data: Dictionary containing claim information
            text_embedding: Pre-computed text embedding (optional)
            modality: Type of claim data

        Returns:
            Dictionary with health insurance-specific recommendations
        """
        try:
            # Generate enhanced text embedding if not provided
            if text_embedding is None:
                description = claim_data.get('description', '')
                text_embedding = self.embedder.embed_text(description, extract_medical_entities=True)

            # Search for similar health insurance claims
            similar_claims = self.qdrant.search_similar_claims(
                query_embedding=text_embedding,
                modality=modality,
                limit=15,
                score_threshold=0.5
            )

            # Analyze similar claims with health insurance focus
            analysis = self._analyze_health_insurance_claims(similar_claims, claim_data)

            # Extract medical codes and procedures
            medical_analysis = self._extract_medical_information(claim_data)

            # Verify provider network status
            provider_analysis = self._analyze_provider_network(claim_data, medical_analysis)

            # Assess policy coverage
            coverage_analysis = self._assess_policy_coverage(claim_data, medical_analysis)

            # Generate enhanced fraud assessment for health insurance
            fraud_assessment = self._assess_health_insurance_fraud(claim_data, similar_claims, medical_analysis)

            # Calculate health insurance specific settlement estimate
            settlement_estimate = self._estimate_health_settlement(claim_data, analysis, coverage_analysis, fraud_assessment)

            # Generate health insurance specific recommendation
            recommendation = self._generate_health_recommendation(analysis, fraud_assessment, settlement_estimate, coverage_analysis)

            # Calculate enhanced confidence scores
            confidence = self._calculate_enhanced_confidence(similar_claims, analysis, medical_analysis)

            result = {
                'recommendation': recommendation,
                'fraud_risk': fraud_assessment,
                'settlement_estimate': settlement_estimate,
                'medical_analysis': medical_analysis,
                'provider_analysis': provider_analysis,
                'coverage_analysis': coverage_analysis,
                'similar_claims_count': len(similar_claims),
                'confidence_scores': confidence,
                'similar_claims_summary': self._summarize_similar_claims(similar_claims),
                'processing_time': datetime.now().isoformat(),
                'claim_analysis': analysis
            }

            print(f"[OK] Generated health insurance recommendation for claim: {claim_data.get('claim_id', 'unknown')}")
            return result

        except Exception as e:
            print(f"[ERROR] Error generating enhanced recommendation: {e}")
            return self._generate_health_fallback_recommendation(claim_data)

    def _extract_medical_information(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medical codes, procedures, and diagnoses from claim data"""
        try:
            description = claim_data.get('description', '').lower()
            claim_type = claim_data.get('claim_type', '').lower()
            
            medical_info = {
                'detected_diagnoses': [],
                'detected_procedures': [],
                'detected_supplies': [],
                'medical_keywords': [],
                'urgency_level': 'routine',
                'treatment_type': 'unknown'
            }

            # Extract ICD-10 diagnoses
            for diagnosis, codes in self.medical_codes['icd10_diagnoses'].items():
                if diagnosis.replace('_', ' ') in description:
                    medical_info['detected_diagnoses'].append({
                        'condition': diagnosis.replace('_', ' ').title(),
                        'icd10_codes': codes,
                        'confidence': self._calculate_keyword_confidence(diagnosis, description)
                    })

            # Extract CPT procedures
            for procedure, codes in self.medical_codes['cpt_procedures'].items():
                if procedure.replace('_', ' ') in description or procedure in description:
                    medical_info['detected_procedures'].append({
                        'procedure': procedure.replace('_', ' ').title(),
                        'cpt_codes': codes,
                        'confidence': self._calculate_keyword_confidence(procedure, description)
                    })

            # Extract HCPCS supplies
            for supply, codes in self.medical_codes['hcpcs_supplies'].items():
                if supply.replace('_', ' ') in description or supply in description:
                    medical_info['detected_supplies'].append({
                        'supply': supply.replace('_', ' ').title(),
                        'hcpcs_codes': codes,
                        'confidence': self._calculate_keyword_confidence(supply, description)
                    })

            # Determine urgency level
            urgency_keywords = {
                'emergency': ['emergency', 'urgent', 'critical', 'life threatening', 'severe'],
                'urgent': ['urgent care', 'immediate', 'serious', 'acute'],
                'routine': ['routine', 'scheduled', 'regular', 'checkup']
            }

            for urgency, keywords in urgency_keywords.items():
                if any(keyword in description for keyword in keywords):
                    medical_info['urgency_level'] = urgency
                    break

            # Determine treatment type
            treatment_keywords = {
                'preventive': ['preventive', 'screening', 'checkup', 'vaccination', 'physical'],
                'diagnostic': ['test', 'x-ray', 'mri', 'blood test', 'diagnosis'],
                'treatment': ['treatment', 'therapy', 'medication', 'procedure'],
                'emergency': ['emergency', 'accident', 'trauma', 'critical']
            }

            for treatment, keywords in treatment_keywords.items():
                if any(keyword in description for keyword in keywords):
                    medical_info['treatment_type'] = treatment
                    break

            # Extract medical keywords
            all_medical_terms = []
            for category in self.medical_codes.values():
                for key in category.keys():
                    all_medical_terms.extend(key.replace('_', ' ').split())
            
            unique_terms = set(all_medical_terms)
            for term in unique_terms:
                if term in description:
                    medical_info['medical_keywords'].append(term)

            return medical_info

        except Exception as e:
            print(f"[ERROR] Error extracting medical information: {e}")
            return {
                'detected_diagnoses': [],
                'detected_procedures': [],
                'detected_supplies': [],
                'medical_keywords': [],
                'urgency_level': 'routine',
                'treatment_type': 'unknown'
            }

    def _calculate_keyword_confidence(self, keyword: str, text: str) -> float:
        """Calculate confidence score for keyword detection"""
        keyword_parts = keyword.replace('_', ' ').split()
        matches = sum(1 for part in keyword_parts if part in text)
        return min(matches / len(keyword_parts), 1.0)

    def _analyze_provider_network(self, claim_data: Dict[str, Any], medical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze provider network status and appropriateness"""
        try:
            description = claim_data.get('description', '').lower()
            amount = claim_data.get('amount', 0)

            provider_info = {
                'network_status': 'unknown',
                'provider_type': 'unknown',
                'cost_appropriateness': 'appropriate',
                'geographic_appropriateness': 'appropriate',
                'specialty_match': 'good',
                'alerts': []
            }

            # Check for provider network indicators
            for provider in self.provider_networks['in_network_providers']:
                if provider.lower() in description:
                    provider_info['network_status'] = 'in_network'
                    provider_info['provider_type'] = provider.lower()
                    break

            # Check for out-of-network alerts
            for alert in self.provider_networks['out_of_network_alerts']:
                if alert.lower() in description:
                    provider_info['network_status'] = 'out_of_network'
                    provider_info['alerts'].append(f"Out-of-network provider: {alert}")
                    break

            # Check for high-cost providers
            for high_cost in self.provider_networks['high_cost_providers']:
                if high_cost.lower() in description:
                    provider_info['alerts'].append(f"High-cost provider: {high_cost}")
                    break

            # Assess cost appropriateness based on medical procedures
            if medical_analysis['detected_procedures']:
                total_procedures = len(medical_analysis['detected_procedures'])
                if amount > 50000 and total_procedures < 3:
                    provider_info['cost_appropriateness'] = 'high'
                elif amount > 20000 and total_procedures < 2:
                    provider_info['cost_appropriateness'] = 'moderate'

            # Assess geographic appropriateness
            if 'international' in description or 'foreign' in description:
                provider_info['geographic_appropriateness'] = 'international'
                provider_info['alerts'].append('International treatment detected')

            # Assess specialty match
            specialties = {
                'cardiac': ['heart', 'chest', 'cardiac', 'ecg'],
                'orthopedic': ['bone', 'fracture', 'joint', 'muscle'],
                'diagnostic': ['test', 'x-ray', 'mri', 'scan'],
                'primary_care': ['checkup', 'physical', 'routine']
            }

            for specialty, keywords in specialties.items():
                if any(keyword in description for keyword in keywords):
                    provider_info['specialty_match'] = specialty
                    break

            return provider_info

        except Exception as e:
            print(f"[ERROR] Error analyzing provider network: {e}")
            return {
                'network_status': 'unknown',
                'provider_type': 'unknown',
                'cost_appropriateness': 'unknown',
                'geographic_appropriateness': 'unknown',
                'specialty_match': 'unknown',
                'alerts': []
            }

    def _assess_policy_coverage(self, claim_data: Dict[str, Any], medical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess insurance policy coverage for the claim"""
        try:
            description = claim_data.get('description', '').lower()
            claim_amount = claim_data.get('amount', 0)
            treatment_type = medical_analysis.get('treatment_type', 'unknown')
            urgency = medical_analysis.get('urgency_level', 'routine')

            coverage_info = {
                'coverage_category': 'unknown',
                'coverage_percentage': 0.0,
                'is_covered': False,
                'requires_preauthorization': False,
                'coverage_limitations': [],
                'deductible_applicable': True,
                'copayment_expected': False
            }

            # Determine coverage category based on treatment type
            for category, coverage in self.policy_coverages.items():
                services = coverage.get('services', [])
                if any(service in description for service in services):
                    coverage_info['coverage_category'] = category
                    coverage_info['coverage_percentage'] = coverage['coverage']
                    coverage_info['is_covered'] = True
                    break

            # Special rules for urgent/emergency care
            if urgency in ['emergency', 'urgent']:
                coverage_info['coverage_category'] = 'emergency_care'
                coverage_info['coverage_percentage'] = max(coverage_info['coverage_percentage'], 0.9)

            # Check for preauthorization requirements
            high_cost_thresholds = {
                'surgery': 10000,
                'hospitalization': 5000,
                'specialist': 2000
            }

            for service_type, threshold in high_cost_thresholds.items():
                if service_type in description and claim_amount > threshold:
                    coverage_info['requires_preauthorization'] = True
                    coverage_info['coverage_limitations'].append(f"Preauthorization required for {service_type} over ${threshold}")

            # Check for experimental procedures
            experimental_keywords = ['experimental', 'investigational', 'clinical trial', 'research']
            if any(keyword in description for keyword in experimental_keywords):
                coverage_info['coverage_limitations'].append('Experimental procedures may not be covered')
                coverage_info['coverage_percentage'] *= 0.5

            # Determine deductible and copayment
            if claim_amount > 1000:  # High-cost claims typically have deductible
                coverage_info['deductible_applicable'] = True
            
            if coverage_info['coverage_category'] in ['preventive_care', 'specialist_care']:
                coverage_info['copayment_expected'] = True

            return coverage_info

        except Exception as e:
            print(f"[ERROR] Error assessing policy coverage: {e}")
            return {
                'coverage_category': 'unknown',
                'coverage_percentage': 0.0,
                'is_covered': False,
                'requires_preauthorization': False,
                'coverage_limitations': [],
                'deductible_applicable': True,
                'copayment_expected': False
            }

    def _assess_health_insurance_fraud(self, claim_data: Dict[str, Any], 
                                      similar_claims: List[Dict[str, Any]], 
                                      medical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess fraud risk specific to health insurance claims
        """
        try:
            risk_score = 0.0
            risk_factors = []
            fraud_patterns = []

            # High amount analysis
            claim_amount = claim_data.get('amount', 0)
            if claim_amount > 100000:
                risk_score += 0.3
                risk_factors.append(f"Extremely high claim amount: ${claim_amount:,.2f}")
            elif claim_amount > 50000:
                risk_score += 0.2
                risk_factors.append(f"High claim amount: ${claim_amount:,.2f}")

            # Upcoding detection
            procedures = medical_analysis.get('detected_procedures', [])
            if len(procedures) == 0 and claim_amount > 10000:
                risk_score += 0.3
                risk_factors.append("High amount with no clear procedures detected")
                fraud_patterns.append('upcoding')

            # Unnecessary services detection
            urgency = medical_analysis.get('urgency_level', 'routine')
            if urgency == 'routine' and claim_amount > 20000:
                risk_score += 0.2
                risk_factors.append("Expensive routine care")
                fraud_patterns.append('unnecessary_services')

            # Duplicate claim detection
            customer_id = claim_data.get('customer_id', '')
            similar_customer_claims = [claim for claim in similar_claims 
                                   if claim.get('customer_id') == customer_id]
            if len(similar_customer_claims) > 5:
                risk_score += 0.3
                risk_factors.append(f"High claim frequency: {len(similar_customer_claims)} similar claims")
                fraud_patterns.append('patient_fraud')

            # Provider pattern analysis
            description = claim_data.get('description', '').lower()
            for pattern_type, keywords in self.health_fraud_patterns.items():
                if any(keyword in description for keyword in keywords):
                    risk_score += 0.2
                    risk_factors.append(f"Suspicious pattern: {pattern_type}")
                    fraud_patterns.append(pattern_type)

            # Similar to previously rejected claims
            rejected_similar = [claim for claim in similar_claims 
                              if claim.get('status') in ['rejected_fraud', 'rejected_duplicate']]
            if rejected_similar:
                risk_score += 0.4
                risk_factors.append("Similar to previously rejected fraudulent claims")
                fraud_patterns.append('provider_fraud')

            # Medical necessity assessment
            diagnoses = medical_analysis.get('detected_diagnoses', [])
            procedures = medical_analysis.get('detected_procedures', [])
            
            if not diagnoses and len(procedures) > 0:
                risk_score += 0.2
                risk_factors.append("Procedures without clear diagnoses")
                fraud_patterns.append('unnecessary_services')

            # Determine risk level
            if risk_score >= 0.8:
                risk_level = "CRITICAL"
            elif risk_score >= 0.6:
                risk_level = "HIGH"
            elif risk_score >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                'risk_score': min(risk_score, 1.0),
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'detected_patterns': fraud_patterns,
                'requires_review': risk_score >= 0.4,
                'investigation_priority': 'HIGH' if risk_score >= 0.6 else 'MEDIUM'
            }

        except Exception as e:
            print(f"[ERROR] Error in health fraud assessment: {e}")
            return {
                'risk_score': 0.3,
                'risk_level': 'MEDIUM',
                'risk_factors': ['Error in risk calculation'],
                'detected_patterns': [],
                'requires_review': True,
                'investigation_priority': 'MEDIUM'
            }

    def _estimate_health_settlement(self, claim_data: Dict[str, Any], 
                                  analysis: Dict[str, Any],
                                  coverage_analysis: Dict[str, Any],
                                  fraud_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate settlement amount for health insurance claims
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

            # Base settlement using coverage percentage
            coverage_percentage = coverage_analysis.get('coverage_percentage', 0.7)
            base_settlement = claim_amount * coverage_percentage

            # Apply fraud adjustments
            fraud_multiplier = 1.0
            risk_level = fraud_assessment.get('risk_level', 'LOW')
            if risk_level == 'CRITICAL':
                fraud_multiplier = 0.1  # 90% reduction
            elif risk_level == 'HIGH':
                fraud_multiplier = 0.3  # 70% reduction
            elif risk_level == 'MEDIUM':
                fraud_multiplier = 0.6  # 40% reduction

            # Apply treatment type adjustments
            treatment_multiplier = 1.0
            coverage_category = coverage_analysis.get('coverage_category', 'unknown')
            if coverage_category == 'preventive_care':
                treatment_multiplier = self.health_settlement_factors['preventive_care']
            elif coverage_category == 'emergency_care':
                treatment_multiplier = self.health_settlement_factors['emergency_care']
            elif coverage_category == 'hospitalization':
                treatment_multiplier = self.health_settlement_factors['covered_services']

            # Apply network adjustments
            network_multiplier = 1.0
            if coverage_analysis.get('requires_preauthorization', False):
                network_multiplier *= 0.9  # 10% penalty if no preauth

            # Calculate final settlement
            estimated_amount = base_settlement * fraud_multiplier * treatment_multiplier * network_multiplier

            # Apply minimum and maximum limits
            minimum_payment = claim_amount * 0.1  # 10% minimum
            maximum_payment = claim_amount * 0.95  # 95% maximum
            estimated_amount = max(minimum_payment, min(estimated_amount, maximum_payment))

            # Calculate confidence range
            confidence_percentage = coverage_analysis.get('coverage_percentage', 0.7) * 100
            if risk_level == 'LOW':
                confidence_range = [estimated_amount * 0.8, estimated_amount * 1.1]
            elif risk_level == 'MEDIUM':
                confidence_range = [estimated_amount * 0.6, estimated_amount * 1.2]
            else:
                confidence_range = [estimated_amount * 0.3, estimated_amount * 1.3]

            factors = []
            if coverage_percentage < 1.0:
                factors.append(f"Coverage limit: {coverage_percentage*100:.0f}%")
            if fraud_multiplier < 1.0:
                factors.append("Fraud risk adjustment")
            if treatment_multiplier != 1.0:
                factors.append(f"Treatment type adjustment: {coverage_category}")
            if network_multiplier < 1.0:
                factors.append("Preauthorization requirement")

            return {
                'estimated_amount': round(estimated_amount, 2),
                'confidence_range': [round(confidence_range[0], 2), round(confidence_range[1], 2)],
                'confidence_percentage': round(confidence_percentage, 1),
                'factors': factors,
                'coverage_applied': coverage_percentage,
                'fraud_adjustment': fraud_multiplier,
                'treatment_adjustment': treatment_multiplier
            }

        except Exception as e:
            print(f"[ERROR] Error in health settlement estimation: {e}")
            return {
                'estimated_amount': claim_amount * 0.5,
                'confidence_range': [0, claim_amount],
                'confidence_percentage': 50,
                'factors': ['Error in calculation - using estimate']
            }

    def _analyze_health_insurance_claims(self, similar_claims: List[Dict[str, Any]], 
                                      current_claim: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze similar health insurance claims to extract patterns"""
        if not similar_claims:
            return {
                'approval_rate': 0.5,
                'average_similarity': 0.0,
                'common_outcomes': [],
                'pattern_analysis': 'No similar claims found',
                'health_insurance_patterns': {}
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

            # Health insurance specific patterns
            health_patterns = {
                'common_procedures': {},
                'average_claim_amount': np.mean([claim.get('amount', 0) for claim in similar_claims]),
                'high_value_claims': len([claim for claim in similar_claims if claim.get('amount', 0) > 50000]),
                'emergency_claims': len([claim for claim in similar_claims if 'emergency' in claim.get('description', '').lower()])
            }

            pattern_analysis = f"Found {len(similar_claims)} similar health insurance claims. " \
                             f"Average similarity: {avg_similarity:.2f}. " \
                             f"Approval rate: {approval_rate:.1%}. " \
                             f"Average amount: ${health_patterns['average_claim_amount']:,.2f}"

            return {
                'approval_rate': approval_rate,
                'average_similarity': avg_similarity,
                'common_outcomes': common_outcomes,
                'pattern_analysis': pattern_analysis,
                'health_insurance_patterns': health_patterns,
                'similar_claims_count': len(similar_claims)
            }

        except Exception as e:
            print(f"[ERROR] Error analyzing health insurance claims: {e}")
            return {
                'approval_rate': 0.5,
                'average_similarity': 0.0,
                'common_outcomes': [],
                'pattern_analysis': 'Error in pattern analysis',
                'health_insurance_patterns': {}
            }

    def _generate_health_recommendation(self, analysis: Dict[str, Any],
                                     fraud_assessment: Dict[str, Any],
                                     settlement_estimate: Dict[str, Any],
                                     coverage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate health insurance specific recommendation"""
        try:
            fraud_level = fraud_assessment.get('risk_level', 'LOW')
            approval_rate = analysis.get('approval_rate', 0.5)
            is_covered = coverage_analysis.get('is_covered', False)
            requires_preauth = coverage_analysis.get('requires_preauthorization', False)

            # Health insurance specific recommendation logic
            if fraud_level == 'CRITICAL':
                recommendation = "REJECT_FRAUD"
                reason = "Critical fraud risk detected - immediate rejection recommended"
            elif fraud_level == 'HIGH':
                recommendation = "INVESTIGATE_FRAUD"
                reason = "High fraud risk - comprehensive investigation required"
            elif not is_covered:
                recommendation = "REJECT_COVERAGE"
                reason = "Service not covered under policy"
            elif requires_preauth and not is_covered:
                recommendation = "REQUIRE_PREAUTH"
                reason = "Preauthorization required for this service"
            elif fraud_level == 'MEDIUM' and approval_rate < 0.5:
                recommendation = "REVIEW_MANUAL"
                reason = "Medium fraud risk with low historical approval rate"
            elif is_covered and approval_rate > 0.8 and fraud_level == 'LOW':
                recommendation = "APPROVE_FAST"
                reason = "Covered service with high approval rate and low fraud risk"
            elif is_covered and approval_rate > 0.6:
                recommendation = "APPROVE_STANDARD"
                reason = "Covered service with good historical approval rate"
            else:
                recommendation = "REVIEW_MANUAL"
                reason = "Requires manual review due to mixed indicators"

            return {
                'action': recommendation,
                'reason': reason,
                'priority': self._get_health_priority(recommendation, fraud_level, is_covered),
                'estimated_processing_time': self._get_health_processing_time(recommendation, requires_preauth),
                'next_steps': self._get_health_next_steps(recommendation, coverage_analysis)
            }

        except Exception as e:
            print(f"[ERROR] Error generating health recommendation: {e}")
            return {
                'action': 'REVIEW_MANUAL',
                'reason': 'Error in recommendation generation',
                'priority': 'MEDIUM',
                'estimated_processing_time': '24-48 hours',
                'next_steps': ['Manual review required']
            }

    def _get_health_priority(self, recommendation: str, fraud_level: str, is_covered: bool) -> str:
        """Get processing priority for health insurance claims"""
        if fraud_level == 'CRITICAL' or recommendation == 'REJECT_FRAUD':
            return 'CRITICAL'
        elif fraud_level == 'HIGH' or recommendation == 'INVESTIGATE_FRAUD':
            return 'HIGH'
        elif recommendation == 'APPROVE_FAST' and is_covered:
            return 'LOW'
        else:
            return 'MEDIUM'

    def _get_health_processing_time(self, recommendation: str, requires_preauth: bool) -> str:
        """Get estimated processing time for health insurance claims"""
        if recommendation == 'APPROVE_FAST':
            return 'Immediate'
        elif recommendation == 'REQUIRE_PREAUTH':
            return 'Preauthorization required (2-5 days)'
        elif recommendation in ['REJECT_FRAUD', 'INVESTIGATE_FRAUD']:
            return '5-10 business days'
        elif recommendation == 'APPROVE_STANDARD':
            return '2-3 business days'
        else:
            return '5-7 business days'

    def _get_health_next_steps(self, recommendation: str, coverage_analysis: Dict[str, Any]) -> List[str]:
        """Get next steps for health insurance claims processing"""
        next_steps = []

        if recommendation == 'APPROVE_FAST':
            next_steps = ['Process payment', 'Update claim status', 'Notify provider']
        elif recommendation == 'APPROVE_STANDARD':
            next_steps = ['Verify coverage', 'Process payment', 'Send explanation of benefits']
        elif recommendation == 'REQUIRE_PREAUTH':
            next_steps = ['Request preauthorization', 'Review medical necessity', 'Pending provider response']
        elif recommendation == 'INVESTIGATE_FRAUD':
            next_steps = ['Assign to fraud investigation team', 'Review medical documentation', 'Contact provider for records']
        elif recommendation == 'REJECT_COVERAGE':
            next_steps = ['Send denial letter', 'Explain coverage limitations', 'Offer appeal process']
        else:
            next_steps = ['Manual review required', 'Verify all documentation', 'Contact provider if needed']

        return next_steps

    def _calculate_enhanced_confidence(self, similar_claims: List[Dict[str, Any]],
                                     analysis: Dict[str, Any],
                                     medical_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate enhanced confidence scores for health insurance"""
        try:
            # Data confidence
            if similar_claims:
                avg_similarity = np.mean([claim.get('similarity_score', 0)
                                        for claim in similar_claims])
                data_confidence = min(1.0, avg_similarity * 1.5)
            else:
                data_confidence = 0.3

            # Medical analysis confidence
            medical_confidence = 0.0
            total_detections = (len(medical_analysis.get('detected_diagnoses', [])) +
                             len(medical_analysis.get('detected_procedures', [])) +
                             len(medical_analysis.get('detected_supplies', [])))
            if total_detections > 0:
                medical_confidence = min(1.0, total_detections / 5.0)

            # Model confidence
            model_confidence = 0.85  # High confidence in our enhanced models

            # Pattern confidence
            pattern_confidence = min(1.0, len(similar_claims) / 15.0)

            # Overall confidence
            overall_confidence = (data_confidence + medical_confidence + model_confidence + pattern_confidence) / 4.0

            return {
                'data_confidence': round(data_confidence, 3),
                'medical_confidence': round(medical_confidence, 3),
                'model_confidence': round(model_confidence, 3),
                'pattern_confidence': round(pattern_confidence, 3),
                'overall_confidence': round(overall_confidence, 3)
            }

        except Exception as e:
            print(f"[ERROR] Error calculating enhanced confidence: {e}")
            return {
                'data_confidence': 0.5,
                'medical_confidence': 0.5,
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
                    'description': claim.get('description', '')[:100] + '...' if len(claim.get('description', '')) > 100 else claim.get('description', ''),
                    'claim_type': claim.get('claim_type', 'unknown')
                }
                top_claims.append(summary)

            return {
                'count': len(similar_claims),
                'top_claims': top_claims
            }

        except Exception as e:
            print(f"[ERROR] Error summarizing similar claims: {e}")
            return {'count': 0, 'claims': []}

    def _generate_health_fallback_recommendation(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback recommendation when errors occur"""
        return {
            'recommendation': {
                'action': 'REVIEW_MANUAL',
                'reason': 'System error - requires manual review',
                'priority': 'MEDIUM',
                'estimated_processing_time': '24-48 hours',
                'next_steps': ['Manual review required']
            },
            'fraud_risk': {
                'risk_score': 0.3,
                'risk_level': 'MEDIUM',
                'risk_factors': ['System error in risk calculation'],
                'detected_patterns': [],
                'requires_review': True,
                'investigation_priority': 'MEDIUM'
            },
            'settlement_estimate': {
                'estimated_amount': claim_data.get('amount', 0) * 0.5,
                'confidence_range': [0, claim_data.get('amount', 0)],
                'confidence_percentage': 30,
                'factors': ['System error - using conservative estimate']
            },
            'medical_analysis': {
                'detected_diagnoses': [],
                'detected_procedures': [],
                'detected_supplies': [],
                'medical_keywords': [],
                'urgency_level': 'routine',
                'treatment_type': 'unknown'
            },
            'provider_analysis': {
                'network_status': 'unknown',
                'provider_type': 'unknown',
                'cost_appropriateness': 'unknown',
                'geographic_appropriateness': 'unknown',
                'specialty_match': 'unknown',
                'alerts': []
            },
            'coverage_analysis': {
                'coverage_category': 'unknown',
                'coverage_percentage': 0.5,
                'is_covered': False,
                'requires_preauthorization': False,
                'coverage_limitations': [],
                'deductible_applicable': True,
                'copayment_expected': False
            },
            'similar_claims_count': 0,
            'confidence_scores': {
                'data_confidence': 0.2,
                'medical_confidence': 0.1,
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
                'pattern_analysis': 'Error in analysis - using fallback',
                'health_insurance_patterns': {}
            }
        }

# Test the enhanced health insurance recommender
if __name__ == "__main__":
    recommender = EnhancedClaimsRecommender()

    # Test with health insurance claim
    sample_claim = {
        'claim_id': 'HEALTH_001',
        'customer_id': 'CUST_HEALTH_123',
        'policy_number': 'POL_HEALTH_456',
        'claim_type': 'health',
        'description': 'Patient presented to emergency room with severe chest pain and shortness of breath. ECG performed showing abnormal rhythm. Blood tests and chest X-ray ordered. Admitted for cardiac monitoring. Treatment included nitroglycerin and aspirin.',
        'amount': 25000.00,
        'date_submitted': '2024-01-15',
        'location': 'General Hospital'
    }

    recommendation = recommender.recommend_outcome(sample_claim)
    print(f"[OK] Generated enhanced health insurance recommendation:")
    print(json.dumps(recommendation, indent=2))
