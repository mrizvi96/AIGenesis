#!/usr/bin/env python3
"""
Test script for the new advanced fraud analysis endpoint
"""

import requests
import json

def test_advanced_fraud_analysis():
    """Test the advanced fraud analysis endpoint"""

    # Test claim data
    test_claim = {
        "claim_data": {
            "customer_id": "CUST123456",
            "policy_number": "POL123456",
            "claim_type": "medical",
            "description": "Patient presents with severe chest pain and shortness of breath. ECG shows abnormal rhythm. Doctor recommends immediate cardiac evaluation.",
            "amount": 15234.67,
            "location": "New York, NY"
        },
        "text_data": "Patient presents with severe chest pain and shortness of breath. ECG shows abnormal rhythm. Doctor recommends immediate cardiac evaluation. Emergency room visit with cardiac monitoring and initial treatment for suspected myocardial infarction."
    }

    print("Testing Advanced Fraud Analysis Endpoint...")
    print("=" * 50)

    try:
        response = requests.post(
            "http://127.0.0.1:8000/advanced_fraud_analysis",
            json=test_claim,
            timeout=15
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Advanced fraud analysis working!")
            print("\nAnalysis Results:")

            if result.get('success') and result.get('data'):
                data = result['data']

                # Basic recommendation
                if 'recommendation' in data:
                    print(f"Recommendation: {data['recommendation'].get('recommendation', 'N/A')}")
                    print(f"Confidence: {data['recommendation'].get('confidence', 0):.2f}")

                # Multi-task classification
                if 'classification' in data:
                    print(f"\nClassification Results:")
                    classification = data['classification']
                    for key, value in classification.items():
                        if isinstance(value, dict):
                            print(f"  {key}: {list(value.keys())}")
                        else:
                            print(f"  {key}: {value}")

                # Risk factors
                if 'risk_factors' in data:
                    print(f"\nRisk Factors:")
                    risk_factors = data['risk_factors']
                    if isinstance(risk_factors, dict):
                        for factor, score in risk_factors.items():
                            print(f"  {factor}: {score}")
                    else:
                        print(f"  {risk_factors}")

                # Inconsistency detection
                if 'inconsistencies' in data:
                    print(f"\nInconsistencies Found:")
                    inconsistencies = data['inconsistencies']
                    for issue in inconsistencies:
                        print(f"  - {issue}")

                # Similar claims
                if 'similar_claims' in data:
                    similar_count = len(data['similar_claims'])
                    print(f"\nFound {similar_count} similar claims for comparison")

                # Risk score
                if 'risk_score' in data:
                    risk_score = data['risk_score']
                    print(f"\nOverall Risk Score: {risk_score:.1f}/100")

                    # Risk interpretation
                    if risk_score > 70:
                        print("   HIGH RISK - Requires immediate review")
                    elif risk_score > 40:
                        print("   MEDIUM RISK - Additional verification recommended")
                    else:
                        print("   LOW RISK - Standard processing acceptable")

            print("\n" + "=" * 50)
            print("Advanced fraud detection is fully operational!")

        else:
            print(f"Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error Details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Connection Error - Backend server not running")
    except requests.exceptions.Timeout:
        print("Timeout Error - Request took too long")
    except Exception as e:
        print(f"Unexpected Error: {e}")

def test_comparison_with_regular_endpoint():
    """Compare results with regular claim submission"""

    print("\n" + "=" * 50)
    print("COMPARISON: Regular vs Advanced Analysis")
    print("=" * 50)

    # Test claim data
    test_claim = {
        "claim_data": {
            "customer_id": "CUST123456",
            "policy_number": "POL123456",
            "claim_type": "medical",
            "description": "Patient presents with severe chest pain and shortness of breath. ECG shows abnormal rhythm. Doctor recommends immediate cardiac evaluation.",
            "amount": 15234.67,
            "location": "New York, NY"
        },
        "text_data": "Patient presents with severe chest pain and shortness of breath. ECG shows abnormal rhythm. Doctor recommends immediate cardiac evaluation."
    }

    # Test regular endpoint
    print("\nRegular Claim Submission:")
    try:
        response = requests.post(
            "http://127.0.0.1:8000/submit_claim",
            json=test_claim,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('data'):
                recommendation = result['data'].get('recommendation', {})
                print(f"  Recommendation: {recommendation.get('recommendation', 'N/A')}")
                print(f"  Confidence: {recommendation.get('confidence', 0):.2f}")
                print(f"  Processing Time: Standard")
        else:
            print(f"  Error: {response.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test advanced endpoint
    print("\nAdvanced Fraud Analysis:")
    try:
        response = requests.post(
            "http://127.0.0.1:8000/advanced_fraud_analysis",
            json=test_claim,
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('data'):
                data = result['data']
                recommendation = data.get('recommendation', {})
                print(f"  Recommendation: {recommendation.get('recommendation', 'N/A')}")
                print(f"  Confidence: {recommendation.get('confidence', 0):.2f}")
                print(f"  Risk Score: {data.get('risk_score', 'N/A')}")
                print(f"  Classification: {len(data.get('classification', {}))} categories")
                print(f"  Risk Factors: {len(data.get('risk_factors', {}))} factors")
                print(f"  Inconsistencies: {len(data.get('inconsistencies', []))} detected")
                print(f"  Processing Time: Enhanced")
        else:
            print(f"  Error: {response.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    test_advanced_fraud_analysis()
    test_comparison_with_regular_endpoint()