"""
Sample Insurance Claims Data for Testing and Demo
Creates sample claims and populates Qdrant with test data
"""

import uuid
import json
import random
from datetime import datetime, timedelta
from qdrant_manager import QdrantManager
from embeddings import MultimodalEmbedder

# Sample claims data
SAMPLE_CLAIMS = [
    {
        "customer_id": "CUST_001",
        "policy_number": "POL_AUTO_001",
        "claim_type": "auto",
        "description": "Minor fender bender in parking lot. Rear bumper damage, no injuries. Police report filed on site.",
        "amount": 2500.00,
        "location": "San Francisco, CA",
        "status": "approved"
    },
    {
        "customer_id": "CUST_002",
        "policy_number": "POL_HOME_002",
        "claim_type": "home",
        "description": "Water damage from burst pipe in kitchen. Significant damage to cabinets and flooring.",
        "amount": 15000.00,
        "location": "Austin, TX",
        "status": "approved"
    },
    {
        "customer_id": "CUST_003",
        "policy_number": "POL_AUTO_003",
        "claim_type": "auto",
        "description": "Highway collision with multiple vehicles. Significant front-end damage, airbags deployed. Driver taken to hospital for evaluation.",
        "amount": 28000.00,
        "location": "Los Angeles, CA",
        "status": "approved"
    },
    {
        "customer_id": "CUST_004",
        "policy_number": "POL_HEALTH_004",
        "claim_type": "health",
        "description": "Emergency room visit for broken arm after fall at home. X-rays and cast applied.",
        "amount": 3500.00,
        "location": "Seattle, WA",
        "status": "approved"
    },
    {
        "customer_id": "CUST_005",
        "policy_number": "POL_AUTO_005",
        "claim_type": "auto",
        "description": "Theft of vehicle from residential driveway. Police report filed, vehicle recovered with minor damage.",
        "amount": 12000.00,
        "location": "Phoenix, AZ",
        "status": "approved"
    },
    {
        "customer_id": "CUST_006",
        "policy_number": "POL_PROP_006",
        "claim_type": "property",
        "description": "Fire damage to garage from electrical short. Contains stored tools and equipment.",
        "amount": 45000.00,
        "location": "Denver, CO",
        "status": "rejected_insufficient_coverage"
    },
    {
        "customer_id": "CUST_007",
        "policy_number": "POL_AUTO_007",
        "claim_type": "auto",
        "description": "Windshield damage from flying debris on highway. No other vehicle involved.",
        "amount": 800.00,
        "location": "Chicago, IL",
        "status": "approved"
    },
    {
        "customer_id": "CUST_008",
        "policy_number": "POL_LIFE_008",
        "claim_type": "life",
        "description": "Life insurance claim for policyholder. Death certificate provided.",
        "amount": 250000.00,
        "location": "Miami, FL",
        "status": "approved"
    },
    {
        "customer_id": "CUST_009",
        "policy_number": "POL_HOME_009",
        "claim_type": "home",
        "description": "Roof damage from severe hailstorm. Multiple leaks throughout house.",
        "amount": 22000.00,
        "location": "Dallas, TX",
        "status": "under_review"
    },
    {
        "customer_id": "CUST_010",
        "policy_number": "POL_AUTO_010",
        "claim_type": "auto",
        "description": "Side-swipe accident on city street. Driver door damage, mirror broken. Other driver fled scene.",
        "amount": 4800.00,
        "location": "Boston, MA",
        "status": "approved"
    },
    {
        "customer_id": "CUST_011",
        "policy_number": "POL_HEALTH_011",
        "claim_type": "health",
        "description": "Surgery for appendicitis. Hospital stay for 3 days with complications.",
        "amount": 45000.00,
        "location": "Portland, OR",
        "status": "approved"
    },
    {
        "customer_id": "CUST_012",
        "policy_number": "PROP_LIABILITY_012",
        "claim_type": "liability",
        "description": "Customer slipped on wet floor in grocery store. Back injury requiring physical therapy.",
        "amount": 8500.00,
        "location": "Las Vegas, NV",
        "status": "approved"
    },
    {
        "customer_id": "CUST_013",
        "policy_number": "POL_AUTO_013",
        "claim_type": "auto",
        "description": "Multiple vehicle pileup during heavy fog. Extensive damage, legal proceedings initiated.",
        "amount": 75000.00,
        "location": "Philadelphia, PA",
        "status": "under_review"
    },
    {
        "customer_id": "CUST_014",
        "policy_number": "POL_HOME_014",
        "claim_type": "home",
        "description": "Burglary with electronics and jewelry stolen. Security footage available.",
        "amount": 15000.00,
        "location": "San Diego, CA",
        "status": "approved"
    },
    {
        "customer_id": "CUST_015",
        "policy_number": "POL_AUTO_015",
        "claim_type": "auto",
        "description": "Engine failure due to oil leak. Major engine repair required.",
        "amount": 6500.00,
        "location": "Nashville, TN",
        "status": "rejected_mechanical_failure"
    },
    # Some fraudulent or suspicious claims
    {
        "customer_id": "CUST_SUSP_001",
        "policy_number": "POL_AUTO_SUSP_001",
        "claim_type": "auto",
        "description": "Extremely expensive sports car claimed stolen but no police report. Security footage shows owner driving car after alleged theft.",
        "amount": 95000.00,
        "location": "Miami, FL",
        "status": "rejected_fraud"
    },
    {
        "customer_id": "CUST_SUSP_002",
        "policy_number": "POL_HOME_SUSP_002",
        "claim_type": "home",
        "description": "Claim for water damage immediately after policy purchase. Neighbors report no recent water issues in area.",
        "amount": 35000.00,
        "location": "Los Angeles, CA",
        "status": "rejected_fraud"
    },
    {
        "customer_id": "CUST_SUSP_003",
        "policy_number": "POL_HEALTH_SUSP_003",
        "claim_type": "health",
        "description": "Multiple claims for same injury across different hospitals within same week. Medical records show inconsistencies.",
        "amount": 25000.00,
        "location": "New York, NY",
        "status": "rejected_fraud"
    }
]

# Policy documents sample
POLICY_DOCUMENTS = [
    {
        "policy_number": "POL_AUTO_STANDARD",
        "document_type": "policy",
        "content": "Standard Auto Insurance Policy - Covers collision, liability, comprehensive with $1000 deductible. Excludes racing and commercial use.",
        "coverage_limits": {
            "liability": 50000,
            "collision": 25000,
            "comprehensive": 15000
        }
    },
    {
        "policy_number": "POL_HOME_PREMIUM",
        "document_type": "policy",
        "content": "Premium Homeowners Insurance - Full replacement coverage, including flood and earthquake protection with $2500 deductible.",
        "coverage_limits": {
            "dwelling": 500000,
            "personal_property": 250000,
            "liability": 1000000
        }
    },
    {
        "policy_number": "POL_HEALTH_BASIC",
        "document_type": "policy",
        "content": "Basic Health Insurance Plan - Covers hospital stays, emergency care, and major medical expenses with $2000 deductible.",
        "coverage_limits": {
            "hospital": 100000,
            "emergency": 25000,
            "major_medical": 200000
        }
    }
]

# Regulations sample
REGULATIONS = [
    {
        "regulation_id": "REG_AUTO_001",
        "document_type": "regulation",
        "content": "Department of Insurance Regulation 2024-A12: All auto insurance claims must be processed within 30 days of submission.",
        "jurisdiction": "California",
        "effective_date": "2024-01-01"
    },
    {
        "regulation_id": "REG_FRAUD_001",
        "document_type": "regulation",
        "content": "Insurance Fraud Prevention Act: Requires verification for all claims exceeding $50000 in value.",
        "jurisdiction": "Federal",
        "effective_date": "2023-07-01"
    },
    {
        "regulation_id": "REG_HEALTH_001",
        "document_type": "regulation",
        "content": "HIPAA Compliance Requirements: All health claim data must be encrypted and access logged.",
        "jurisdiction": "Federal",
        "effective_date": "2023-04-01"
    }
]

def create_sample_claims():
    """Create sample claims and populate Qdrant"""
    print("[INFO] Creating sample insurance claims data...")

    # Initialize components
    qdrant_manager = QdrantManager()
    embedder = MultimodalEmbedder()

    # Create and store sample claims
    created_claims = []

    for i, claim_data in enumerate(SAMPLE_CLAIMS):
        try:
            # Generate claim ID and add metadata
            claim_id = f"SAMPLE_{(i+1):03d}"
            claim_data['claim_id'] = claim_id
            claim_data['date_submitted'] = (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat()
            claim_data['processed_at'] = datetime.now().isoformat()
            claim_data['sample_data'] = True

            # Generate text embedding
            description = claim_data['description']
            text_embedding = embedder.embed_text(description)

            # Store claim in Qdrant
            point_id = qdrant_manager.add_claim(
                claim_data=claim_data,
                embedding=text_embedding,
                modality='text_claims'
            )

            created_claims.append({
                'claim_id': claim_id,
                'point_id': point_id,
                'status': claim_data['status']
            })

            print(f"[OK] Created sample claim: {claim_id}")

        except Exception as e:
            print(f"[ERROR] Failed to create claim {i+1}: {e}")

    # Store policy documents
    for policy in POLICY_DOCUMENTS:
        try:
            policy_id = policy['policy_number']
            policy['document_id'] = policy_id
            policy['created_at'] = datetime.now().isoformat()

            # Generate embedding
            text_embedding = embedder.embed_text(policy['content'])

            # Store policy
            point_id = qdrant_manager.add_claim(
                claim_data=policy,
                embedding=text_embedding,
                modality='policies'
            )

            print(f"[OK] Created policy document: {policy_id}")

        except Exception as e:
            print(f"[ERROR] Failed to create policy {policy['policy_number']}: {e}")

    # Store regulations
    for regulation in REGULATIONS:
        try:
            reg_id = regulation['regulation_id']
            regulation['document_id'] = reg_id
            regulation['created_at'] = datetime.now().isoformat()

            # Generate embedding
            text_embedding = embedder.embed_text(regulation['content'])

            # Store regulation
            point_id = qdrant_manager.add_claim(
                claim_data=regulation,
                embedding=text_embedding,
                modality='regulations'
            )

            print(f"[OK] Created regulation: {reg_id}")

        except Exception as e:
            print(f"[ERROR] Failed to create regulation {regulation['regulation_id']}: {e}")

    # Summary
    print(f"\n[OK] Sample data creation completed!")
    print(f"Claims created: {len(created_claims)}")
    print(f"Policy documents: {len(POLICY_DOCUMENTS)}")
    print(f"Regulations: {len(REGULATIONS)}")

    # Save sample data to file
    sample_data = {
        'claims': SAMPLE_CLAIMS,
        'policies': POLICY_DOCUMENTS,
        'regulations': REGULATIONS,
        'created_at': datetime.now().isoformat()
    }

    with open('data/sample_claims.json', 'w') as f:
        json.dump(sample_data, f, indent=2)

    print(f"[OK] Sample data saved to data/sample_claims.json")

    return created_claims

def get_test_scenarios():
    """Get predefined test scenarios for demo"""
    return {
        'high_value_claim': {
            'customer_id': 'CUST_TEST_001',
            'policy_number': 'POL_AUTO_TEST_001',
            'claim_type': 'auto',
            'description': 'Luxury sports car collision on highway. Front-end completely destroyed, airbags deployed, driver hospitalized.',
            'amount': 85000.00,
            'location': 'Los Angeles, CA'
        },
        'fraud_suspicious': {
            'customer_id': 'CUST_TEST_002',
            'policy_number': 'POL_AUTO_TEST_002',
            'claim_type': 'auto',
            'description': 'Brand new luxury vehicle claimed stolen immediately after purchase. No police report, no witnesses, no security footage available.',
            'amount': 95000.00,
            'location': 'Miami, FL'
        },
        'typical_claim': {
            'customer_id': 'CUST_TEST_003',
            'policy_number': 'POL_AUTO_TEST_003',
            'claim_type': 'auto',
            'description': 'Minor parking lot accident. Rear bumper damage, no injuries, other driver exchanged information.',
            'amount': 2200.00,
            'location': 'San Francisco, CA'
        },
        'water_damage': {
            'customer_id': 'CUST_TEST_004',
            'policy_number': 'POL_HOME_TEST_004',
            'claim_type': 'home',
            'description': 'Severe water damage from broken pipe in basement. Furniture, appliances, and flooring affected.',
            'amount': 28000.00,
            'location': 'Seattle, WA'
        }
    }

if __name__ == "__main__":
    print("=== AI Insurance Claims Processor - Sample Data Generator ===\n")

    try:
        created_claims = create_sample_claims()

        # Display test scenarios
        print("\n=== Test Scenarios for Demo ===")
        test_scenarios = get_test_scenarios()

        for name, scenario in test_scenarios.items():
            print(f"\n{name.replace('_', ' ').title()}:")
            print(f"  Amount: ${scenario['amount']:,.2f}")
            print(f"  Type: {scenario['claim_type']}")
            print(f"  Description: {scenario['description'][:100]}...")

        print(f"\n[OK] Ready for testing! You can now:")
        print(f"1. Start the backend: cd backend && python main.py")
        print(f"2. Start the frontend: cd frontend && streamlit run ui.py")
        print(f"3. Use the test scenarios above to demonstrate the system")

    except Exception as e:
        print(f"[ERROR] Failed to create sample data: {e}")
        import traceback
        traceback.print_exc()