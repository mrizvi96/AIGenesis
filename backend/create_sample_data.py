"""
Create Sample Data for AI Insurance Claims Testing
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from qdrant_manager import QdrantManager
from embeddings import MultimodalEmbedder
from sample_claims import SAMPLE_CLAIMS, POLICY_DOCUMENTS, REGULATIONS
import uuid
import json
import random
from datetime import datetime, timedelta

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

    return created_claims

if __name__ == "__main__":
    try:
        created_claims = create_sample_claims()
        print(f"[OK] Sample data populated successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to create sample data: {e}")
        import traceback
        traceback.print_exc()