"""
AI Insurance Claims Processing System Demo
Complete system demonstration and status check
"""

import sys
import os
import json
import time
import requests
from datetime import datetime

# Add backend to path
sys.path.append('backend')

def check_system_status():
    """Check overall system status"""
    print("=== AI INSURANCE CLAIMS PROCESSING SYSTEM STATUS ===\n")

    try:
        # Test imports
        print("[INFO] Testing Python imports...")
        from qdrant_manager import QdrantManager
        from embeddings import MultimodalEmbedder
        from recommender import ClaimsRecommender
        print("[OK] All Python modules imported successfully")

        # Test Qdrant connection
        print("\n[INFO] Testing Qdrant Cloud connection...")
        qdrant = QdrantManager()
        if qdrant.test_connection():
            print("[OK] Qdrant Cloud connection successful")
        else:
            print("[ERROR] Qdrant Cloud connection failed")
            return False

        # Test embedding system
        print("\n[INFO] Testing embedding system...")
        embedder = MultimodalEmbedder()
        test_embedding = embedder.embed_text("Test claim about car accident")
        print(f"[OK] Text embedding generated (dimension: {len(test_embedding)})")

        # Test recommendation engine
        print("\n[INFO] Testing recommendation engine...")
        recommender = ClaimsRecommender(qdrant, embedder)
        print("[OK] Recommendation engine initialized")

        # Check collections
        print("\n[INFO] Checking Qdrant collections...")
        collections = qdrant.get_collection_info()
        for name, info in collections.items():
            if "error" not in info:
                print(f"[OK] Collection '{name}': Ready")
            else:
                print(f"[WARNING] Collection '{name}': {info['error']}")

        print(f"\n[OK] System status: HEALTHY")
        return True

    except Exception as e:
        print(f"[ERROR] System check failed: {e}")
        return False

def demo_claim_submission():
    """Demonstrate claim submission"""
    print("\n=== DEMO: CLAIM SUBMISSION ===\n")

    try:
        # Test claim data
        test_claim = {
            "claim_id": "DEMO_001",
            "customer_id": "CUST_DEMO_001",
            "policy_number": "POL_DEMO_001",
            "claim_type": "auto",
            "description": "Minor rear-end collision at traffic light. Bumper damage, no injuries.",
            "amount": 3500.00,
            "location": "Demo City, USA"
        }

        print(f"Submitting claim: {test_claim['description']}")
        print(f"Amount: ${test_claim['amount']:,.2f}")

        # Import modules
        from qdrant_manager import QdrantManager
        from embeddings import MultimodalEmbedder
        from recommender import ClaimsRecommender

        # Initialize components
        qdrant = QdrantManager()
        embedder = MultimodalEmbedder()
        recommender = ClaimsRecommender(qdrant, embedder)

        # Process claim
        text_embedding = embedder.embed_text(test_claim['description'])
        point_id = qdrant.add_claim(test_claim, text_embedding, 'text_claims')
        recommendation = recommender.recommend_outcome(test_claim, text_embedding)

        print(f"[OK] Claim submitted with ID: {point_id}")
        print(f"[OK] Recommendation: {recommendation['recommendation']['action']}")
        print(f"[OK] Fraud Risk: {recommendation['fraud_risk']['risk_level']} ({recommendation['fraud_risk']['risk_score']:.1%})")
        print(f"[OK] Settlement Estimate: ${recommendation['settlement_estimate']['estimated_amount']:,.2f}")
        print(f"[OK] Similar Claims Found: {recommendation['similar_claims_count']}")

        return True

    except Exception as e:
        print(f"[ERROR] Demo claim submission failed: {e}")
        return False

def demo_search_functionality():
    """Demonstrate search functionality"""
    print("\n=== DEMO: SEARCH FUNCTIONALITY ===\n")

    try:
        from qdrant_manager import QdrantManager
        from embeddings import MultimodalEmbedder

        qdrant = QdrantManager()
        embedder = MultimodalEmbedder()

        # Test searches
        test_queries = [
            "car accident",
            "water damage",
            "theft",
            "medical emergency",
            "suspicious claim"
        ]

        for query in test_queries:
            print(f"Searching for: '{query}'")
            embedding = embedder.embed_text(query)
            results = qdrant.search_similar_claims(embedding, 'text_claims', limit=3)
            print(f"Found {len(results)} similar claims")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result.get('description', 'N/A')[:80]}... (Score: {result.get('similarity_score', 0):.1%})")
            print()

        return True

    except Exception as e:
        print(f"[ERROR] Search demo failed: {e}")
        return False

def check_api_server():
    """Check if API server is running"""
    print("\n=== API SERVER STATUS ===\n")

    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("[OK] API server is running on localhost:8000")
            print(f"    Status: {data.get('status', 'unknown')}")
            print(f"    Qdrant Connected: {data.get('services', {}).get('qdrant_connected', False)}")
            print(f"    Embedder Ready: {data.get('services', {}).get('embedder_loaded', False)}")
            return True
        else:
            print(f"[ERROR] API server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[WARNING] API server not running on localhost:8000")
        print("To start the API server, run:")
        print("  cd backend && python main.py")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to check API server: {e}")
        return False

def generate_launch_script():
    """Generate launch script for easy demo"""
    script_content = """#!/bin/bash
# AI Insurance Claims Processing System Launch Script

echo "Starting AI Insurance Claims Processing System..."

# Start backend server
echo "Starting FastAPI backend server..."
cd backend
python main.py &
BACKEND_PID=$!

echo "Backend server started with PID: $BACKEND_PID"
echo "API available at: http://localhost:8000"

# Wait for backend to start
sleep 10

# Start frontend
echo "Starting Streamlit frontend..."
cd ../frontend
streamlit run ui.py &
FRONTEND_PID=$!

echo "Frontend started with PID: $FRONTEND_PID"
echo "Frontend available at: http://localhost:8501"

echo ""
echo "=== SYSTEM READY ==="
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:8501"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "To stop the system:"
echo "kill $BACKEND_PID $FRONTEND_PID"
"""

    with open('start_system.sh', 'w') as f:
        f.write(script_content)

    # For Windows
    batch_content = """@echo off
echo Starting AI Insurance Claims Processing System...

echo Starting FastAPI backend server...
cd backend
start /B python main.py

echo Waiting for backend to start...
timeout /t 10 /nobreak

echo Starting Streamlit frontend...
cd ..\\frontend
start /B streamlit run ui.py

echo.
echo === SYSTEM READY ===
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:8501
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to continue...
pause
"""

    with open('start_system.bat', 'w') as f:
        f.write(batch_content)

    print("[OK] Launch scripts created:")
    print("  - start_system.sh (Linux/Mac)")
    print("  - start_system.bat (Windows)")

def main():
    """Main demo function"""
    print("AI INSURANCE CLAIMS PROCESSING SYSTEM DEMO")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # System status check
    if not check_system_status():
        print("\n[ERROR] System is not healthy. Please check the errors above.")
        return

    # API server check
    api_running = check_api_server()

    # Demo functionality
    print("\n" + "=" * 50)
    print("RUNNING FUNCTIONALITY DEMOS")
    print("=" * 50)

    demo_success = True
    demo_success &= demo_claim_submission()
    demo_success &= demo_search_functionality()

    # Generate launch scripts
    generate_launch_script()

    # Final summary
    print("\n" + "=" * 50)
    print("DEMO SUMMARY")
    print("=" * 50)

    system_health = "HEALTHY" if check_system_status() else "UNHEALTHY"
    api_status = "RUNNING" if api_running else "STOPPED"
    demo_status = "SUCCESS" if demo_success else "FAILED"

    print(f"System Health: {system_health}")
    print(f"API Server: {api_status}")
    print(f"Functionality Demo: {demo_status}")

    if system_health == "HEALTHY":
        print("\n[SUCCESS] The AI Insurance Claims Processing System is ready!")
        print("\nTo start the complete system:")
        print("1. Backend: cd backend && python main.py")
        print("2. Frontend: cd frontend && streamlit run ui.py")
        print("3. Or use the launch scripts: start_system.sh (Linux/Mac) or start_system.bat (Windows)")
        print("\nAccess the system at:")
        print("- Frontend: http://localhost:8501")
        print("- API: http://localhost:8000")
        print("- API Docs: http://localhost:8000/docs")
    else:
        print("\n[ERROR] System is not ready. Please resolve the issues above.")

if __name__ == "__main__":
    main()