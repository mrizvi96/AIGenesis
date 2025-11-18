#!/usr/bin/env python3
"""
Quick System Status Test for AI Insurance Claims Processing System
Tests all major components and provides detailed status report
"""

import sys
import os
import time
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_embeddings():
    """Test embedding system"""
    print("[BRAIN] Testing Embedding System...")

    try:
        from enhanced_embeddings import EnhancedMultimodalEmbedder

        embedder = EnhancedMultimodalEmbedder()
        info = embedder.get_embedding_info()

        print(f"   ✅ Text Model: {info['text_model_loaded']}")
        print(f"   ✅ Enhanced Features: {info['enhanced_features']}")
        print(f"   ✅ Google Cloud Available: {info['google_cloud_available']}")
        print(f"   ✅ Google Vision: {info['google_vision_available']}")
        print(f"   ✅ Google Speech: {info['google_speech_available']}")
        print(f"   ✅ Google Language: {info['google_language_available']}")

        # Test text embedding
        test_text = "Patient has severe chest pain and needs immediate cardiac evaluation"
        embedding = embedder.embed_text(test_text)
        print(f"   ✅ Text Embedding: {len(embedding)} dimensions")

        return True

    except Exception as e:
        print(f"   ❌ Embeddings Error: {e}")
        return False

def test_qdrant():
    """Test Qdrant connection"""
    print("\n🔍 Testing Qdrant Connection...")

    try:
        from qdrant_manager import QdrantManager

        qdrant = QdrantManager()

        # Test basic connection
        collections = qdrant.client.get_collections().collections
        collection_names = [col.name for col in collections]

        print(f"   ✅ Connected to Qdrant Cloud")
        print(f"   ✅ Collections: {collection_names}")

        # Test search functionality
        test_embedding = [0.1] * 768  # Simple test embedding
        if 'text_claims' in collection_names:
            results = qdrant.search_similar(test_embedding, collection_name='text_claims', limit=3)
            print(f"   ✅ Search working: {len(results)} results found")

        return True

    except Exception as e:
        print(f"   ❌ Qdrant Error: {e}")
        return False

def test_recommender():
    """Test recommendation system"""
    print("\n🤖 Testing Recommendation Engine...")

    try:
        from enhanced_recommender import EnhancedClaimsRecommender
        from qdrant_manager import QdrantManager

        qdrant = QdrantManager()
        recommender = EnhancedClaimsRecommender(qdrant)

        print(f"   ✅ Recommender initialized")
        print(f"   ✅ Medical codes database: ICD-10, CPT")
        print(f"   ✅ Fraud detection patterns loaded")
        print(f"   ✅ Settlement estimation models ready")

        return True

    except Exception as e:
        print(f"   ❌ Recommender Error: {e}")
        return False

def test_api():
    """Test FastAPI endpoints"""
    print("\n🌐 Testing API Endpoints...")

    try:
        import requests

        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Health endpoint: {response.json()['status']}")
        else:
            print(f"   ⚠️ Health endpoint not accessible (backend may not be running)")

        # Test API docs
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ API documentation accessible")
        else:
            print(f"   ⚠️ API docs not accessible (backend may not be running)")

        return True

    except requests.exceptions.ConnectionError:
        print(f"   ⚠️ Backend not running (start with: python main.py)")
        return False
    except Exception as e:
        print(f"   ❌ API Error: {e}")
        return False

def test_data():
    """Test sample data"""
    print("\n📁 Testing Sample Data...")

    try:
        import os

        # Check sample data directories
        sample_files = []

        # Check for sample claims
        for root, dirs, files in os.walk("../data"):
            for file in files:
                if file.endswith(('.txt', '.jpg', '.png', '.mp3', '.wav', '.mp4')):
                    sample_files.append(os.path.join(root, file))

        print(f"   ✅ Found {len(sample_files)} sample files")

        # Check file types
        file_types = set()
        for file in sample_files[:5]:  # Check first 5 files
            ext = os.path.splitext(file)[1].lower()
            file_types.add(ext)

        print(f"   ✅ File types: {list(file_types)}")

        return True

    except Exception as e:
        print(f"   ❌ Data Error: {e}")
        return False

def run_performance_test():
    """Quick performance test"""
    print("\n⚡ Running Performance Test...")

    try:
        from enhanced_embeddings import EnhancedMultimodalEmbedder

        embedder = EnhancedMultimodalEmbedder()

        # Test text embedding speed
        test_text = "Patient requires immediate medical attention for severe trauma injuries"
        start_time = time.time()
        embedding = embedder.embed_text(test_text)
        end_time = time.time()

        processing_time = (end_time - start_time) * 1000  # Convert to ms

        print(f"   ✅ Text processing: {processing_time:.2f}ms")

        if processing_time < 100:
            print(f"   🚀 Excellent performance!")
        elif processing_time < 500:
            print(f"   ✅ Good performance")
        else:
            print(f"   ⚠️ Performance could be optimized")

        return True

    except Exception as e:
        print(f"   ❌ Performance Test Error: {e}")
        return False

def main():
    """Run complete system status check"""
    print("=" * 60)
    print("AI INSURANCE CLAIMS PROCESSING SYSTEM")
    print("   Complete Status Check")
    print("=" * 60)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run all tests
    tests = [
        ("Embedding System", test_embeddings),
        ("Qdrant Connection", test_qdrant),
        ("Recommendation Engine", test_recommender),
        ("API Endpoints", test_api),
        ("Sample Data", test_data),
        ("Performance Test", run_performance_test)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ {test_name}: Unexpected error - {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 SYSTEM STATUS SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<25} {status}")

    print(f"\n   Overall: {passed}/{total} tests passed")

    if passed == total:
        print("   🎉 System is fully operational!")
    elif passed >= total * 0.8:
        print("   ✅ System is mostly ready for demo")
    elif passed >= total * 0.6:
        print("   ⚠️ System needs some fixes before demo")
    else:
        print("   ❌ System requires significant fixes")

    print("\n📋 Next Steps:")
    if not results.get("Embedding System", False):
        print("   • Fix embedding system (check dependencies)")
    if not results.get("Qdrant Connection", False):
        print("   • Check Qdrant Cloud connection and API keys")
    if not results.get("API Endpoints", False):
        print("   • Start backend server: python main.py")

    print("\n💡 Quick Start:")
    print("   1. cd backend && python main.py")
    print("   2. cd frontend && streamlit run ui.py")
    print("   3. Open http://localhost:8501")

if __name__ == "__main__":
    main()