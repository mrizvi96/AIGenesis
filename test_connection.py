#!/usr/bin/env python3
"""
Simple connection test for debugging the API connection issue
"""

import requests
import json
import time

def test_connection():
    """Test the backend connection from the same environment as the frontend"""

    print("=" * 50)
    print("API CONNECTION TEST")
    print("=" * 50)

    # Test both localhost and 127.0.0.1
    urls = [
        "http://localhost:8000/health",
        "http://127.0.0.1:8000/health"
    ]

    for url in urls:
        print(f"\n[TEST] Testing: {url}")
        try:
            response = requests.get(url, timeout=5)
            print(f"   ✅ Status Code: {response.status_code}")
            print(f"   ✅ Response: {response.text}")

            # Test the full system info endpoint
            info_response = requests.get(f"{url.replace('/health', '/system_info')}", timeout=5)
            print(f"   ✅ System Info: {info_response.status_code}")

        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection Error - Cannot reach server")
        except requests.exceptions.Timeout:
            print(f"   ❌ Timeout Error - Server not responding")
        except Exception as e:
            print(f"   ❌ Other Error: {e}")

    print("\n" + "=" * 50)
    print("MANUAL API TEST")
    print("=" * 50)

    # Test processing a claim directly
    url = "http://127.0.0.1:8000/process_text_claim"
    test_data = {
        "text": "Patient presents with severe chest pain and shortness of breath. ECG shows abnormal rhythm.",
        "claim_type": "medical",
        "priority": "high"
    }

    print(f"\n[PROCESS] Processing test claim...")
    print(f"   URL: {url}")
    print(f"   Data: {json.dumps(test_data, indent=2)}")

    try:
        response = requests.post(url, json=test_data, timeout=10)
        print(f"   ✅ Status Code: {response.status_code}")
        print(f"   ✅ Response: {json.dumps(response.json(), indent=2)}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_connection()