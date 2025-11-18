import requests
import json

def test_backend():
    print("Testing backend API...")

    # Test health endpoint
    try:
        resp = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print(f"Health check: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # Test claim processing
    try:
        data = {
            "text": "Patient has severe chest pain and needs immediate care",
            "claim_type": "medical"
        }
        resp = requests.post("http://127.0.0.1:8000/process_text_claim", json=data, timeout=10)
        print(f"Claim processing: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            print("SUCCESS: AI processing working!")
            print(f"Recommendation: {result.get('data', {}).get('recommendation', 'N/A')}")
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Claim processing failed: {e}")

if __name__ == "__main__":
    test_backend()