"""
Detailed API Test - Show Full Outputs
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_and_display(visit_id: int, description: str):
    """Test and display full output"""
    print(f"\n{'='*100}")
    print(f"VISIT {visit_id}: {description}")
    print(f"{'='*100}\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/process_visit",
            json={"visit_id": visit_id},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Display in readable format
            print(f"✅ API RESPONSE (Status: {response.status_code})\n")
            print(json.dumps(data, indent=2))
            
            return data
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("\n" + "="*100)
    print("DETAILED API TEST RESULTS - RULE-BASED TRIAGE ENGINE")
    print("="*100)
    
    tests = [
        (1, "Headache + Neck Pain (Severity 5)"),
        (2, "Seizures + Headache + Chest Pain + Fever"),
        (4, "Chest Pain + Loss of Consciousness"),
        (9, "Chest Pain (Patient with Coronary Artery Disease)"),
    ]
    
    for visit_id, description in tests:
        test_and_display(visit_id, description)
    
    print(f"\n{'='*100}")
    print("END OF TESTS")
    print(f"{'='*100}\n")
