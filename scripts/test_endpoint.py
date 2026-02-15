"""
Test the FastAPI endpoint to ensure it returns correct JSON format
without saving to database (main backend will handle that).
"""
import requests
import json

# Test with a known visit ID
visit_id = 1

print(f"Testing /api/v1/process_visit with visit_id={visit_id}")
print("="*60)

try:
    response = requests.post(
        "http://localhost:8000/api/v1/process_visit",
        json={"visit_id": visit_id}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS - Endpoint returned:")
        print(json.dumps(result, indent=2))
        
        # Verify structure
        required_fields = ['visit_id', 'risk_level', 'risk_score', 
                          'recommended_department', 'department_scores', 'explainability']
        missing = [f for f in required_fields if f not in result]
        
        if missing:
            print(f"\n⚠️ WARNING: Missing fields: {missing}")
        else:
            print("\n✅ All required fields present")
            print(f"\n📊 Summary:")
            print(f"  Visit ID: {result['visit_id']}")
            print(f"  Risk: {result['risk_level']} ({result['risk_score']:.2f})")
            print(f"  Department: {result['recommended_department']}")
            
    else:
        print(f"❌ ERROR: Status {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Cannot connect to FastAPI server")
    print("Please start the server first: uvicorn app.main:app --reload")
except Exception as e:
    print(f"❌ ERROR: {e}")
