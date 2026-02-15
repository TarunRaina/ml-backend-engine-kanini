"""
Test API endpoints with different visit scenarios
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_visit(visit_id: int, description: str):
    """Test a single visit via API"""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"Visit ID: {visit_id}")
    print(f"{'='*80}\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/process_visit",
            json={"visit_id": visit_id},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ SUCCESS - Status Code: {response.status_code}")
            print(f"\n📊 RESULTS:")
            print(f"   Risk Level: {data['risk_level']}")
            print(f"   Risk Score: {data['risk_score']:.4f}")
            print(f"   Primary Department: {data['primary_department']}")
            
            print(f"\n🏥 DEPARTMENT SCORES:")
            for dept, score in sorted(data['department_scores'].items(), key=lambda x: x[1], reverse=True):
                marker = "✅" if score >= 0.35 else ""
                print(f"   {dept}: {score:.3f} {marker}")
            
            print(f"\n💡 EXPLAINABILITY:")
            if 'risk_factors' in data['explainability']:
                print(f"   Risk Factors: {list(data['explainability']['risk_factors'].keys())}")
            if 'department_reasoning' in data['explainability']:
                print(f"   Departments Explained: {list(data['explainability']['department_reasoning'].keys())}")
            if 'score_breakdown' in data['explainability']:
                breakdown = data['explainability']['score_breakdown']
                print(f"   Score Breakdown: Symptoms={breakdown.get('symptom_score')}, Vitals={breakdown.get('vitals_score')}, History={breakdown.get('history_score')}, Total={breakdown.get('total')}")
            
            print(f"\n🎯 CONFIDENCE:")
            conf = data['confidence']
            print(f"   Overall: {conf['overall']:.3f}")
            print(f"   Data Completeness: {conf['data_completeness']:.3f}")
            print(f"   Has Critical Indicators: {conf['has_critical_indicators']}")
            
            return True
        else:
            print(f"❌ FAILED - Status Code: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("API ENDPOINT TESTING - RULE-BASED TRIAGE ENGINE")
    print("="*80)
    
    # Test different scenarios
    tests = [
        (1, "Headache + Neck Pain (High Severity)"),
        (2, "Seizures + Headache + Chest Pain"),
        (4, "Chest Pain + Loss of Consciousness"),
        (9, "Chest Pain (Patient with CAD History)"),
    ]
    
    results = []
    for visit_id, description in tests:
        success = test_visit(visit_id, description)
        results.append((visit_id, success))
    
    print(f"\n\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    passed = sum(1 for _, success in results if success)
    print(f"Passed: {passed}/{len(tests)}")
    for visit_id, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Visit {visit_id}: {status}")
    print(f"{'='*80}\n")
