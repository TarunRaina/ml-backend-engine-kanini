"""
Test Orthopedics Scoring with Sample Data
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from app.models.rule_engine import RuleBasedTriageEngine
import json

def test_orthopedic_case():
    """Test with orthopedic symptoms"""
    
    # Sample patient with orthopedic symptoms
    patient_data = {
        'age': 55,
        'gender': 'M',
        'chief_complaint': 'back stiffness and joint pain',
        'vitals': {
            'bp_systolic': 130,
            'bp_diastolic': 85,
            'heart_rate': 75,
            'temperature': 98.6
        },
        'symptoms': [
            {'symptom_name': 'Back stiffness', 'severity_score': 4, 'duration': '2 days'},
            {'symptom_name': 'Joint pain', 'severity_score': 5, 'duration': '1 week'},
            {'symptom_name': 'Muscle weakness', 'severity_score': 3, 'duration': '3 days'}
        ],
        'medical_history': [
            {'condition_name': 'Arthritis', 'is_chronic': True, 'diagnosis_date': '2020-01-01'},
            {'condition_name': 'Hypertension', 'is_chronic': True, 'diagnosis_date': '2018-06-15'}
        ]
    }
    
    print("\n" + "="*100)
    print("ORTHOPEDICS TEST CASE")
    print("="*100)
    print(f"\nPatient: {patient_data['age']} year old {patient_data['gender']}")
    print(f"Chief Complaint: {patient_data['chief_complaint']}")
    print(f"\nSymptoms:")
    for s in patient_data['symptoms']:
        print(f"  - {s['symptom_name']} (severity: {s['severity_score']})")
    print(f"\nMedical History:")
    for h in patient_data['medical_history']:
        print(f"  - {h['condition_name']}")
    
    # Run triage
    engine = RuleBasedTriageEngine()
    result = engine.predict(patient_data)
    
    print(f"\n{'='*100}")
    print("TRIAGE RESULTS")
    print(f"{'='*100}\n")
    
    print(f"Risk Level: {result['risk_level']}")
    print(f"Risk Score: {result['risk_score']:.4f}")
    print(f"Primary Department: {result['primary_department']}")
    
    print(f"\nDEPARTMENT SCORES:")
    for dept, score in sorted(result['department_scores'].items(), key=lambda x: x[1], reverse=True):
        marker = "✅ PRIMARY" if dept == result['primary_department'] else ("✅ QUEUE" if score >= 0.35 else "")
        print(f"  {dept}: {score:.3f} {marker}")
    
    print(f"\nEXPLAINABILITY:")
    if 'department_reasoning' in result['explainability']:
        print(f"  Department Reasoning:")
        for dept, reason in result['explainability']['department_reasoning'].items():
            print(f"    - {dept}: {reason}")
    
    if 'score_breakdown' in result['explainability']:
        breakdown = result['explainability']['score_breakdown']
        print(f"\n  Score Breakdown:")
        print(f"    - Symptoms: {breakdown.get('symptom_score')}")
        print(f"    - Vitals: {breakdown.get('vitals_score')}")
        print(f"    - History: {breakdown.get('history_score')}")
        print(f"    - Age: {breakdown.get('age_score')}")
        print(f"    - TOTAL: {breakdown.get('total')}")
    
    print(f"\n{'='*100}\n")
    
    # Save result
    with open('orthopedics_test_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("✅ Saved result to orthopedics_test_result.json\n")
    
    return result

if __name__ == "__main__":
    test_orthopedic_case()
