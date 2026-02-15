"""
Test ALL Department Scoring
Verify that each department gets proper priority when relevant symptoms/history are present
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from app.models.rule_engine import RuleBasedTriageEngine
import json

def test_all_departments():
    """Test cases for all 6 departments"""
    
    test_cases = [
        {
            'name': 'EMERGENCY - Severe Trauma',
            'data': {
                'age': 45,
                'gender': 'M',
                'chief_complaint': 'loss of consciousness after fall',
                'vitals': {'bp_systolic': 185, 'bp_diastolic': 115, 'heart_rate': 125, 'temperature': 98.6},
                'symptoms': [
                    {'symptom_name': 'Loss of consciousness', 'severity_score': 5, 'duration': '10 minutes'},
                    {'symptom_name': 'Severe headache', 'severity_score': 5, 'duration': '30 minutes'}
                ],
                'medical_history': []
            },
            'expected_primary': 'Emergency'
        },
        {
            'name': 'CARDIOLOGY - Chest Pain + Cardiac History',
            'data': {
                'age': 62,
                'gender': 'M',
                'chief_complaint': 'severe chest pain',
                'vitals': {'bp_systolic': 165, 'bp_diastolic': 100, 'heart_rate': 110, 'temperature': 99.0},
                'symptoms': [
                    {'symptom_name': 'Chest pain', 'severity_score': 5, 'duration': '2 hours'},
                    {'symptom_name': 'Shortness of breath', 'severity_score': 4, 'duration': '1 hour'},
                    {'symptom_name': 'Palpitations', 'severity_score': 3, 'duration': '30 minutes'}
                ],
                'medical_history': [
                    {'condition_name': 'Coronary Artery Disease', 'is_chronic': True, 'diagnosis_date': '2020-01-01'},
                    {'condition_name': 'Hypertension', 'is_chronic': True, 'diagnosis_date': '2018-01-01'}
                ]
            },
            'expected_primary': 'Cardiology'
        },
        {
            'name': 'NEUROLOGY - Seizures + Neuro History',
            'data': {
                'age': 38,
                'gender': 'F',
                'chief_complaint': 'seizure episode',
                'vitals': {'bp_systolic': 130, 'bp_diastolic': 85, 'heart_rate': 88, 'temperature': 98.6},
                'symptoms': [
                    {'symptom_name': 'Seizures', 'severity_score': 5, 'duration': '5 minutes'},
                    {'symptom_name': 'Dizziness', 'severity_score': 4, 'duration': '1 hour'},
                    {'symptom_name': 'Numbness', 'severity_score': 3, 'duration': '30 minutes'}
                ],
                'medical_history': [
                    {'condition_name': 'Epilepsy', 'is_chronic': True, 'diagnosis_date': '2015-01-01'}
                ]
            },
            'expected_primary': 'Neurology'
        },
        {
            'name': 'RESPIRATORY - Breathing Issues + Asthma',
            'data': {
                'age': 55,
                'gender': 'F',
                'chief_complaint': 'severe shortness of breath',
                'vitals': {'bp_systolic': 135, 'bp_diastolic': 88, 'heart_rate': 95, 'temperature': 100.5},
                'symptoms': [
                    {'symptom_name': 'Shortness of breath', 'severity_score': 5, 'duration': '3 hours'},
                    {'symptom_name': 'Wheezing', 'severity_score': 4, 'duration': '2 hours'},
                    {'symptom_name': 'Persistent cough', 'severity_score': 4, 'duration': '1 day'}
                ],
                'medical_history': [
                    {'condition_name': 'Asthma', 'is_chronic': True, 'diagnosis_date': '2010-01-01'},
                    {'condition_name': 'COPD', 'is_chronic': True, 'diagnosis_date': '2018-01-01'}
                ]
            },
            'expected_primary': 'Respiratory'
        },
        {
            'name': 'ORTHOPEDICS - Joint Pain + Arthritis',
            'data': {
                'age': 60,
                'gender': 'M',
                'chief_complaint': 'severe back and joint pain',
                'vitals': {'bp_systolic': 128, 'bp_diastolic': 82, 'heart_rate': 75, 'temperature': 98.6},
                'symptoms': [
                    {'symptom_name': 'Back stiffness', 'severity_score': 5, 'duration': '1 week'},
                    {'symptom_name': 'Joint pain', 'severity_score': 5, 'duration': '3 days'},
                    {'symptom_name': 'Muscle weakness', 'severity_score': 4, 'duration': '2 days'}
                ],
                'medical_history': [
                    {'condition_name': 'Arthritis', 'is_chronic': True, 'diagnosis_date': '2015-01-01'},
                    {'condition_name': 'Osteoporosis', 'is_chronic': True, 'diagnosis_date': '2019-01-01'}
                ]
            },
            'expected_primary': 'Orthopedics'
        },
        {
            'name': 'GENERAL MEDICINE - Low Risk Multiple Chronic',
            'data': {
                'age': 50,
                'gender': 'F',
                'chief_complaint': 'general fatigue and weakness',
                'vitals': {'bp_systolic': 125, 'bp_diastolic': 80, 'heart_rate': 78, 'temperature': 98.6},
                'symptoms': [
                    {'symptom_name': 'Fatigue', 'severity_score': 2, 'duration': '1 week'},
                    {'symptom_name': 'Headache', 'severity_score': 2, 'duration': '2 days'}
                ],
                'medical_history': [
                    {'condition_name': 'Diabetes', 'is_chronic': True, 'diagnosis_date': '2015-01-01'},
                    {'condition_name': 'Hypertension', 'is_chronic': True, 'diagnosis_date': '2016-01-01'},
                    {'condition_name': 'Hypothyroidism', 'is_chronic': True, 'diagnosis_date': '2017-01-01'}
                ]
            },
            'expected_primary': 'General Medicine'
        }
    ]
    
    engine = RuleBasedTriageEngine()
    results = []
    
    print("\n" + "="*120)
    print("COMPREHENSIVE DEPARTMENT SCORING TEST")
    print("="*120)
    
    for test_case in test_cases:
        print(f"\n{'='*120}")
        print(f"TEST: {test_case['name']}")
        print(f"Expected Primary: {test_case['expected_primary']}")
        print(f"{'='*120}\n")
        
        result = engine.predict(test_case['data'])
        
        print(f"✅ Actual Primary: {result['primary_department']}")
        print(f"   Risk Level: {result['risk_level']} ({result['risk_score']:.4f})")
        
        print(f"\n   DEPARTMENT SCORES:")
        for dept, score in sorted(result['department_scores'].items(), key=lambda x: x[1], reverse=True):
            marker = "🎯 PRIMARY" if dept == result['primary_department'] else ("✅ QUEUE" if score >= 0.35 else "")
            print(f"      {dept}: {score:.3f} {marker}")
        
        if 'department_reasoning' in result['explainability']:
            print(f"\n   REASONING:")
            for dept, reason in result['explainability']['department_reasoning'].items():
                print(f"      {dept}: {reason}")
        
        # Check if expected matches actual
        match = result['primary_department'] == test_case['expected_primary']
        status = "✅ PASS" if match else "❌ FAIL"
        print(f"\n   {status}")
        
        results.append({
            'test': test_case['name'],
            'expected': test_case['expected_primary'],
            'actual': result['primary_department'],
            'match': match,
            'scores': result['department_scores']
        })
    
    print(f"\n\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    passed = sum(1 for r in results if r['match'])
    print(f"Passed: {passed}/{len(results)}")
    for r in results:
        status = "✅" if r['match'] else "❌"
        print(f"   {status} {r['test']}: Expected={r['expected']}, Actual={r['actual']}")
    print(f"{'='*120}\n")
    
    # Save results
    with open('all_departments_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✅ Saved results to all_departments_test.json\n")

if __name__ == "__main__":
    test_all_departments()
