import sys
import os
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

print(f"Added {project_root} to sys.path")

try:
    from app.models.ml_engine import MLEngine
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def test_prediction():
    try:
        engine = MLEngine(model_dir='app/models')
        
        # Patient 16 (Healthy, Age 34)
        test_patient = {
            'visit_id': 15, 
            'age': 36, 
            'bp_systolic': 111, 
            'bp_diastolic': 78,
            'heart_rate': 75, 
            'temperature': 99, 
            'chest_pain_severity': 5,
            'max_severity': 2, 
            'symptom_count': 1, 
            'comorbidities_count': 1,
            'cardiac_history': 3, 
            'diabetes_status': 2, 
            'respiratory_history': 2,
            'chronic_conditions': 1
        }
        
        print(f"Testing Patient: {test_patient}")
        result = engine.predict(test_patient)
        
        output_file = 'verification_output.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"Successfully wrote output to {output_file}")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
