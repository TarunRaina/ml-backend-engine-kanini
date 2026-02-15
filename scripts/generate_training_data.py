import pandas as pd
import numpy as np
import os
import random
from typing import Dict, List

def generate_synthetic_visit(visit_id: int) -> Dict:
    """FULL medical visit with patient history - ALL VARS INITIALIZED."""
    
    # Initialize ALL variables first
    age = np.clip(np.random.normal(45, 20), 18, 90).astype(int)
    heart_rate = 80
    bp_systolic = 120
    bp_diastolic = 80
    temp = 98.6
    chest_pain = 0
    
    # Patient History
    comorbidities_count = 0
    cardiac_history = 0
    diabetes_status = 0
    respiratory_history = 0
    chronic_conditions = 0
    
    if random.random() < 0.20:
        comorbidities_count = random.choices([1, 2, 3, 4], weights=[50, 30, 15, 5])[0]
        cardiac_history = 1 if random.random() < (0.08 + age * 0.001) else 0
        diabetes_status = 1 if random.random() < 0.12 else 0
        respiratory_history = 1 if random.random() < 0.10 else 0
        chronic_conditions = 1 if comorbidities_count >= 2 else 0
    
    # Vitals (history correlated) - NOW SAFE
    if cardiac_history == 1:
        heart_rate = np.clip(np.random.normal(95, 15), 70, 140).astype(int)
        bp_systolic = np.clip(np.random.normal(145, 15), 120, 180).astype(int)
    elif random.random() < 0.10:  # High risk
        heart_rate = np.clip(np.random.normal(140, 15), 100, 200).astype(int)
        bp_systolic = np.clip(np.random.normal(170, 10), 140, 220).astype(int)
        chest_pain = random.choice([4, 5])
        temp = np.clip(np.random.normal(102, 1), 98, 105)
    elif random.random() < 0.30:  # Medium risk
        heart_rate = np.clip(np.random.normal(105, 8), 80, 130).astype(int)
        bp_systolic = np.clip(np.random.normal(140, 8), 120, 160).astype(int)
        chest_pain = random.choice([2, 3])
        temp = np.clip(np.random.normal(100.5, 0.5), 98, 102)
    else:  # Low risk
        heart_rate = np.clip(np.random.normal(80, 10), 60, 100).astype(int)
        bp_systolic = np.clip(np.random.normal(120, 10), 100, 140).astype(int)
        chest_pain = random.choice([0, 1])
        temp = np.clip(np.random.normal(98.6, 0.5), 97, 100)
    
    bp_diastolic = np.clip(bp_systolic * 0.6 + np.random.normal(0, 5), 60, 100).astype(int)
    max_severity = random.choices([1, 2, 3, 4, 5], weights=[30, 25, 20, 15, 10])[0]
    symptom_count = random.choices([1, 2, 3, 4, 5, 6], weights=[20, 25, 20, 15, 10, 10])[0]
    complaints = ['headache', 'chest pain', 'fever', 'cough', 'fatigue', 'dizziness', 'abdominal pain']
    complaint = random.choice(complaints)
    
    return {
        'visit_id': visit_id, 'age': age,
        'bp_systolic': bp_systolic, 'bp_diastolic': bp_diastolic,
        'heart_rate': heart_rate, 'temperature': round(temp, 1),
        'chest_pain_severity': chest_pain, 'max_severity': max_severity,
        'symptom_count': symptom_count, 'chief_complaint': complaint,
        'comorbidities_count': comorbidities_count, 'cardiac_history': cardiac_history,
        'diabetes_status': diabetes_status, 'respiratory_history': respiratory_history,
        'chronic_conditions': chronic_conditions
    }

def generate_labels(features: Dict) -> Dict:
    heart_rate = features['heart_rate']
    bp_systolic = features['bp_systolic']
    temp = features['temperature']
    chest_pain = features['chest_pain_severity']
    cardiac_hist = features['cardiac_history']
    diabetes = features['diabetes_status']
    respiratory_hist = features['respiratory_history']
    chronic = features['chronic_conditions']
    
    risk_score = 0
    if heart_rate > 120: risk_score += 3
    if bp_systolic > 160: risk_score += 3
    if temp > 101.5: risk_score += 2
    if chest_pain >= 4: risk_score += 4
    if cardiac_hist: risk_score += 2
    if diabetes: risk_score += 1
    if chronic: risk_score += 1
    
    if risk_score >= 6: risk_level = 0
    elif risk_score >= 3: risk_level = 1
    else: risk_level = 2
    
    complaint = features['chief_complaint'].lower()
    dept_scores = [0.1] * 6
    
    if chest_pain >= 3 or 'chest' in complaint or cardiac_hist:
        dept_scores[0] = 0.95; dept_scores[1] = 0.90
    if 'headache' in complaint or 'dizzy' in complaint:
        dept_scores[3] = 0.88
    if temp > 101 or respiratory_hist:
        dept_scores[2] = 0.75; dept_scores[4] = 0.78
    if 'abdominal' in complaint:
        dept_scores[4] = 0.85
    
    return {'risk_level': risk_level, 'dept_scores': dept_scores}

def main():
    os.makedirs("../data", exist_ok=True)
    
    data = []
    for i in range(1, 501):
        features = generate_synthetic_visit(i)
        labels = generate_labels(features)
        features.update(labels)
        data.append(features)
        if i % 100 == 0:
            print(f"Generated {i}/500")
    
    df = pd.DataFrame(data)
    df.to_csv('../data/train.csv', index=False)
    print(f"\n✅ SAVED 500 FULL samples to data/train.csv")
    print("History:", df[['cardiac_history','diabetes_status','respiratory_history']].sum().to_dict())
    print("Risk:", dict(df['risk_level'].value_counts().sort_index()))

if __name__ == "__main__":
    main()
