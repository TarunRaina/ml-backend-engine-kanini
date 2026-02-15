import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import os

# Exact feature order (matches your ml_engine.py)
FEATURES = [
    'age', 'bp_systolic', 'bp_diastolic', 'heart_rate', 'temperature',
    'chest_pain_severity', 'max_severity', 'symptom_count',
    'comorbidities_count', 'cardiac_history', 'diabetes_status', 
    'respiratory_history', 'chronic_conditions'
]

DEPARTMENTS = ["emergency", "cardiology", "respiratory", "neurology", "general_medicine", "orthopedics"]

def train_model():
    os.makedirs("../app/models", exist_ok=True)
    
    # Load your data
    df = pd.read_csv('../data/train.csv')
    print(f"✅ Loaded {len(df)} training samples")
    
    # Prepare features (EXACT order for ml_engine.py)
    X = df[FEATURES]
    y_risk = df['risk_level']
    
    # Department scores (6 outputs)
    y_dept = np.array(df['dept_scores'].apply(eval).tolist())
    
    # Split
    X_train, X_test, y_risk_train, y_risk_test, y_dept_train, y_dept_test = train_test_split(
        X, y_risk, y_dept, test_size=0.2, random_state=42
    )
    
    # 1. RISK CLASSIFIER (0=high, 1=medium, 2=low)
    print("🚀 Training Risk Classifier...")
    risk_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
    )
    risk_model.fit(X_train, y_risk_train)
    
    # 2. DEPARTMENT REGRESSOR (6 scores)
    print("🚀 Training Department Recommender...")
    dept_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    dept_model.fit(X_train, y_dept_train)
    
    # 3. SHAP EXPLAINERS
    print("🔍 Creating SHAP explainers...")
    risk_explainer = shap.TreeExplainer(risk_model)
    dept_explainer = shap.TreeExplainer(dept_model)
    
    # Model accuracy
    risk_acc = risk_model.score(X_test, y_risk_test)
    print(f"✅ Risk accuracy: {risk_acc:.1%}")
    
    # Save EVERYTHING ml_engine.py expects
    joblib.dump({
        'risk_model': risk_model,
        'dept_model': dept_model,
        'risk_explainer': risk_explainer,
        'dept_explainer': dept_explainer,
        'features': FEATURES
    }, '../app/models/trained_model.joblib')
    
    print("🎉 XGBoost + SHAP SAVED to app/models/trained_model.joblib")
    print("✅ ml_engine.py will auto-detect and use it!")

if __name__ == "__main__":
    train_model()
