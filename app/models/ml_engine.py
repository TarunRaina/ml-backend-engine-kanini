import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from typing import Dict, Any, Tuple
import logging
import os

# FEATURES (EXACT ORDER FROM TRAINING)
FEATURES = [
    'age', 'bp_systolic', 'bp_diastolic', 'heart_rate', 'temperature',
    'chest_pain_severity', 'max_severity', 'symptom_count', 'comorbidities_count',
    'cardiac_history', 'diabetes_status', 'respiratory_history', 'chronic_conditions'
]

class MLEngine:
    def __init__(self, model_dir: str = "app/models"):
        """Load production models + SHAP explainers"""
        self.model_data = self._load_models(model_dir)
        logging.info("✅ MLEngine loaded: Risk & Single-Label Dept Model + SHAP")
    
    def _load_models(self, model_dir: str) -> Dict[str, Any]:
        """Load risk model, dept model, encoders, SHAP explainers"""
        try:
            # Construct absolute paths if needed
            if not os.path.isabs(model_dir):
                 current_dir = os.path.dirname(os.path.abspath(__file__))
                 if os.path.basename(current_dir) == 'models':
                     model_dir = current_dir
                 else:
                     model_dir = os.path.join(current_dir, "../../", model_dir)
            
            # Fallback
            if not os.path.exists(os.path.join(model_dir, "risk_model.joblib")):
                potential_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'risk_model.joblib')
                if os.path.exists(potential_path):
                    model_dir = os.path.dirname(os.path.abspath(__file__))

            logging.info(f"Loading models from: {model_dir}")

            risk_model = joblib.load(os.path.join(model_dir, "risk_model.joblib"))
            dept_model = joblib.load(os.path.join(model_dir, "dept_model.joblib")) # Single model
            risk_encoder = joblib.load(os.path.join(model_dir, "risk_encoder.joblib"))
            dept_encoder = joblib.load(os.path.join(model_dir, "dept_encoder.joblib")) # Single encoder
            
            # SHAP explainer
            risk_explainer = shap.TreeExplainer(risk_model)
            
            return {
                'risk_model': risk_model,
                'dept_model': dept_model,
                'risk_encoder': risk_encoder,
                'dept_encoder': dept_encoder,
                'risk_explainer': risk_explainer,
                'features': FEATURES
            }
        except Exception as e:
            logging.error(f"❌ Model load failed: {e}")
            raise
    
    def preprocess_input(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert raw input → model-ready DataFrame"""
        data = {feature: 0 for feature in FEATURES}
        for key, val in patient_data.items():
            if key in data:
                data[key] = val
        return pd.DataFrame([data], columns=FEATURES)
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """FULL PRODUCTION PIPELINE: risk + single-label dept probas"""
        X = self.preprocess_input(patient_data)
        
        # 1. RISK TRIAGE
        risk_pred_idx = self.model_data['risk_model'].predict(X)[0]
        risk_proba = self.model_data['risk_model'].predict_proba(X)[0]
        
        risk_level = self.model_data['risk_encoder'].inverse_transform([risk_pred_idx])[0]
        risk_score = float(np.max(risk_proba))

        # 2. DEPARTMENT SCORES (Single Label Multi-Class)
        dept_pred_idx = self.model_data['dept_model'].predict(X)[0]
        dept_probas = self.model_data['dept_model'].predict_proba(X)[0]
        recommended_dept = self.model_data['dept_encoder'].inverse_transform([dept_pred_idx])[0]

        # Map all department scores
        dept_classes = self.model_data['dept_encoder'].classes_
        dept_scores = {
            str(cls): float(prob) 
            for cls, prob in zip(dept_classes, dept_probas)
        }
        
        # 3. EXPLAINABILITY
        explainability = self._real_shap_explanation(X)
        
        return {
            'risk_level': str(risk_level),
            'risk_score': round(risk_score, 4),
            'recommended_department': str(recommended_dept),
            'department_scores': dept_scores,
            'explainability': explainability
        }
    
    def _real_shap_explanation(self, X: pd.DataFrame) -> Dict[str, float]:
        """Generate SHAP values for the prediction"""
        try:
            explainer = self.model_data['risk_explainer']
            shap_values = explainer(X)
            
            if hasattr(shap_values, 'values'):
                vals = shap_values.values
            else:
                vals = shap_values
                
            if len(vals.shape) == 3:
                vals = np.abs(vals[0]).sum(axis=1) 
            elif len(vals.shape) == 2:
                vals = np.abs(vals[0])
            else:
                vals = np.abs(vals)

            top_indices = np.argsort(vals)[-5:][::-1]
            
            explainability = {
                FEATURES[i]: float(vals[i]) 
                for i in top_indices
            }
            return {k: round(v, 4) for k, v in explainability.items()}
            
        except Exception as e:
            logging.error(f"❌ SHAP CRASH: {e}")
            return {}

# TEST IT
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        engine = MLEngine(model_dir=current_dir)
        
        test_patient = {
            'visit_id': 3, 'age': 36, 'bp_systolic': 173, 'bp_diastolic': 100,
            'heart_rate': 128, 'temperature': 103.6, 'chest_pain_severity': 5,
            'max_severity': 2, 'symptom_count': 4, 'comorbidities_count': 0,
            'cardiac_history': 0, 'diabetes_status': 0, 'respiratory_history': 0,
            'chronic_conditions': 0
        }
        
        result = engine.predict(test_patient)
        print("\n🎯 PREDICTION RESULT:")
        import json
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Test failed: {e}")
