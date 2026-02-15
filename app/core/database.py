from supabase import create_client, Client
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class Database:
    _client: Client = None
    
    @classmethod
    def get_client(cls) -> Client:
        if cls._client is None:
            cls._client = create_client(settings.supabase_url, settings.supabase_key)
            logger.info("Supabase client initialized")
        return cls._client
    
    @classmethod
    def get_visit_features(cls, visit_id: int) -> dict:
        """REAL DB queries - fetch actual visit data."""
        client = cls.get_client()
        
        # Get visit basics
        response = client.table('patient_visits').select('visit_id, chief_complaint, patient_id').eq('visit_id', visit_id).single().execute()
        visit = response.data
        
        if not visit:
            return {
                'visit_id': visit_id, 'age': 40, 'heart_rate': 80, 'bp_systolic': 120,
                'bp_diastolic': 80, 'temperature': 98.6, 'chest_pain_severity': 0,
                'max_severity': 1, 'symptom_count': 1, 'chief_complaint': 'general'
            }
        
        # Patient info
        patient_response = client.table('patients').select('age, gender').eq('patient_id', visit['patient_id']).single().execute()
        patient = patient_response.data or {}
        
        # Vitals
        vitals_response = client.table('vitals').select('bp_systolic, bp_diastolic, heart_rate, temperature').eq('visit_id', visit_id).execute()
        vitals = vitals_response.data[0] if vitals_response.data else {}
        
        # Symptoms
        symptoms_response = client.table('visit_symptoms').select('symptom_name, severity_score').eq('visit_id', visit_id).execute()
        symptoms = symptoms_response.data
        chest_pain_sev = max([s['severity_score'] for s in symptoms if 'chest' in s['symptom_name'].lower()] or [0])
        max_sev = max([s['severity_score'] for s in symptoms] or [1])
        symptom_count = len(symptoms) or 1
        
        return {
            'visit_id': visit_id,
            'age': patient.get('age', 40),
            'gender': patient.get('gender', 'M'),
            'bp_systolic': vitals.get('bp_systolic', 120),
            'bp_diastolic': vitals.get('bp_diastolic', 80),
            'heart_rate': vitals.get('heart_rate', 80),
            'temperature': float(vitals.get('temperature', 98.6)),
            'chief_complaint': visit.get('chief_complaint', 'general'),
            'chest_pain_severity': chest_pain_sev,
            'max_severity': max_sev,
            'symptom_count': symptom_count
        }

    @classmethod
    def save_prediction(cls, visit_id: int, prediction: dict) -> dict:
        """Save ML prediction to Supabase."""
        client = cls.get_client()
        
        data = {
            'visit_id': visit_id,
            'risk_level': prediction['risk_level'],
            'risk_score': prediction['risk_score'],
            'recommended_department': prediction['recommended_department'],
            'department_scores': prediction['department_scores'],
            'explainability': prediction['explainability']
        }
        
        try:
            response = client.table('triage_predictions').insert(data).execute()
            if response.data:
                logger.info(f"✅ Saved prediction for visit {visit_id}")
                return response.data[0]
            else:
                logger.warning(f"⚠️ Failed to save prediction for visit {visit_id}")
                return {}
        except Exception as e:
            logger.error(f"❌ DB Insert Error: {e}")
            return {}
