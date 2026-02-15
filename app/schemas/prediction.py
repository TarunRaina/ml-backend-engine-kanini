"""Pydantic schemas for ML predictions."""
from pydantic import BaseModel
from typing import Dict, List

class PredictionRequest(BaseModel):
    visit_id: int

class PredictionResponse(BaseModel):
    risk_level: str
    risk_score: float
    recommended_departments: List[str]  # Multi-label: all departments above threshold
    primary_department: str  # Highest scoring department
    department_scores: Dict[str, float]
    explainability: Dict[str, float]  # Top 5 SHAP values
