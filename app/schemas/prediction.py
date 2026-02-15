"""Pydantic schemas for ML predictions."""
from pydantic import BaseModel
from typing import Dict, Any

class ProcessVisitRequest(BaseModel):
    visit_id: int

class ProcessVisitResponse(BaseModel):
    visit_id: int
    risk_level: str  # "high", "medium", "low"
    risk_score: float  # 0.0-1.0
    recommended_department: str
    department_scores: Dict[str, float]
    explainability: Dict[str, float]  # Top 5 SHAP values
