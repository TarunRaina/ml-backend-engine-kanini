from pydantic import BaseModel
from typing import Optional

class ProcessVisitRequest(BaseModel):
    """API Input: POST /ml/process_visit"""
    visit_id: int

class VisitCreate(BaseModel):
    """Patient visit data structure"""
    patient_age: Optional[int] = None
    chief_complaint: Optional[str] = None
    vitals_bp: Optional[str] = None
    vitals_pr: Optional[int] = None
    vitals_rr: Optional[int] = None
    vitals_temp: Optional[float] = None
