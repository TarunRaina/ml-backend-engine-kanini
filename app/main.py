"""FastAPI ML Backend - Main entrypoint."""
from fastapi import FastAPI
from app.api.v1 import ml
from app.core.config import settings
import logging

# Configure logging (uses config.py settings)
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Triage Backend",
    description="XGBoost + SHAP patient triage engine",
    version="1.0.0"
)

# Include API routes
app.include_router(ml.router, prefix="/api/v1", tags=["ml"])

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ML Triage Backend",
        "model_loaded": "app/models/trained_model.joblib" in settings.model_path,
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Production health check."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
