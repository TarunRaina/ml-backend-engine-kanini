"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import logging

class Settings(BaseSettings):
    supabase_url: str
    supabase_key: str
    model_path: str = "app/models/trained_model.joblib"
    debug: bool = False
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

# Global settings + standard logging setup
settings = Settings()

# Standard Python logging (NO loguru)
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("⚙️ Configuration loaded successfully")
