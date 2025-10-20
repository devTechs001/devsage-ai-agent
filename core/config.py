# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # AI Configuration
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama2"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    redis_url: str = "redis://localhost:6379"
    
    # Security
    secret_key: str = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    
    # Paths
    data_dir: str = "./data"
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

def get_settings() -> Settings:
    return Settings()