from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str = "devsage_secret_key_change_in_production"
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"

settings = Settings()