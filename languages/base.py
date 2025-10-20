from abc import ABC, abstractmethod
from typing import Dict, Any, List
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        return []  # Default implementation
    
    def format_code(self, code: str) -> str:
        return code  # Default implementation
    
    def get_dependencies(self, filepath: str) -> List[str]:
        return []  # Default implementation