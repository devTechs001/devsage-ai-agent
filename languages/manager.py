import os
from typing import Dict
from languages.base import BaseLanguage
from languages.python.executor import PythonLanguage

class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            # Add more languages as they are implemented
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")