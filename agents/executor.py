import subprocess
import os
import logging
from languages.manager import LanguageManager

class ExecutorAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.language_manager = LanguageManager()
    
    async def process(self, user_input: str, context: dict) -> dict:
        try:
            filepath = self._extract_filepath(user_input, context)
            
            if not filepath or not os.path.exists(filepath):
                return {"error": f"File not found: {filepath}"}
            
            # Get appropriate language executor
            language = self.language_manager.get_language(filepath)
            result = language.execute(filepath)
            
            return {
                "success": result["success"],
                "output": result["stdout"],
                "error": result["stderr"],
                "filepath": filepath
            }
        except Exception as e:
            self.logger.error(f"Executor agent failed: {e}")
            return {"error": str(e)}
    
    def _extract_filepath(self, user_input: str, context: dict) -> str:
        words = user_input.split()
        for word in words:
            if word.endswith(('.py', '.js', '.java', '.go', '.ts')):
                return word
        return context.get('filepath', '')