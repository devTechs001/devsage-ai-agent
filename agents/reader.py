import os
import logging

class ReaderAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def process(self, user_input: str, context: dict) -> dict:
        try:
            # Extract filepath from user input or context
            filepath = self._extract_filepath(user_input, context)
            
            if not filepath or not os.path.exists(filepath):
                return {"error": f"File not found: {filepath}"}
            
            content = self.read_file(filepath)
            summary = await self.summarize_content(content)
            
            return {
                "filepath": filepath,
                "content": content,
                "summary": summary,
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Reader agent failed: {e}")
            return {"error": str(e)}
    
    def read_file(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    async def summarize_content(self, content: str) -> str:
        # Simple summary for now - can be enhanced with AI
        lines = content.split('\n')
        return f"File contains {len(lines)} lines of code."
    
    def _extract_filepath(self, user_input: str, context: dict) -> str:
        # Simple extraction logic
        words = user_input.split()
        for word in words:
            if word.endswith(('.py', '.js', '.java', '.go', '.ts')):
                return word
        return context.get('filepath', '')