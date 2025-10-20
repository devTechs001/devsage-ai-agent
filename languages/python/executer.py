from languages.base import BaseLanguage
import subprocess
from typing import Dict, Any

class PythonLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("python", [".py", ".pyw"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                ["python", filepath],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timeout",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }