# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.# 🚀 DevSage: Complete Full-Stack AI Coding Agent Documentation

## 📋 Table of Contents
1. [Enhanced Project Structure](#-enhanced-project-structure)
2. [Comprehensive Setup Guide](#-comprehensive-setup-guide)
3. [Multi-Language Support Architecture](#-multi-language-support-architecture)
4. [Advanced Agent System](#-advanced-agent-system)
5. [Enhanced Backend API](#-enhanced-backend-api)
6. [IDE Integration Best Practices](#-ide-integration-best-practices)
7. [Development Best Practices](#-development-best-practices)
8. [Additional Features](#-additional-features)
9. [Deployment & Scaling](#-deployment--scaling)

---

## 🏗️ Enhanced Project Structure

```
devsage/
├── agents/                          # Core AI Agents
│   ├── reader.py
│   ├── writer.py
│   ├── editor.py
│   ├── executor.py
│   ├── memory.py
│   ├── coordinator.py
│   ├── analyzer.py                  # NEW: Code analysis agent
│   ├── debugger.py                  # NEW: Debugging assistant
│   └── security.py                  # NEW: Security scanner
├── api/                             # Backend Services
│   ├── main.py
│   ├── models.py                    # Pydantic models
│   ├── routes/                      # Modular routes
│   │   ├── agents.py
│   │   ├── files.py
│   │   └── projects.py
│   └── middleware/                  # Auth & security
├── core/                            # Core Utilities
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging
│   ├── security.py                  # Security utilities
│   └── exceptions.py                # Custom exceptions
├── languages/                       # Multi-language Support
│   ├── python/
│   ├── javascript/
│   ├── java/
│   ├── go/
│   └── base.py                     # Base language class
├── vectorstore/                     # Enhanced Semantic Search
│   ├── index.py
│   ├── embeddings.py
│   └── chunking.py
├── gui/                            # Electron Desktop App
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── extensions/                     # IDE Extensions
│   ├── vscode/
│   ├── jupyter/
│   └── intellij/                   # NEW: IntelliJ support
├── docker/                         # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/
├── tests/                          # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
│   ├── setup/
│   ├── api/
│   └── examples/
└── scripts/                        # Development scripts
    ├── setup.sh
    ├── deploy.sh
    └── test.sh
```

---

## 🛠️ Comprehensive Setup Guide

### ✅ Prerequisites

```bash
# System Requirements
# - Python 3.10+
# - Node.js 18+
# - Docker & Docker Compose
# - Git

# Verify installations
python --version
node --version
docker --version
docker-compose --version
```

### ✅ Python Environment Setup

```bash
# Using conda (recommended)
conda create -n devsage python=3.10
conda activate devsage

# OR using pyenv + virtualenv
pyenv install 3.10.12
pyenv virtualenv 3.10.12 devsage
pyenv activate devsage

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### ✅ Enhanced Requirements.txt

```txt
# Core AI & ML
langchain==0.0.354
langchain-community==0.0.10
openai==1.3.0
ollama==0.1.7
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
watchdog==3.0.0

# Code Analysis
tree-sitter==0.20.4
black==23.11.0
ruff==0.1.6
pylint==3.0.2

# Security
bandit==1.7.5
safety==2.3.5

# Database & Storage
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0
```

### ✅ Node.js Environment Setup

```bash
# Install Node.js dependencies
cd gui && npm install

# Install VS Code extension dependencies
cd extensions/vscode && npm install

# Install JupyterLab extension
cd extensions/jupyter && npm install
jupyter labextension install .

# Install IntelliJ extension (if applicable)
cd extensions/intellij && ./gradlew build
```

### ✅ Ollama Setup with Multiple Models

```bash
# Pull various models for different tasks
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
ollama pull llama2:7b
ollama pull mistral:7b

# Verify models
ollama list

# Start Ollama service
ollama serve
```

### ✅ Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 devsage
USER devsage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  devsage:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
```

---

## 🌐 Multi-Language Support Architecture

### Base Language Class

```python
# languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class BaseLanguage(ABC):
    def __init__(self, name: str, extensions: List[str]):
        self.name = name
        self.extensions = extensions
    
    @abstractmethod
    def execute(self, filepath: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def format_code(self, code: str) -> str:
        pass
    
    @abstractmethod
    def get_dependencies(self, filepath: str) -> List[str]:
        pass

# languages/python.py
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
    
    def lint(self, filepath: str) -> List[Dict[str, Any]]:
        # Implement pylint/ruff integration
        pass
    
    def format_code(self, code: str) -> str:
        # Implement black integration
        pass

# languages/javascript.py
class JavaScriptLanguage(BaseLanguage):
    def __init__(self):
        super().__init__("javascript", [".js", ".jsx", ".ts", ".tsx"])
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        if filepath.endswith(('.ts', '.tsx')):
            # Compile TypeScript first
            compile_result = subprocess.run(
                ["npx", "tsc", filepath],
                capture_output=True,
                text=True
            )
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
        
        result = subprocess.run(
            ["node", filepath],
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

# Language Manager
class LanguageManager:
    def __init__(self):
        self.languages: Dict[str, BaseLanguage] = {
            "python": PythonLanguage(),
            "javascript": JavaScriptLanguage(),
            # Add more languages...
        }
    
    def get_language(self, filepath: str) -> BaseLanguage:
        extension = os.path.splitext(filepath)[1].lower()
        for lang in self.languages.values():
            if extension in lang.extensions:
                return lang
        raise ValueError(f"Unsupported language for file: {filepath}")
```

---

## 🤖 Advanced Agent System

### Enhanced Coordinator Agent

```python
# agents/coordinator.py
from typing import Dict, Any, List
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class EnhancedCoordinatorAgent:
    def __init__(self):
        self.agents = {
            "reader": ReaderAgent(),
            "writer": WriterAgent(),
            "editor": EditorAgent(),
            "executor": ExecutorAgent(),
            "memory": MemoryAgent(),
            "analyzer": CodeAnalyzerAgent(),
            "debugger": DebuggerAgent(),
            "security": SecurityAgent()
        }
        self.language_manager = LanguageManager()
        self.logger = logging.getLogger(__name__)
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
            Classify the user's intent into one of these categories:
            - read_file: Reading or understanding code
            - write_file: Creating new code files
            - edit_file: Modifying existing code
            - execute_code: Running code
            - analyze_code: Code analysis, linting, or review
            - debug_code: Debugging assistance
            - security_scan: Security analysis
            - memory_recall: Recalling past conversations
            
            User Input: {user_input}
            Context: {context}
            
            Respond ONLY with the intent category name.
            """
        )
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any]) -> str:
        """Use LLM to classify user intent"""
        try:
            # You can replace this with your preferred LLM
            chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
            result = await chain.arun(user_input=user_input, context=context)
            return result.strip().lower()
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "unknown"
    
    async def route_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced routing with intent classification"""
        if context is None:
            context = {}
        
        intent = await self.classify_intent(user_input, context)
        self.logger.info(f"Classified intent: {intent}")
        
        # Route to appropriate agent
        if intent == "read_file":
            return await self.agents["reader"].process(user_input, context)
        elif intent == "write_file":
            return await self.agents["writer"].process(user_input, context)
        elif intent == "edit_file":
            return await self.agents["editor"].process(user_input, context)
        elif intent == "execute_code":
            return await self.agents["executor"].process(user_input, context)
        elif intent == "analyze_code":
            return await self.agents["analyzer"].process(user_input, context)
        elif intent == "debug_code":
            return await self.agents["debugger"].process(user_input, context)
        elif intent == "security_scan":
            return await self.agents["security"].process(user_input, context)
        else:
            return {"error": f"Unknown intent: {intent}", "suggestions": self.get_suggestions(user_input)}
```

### Code Analyzer Agent

```python
# agents/analyzer.py
import ast
import astroid
from pylint import epylint as lint

class CodeAnalyzerAgent:
    def __init__(self):
        self.supported_analyses = ["complexity", "quality", "performance", "maintainability"]
    
    async def analyze_code(self, filepath: str, analysis_type: str = "quality") -> Dict[str, Any]:
        if analysis_type not in self.supported_analyses:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if analysis_type == "quality":
                return await self._analyze_quality(code, filepath)
            elif analysis_type == "complexity":
                return await self._analyze_complexity(code)
            elif analysis_type == "performance":
                return await self._analyze_performance(code)
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_quality(self, code: str, filepath: str) -> Dict[str, Any]:
        """Analyze code quality using pylint"""
        (pylint_stdout, pylint_stderr) = lint.py_run(
            f"{filepath} --output-format=json", return_std=True
        )
        
        # Parse pylint output and return structured results
        return {
            "score": self._calculate_quality_score(pylint_stdout),
            "issues": self._parse_pylint_issues(pylint_stdout),
            "suggestions": self._generate_suggestions(pylint_stdout)
        }
```

---

## 🚀 Enhanced Backend API

### FastAPI with Best Practices

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app, Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting DevSage API")
    yield
    # Shutdown
    logging.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "yourdomain.com"]
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Enhanced agent endpoint
@app.post("/v1/agent/query", response_model=AgentResponse)
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    REQUEST_COUNT.labels(method='POST', endpoint='/v1/agent/query').inc()
    
    with REQUEST_DURATION.time():
        try:
            result = await coordinator.route_request(
                user_input=request.prompt,
                context=request.context
            )
            
            # Store in memory as background task
            background_tasks.add_task(
                coordinator.agents["memory"].store,
                request.prompt,
                result
            )
            
            return AgentResponse(
                success=True,
                result=result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Agent query failed: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                timestamp=datetime.utcnow()
            )

# Health check with dependencies
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "ollama": await check_ollama_health(),
            "redis": await check_redis_health(),
            "database": await check_db_health()
        }
    }
    
    # If any dependency is down, mark as unhealthy
    if any(not status for status in health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 🔌 IDE Integration Best Practices

### VS Code Extension Enhancement

```javascript
// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import { DevSageClient } from './devsage-client';
import { CodeActionProvider } from './code-action-provider';

export function activate(context: vscode.ExtensionContext) {
    const client = new DevSageClient();
    const actionProvider = new CodeActionProvider(client);

    // Register commands
    const askCommand = vscode.commands.registerCommand('devsage.ask', async () => {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            const response = await client.query(prompt, {
                currentFile: vscode.window.activeTextEditor?.document.fileName,
                workspace: vscode.workspace.rootPath
            });
            
            vscode.window.showInformationMessage(response.result);
        }
    });

    // Code lens for intelligent suggestions
    const codeLensProvider = vscode.languages.registerCodeLensProvider('*', {
        provideCodeLenses: (document) => {
            return actionProvider.provideCodeLenses(document);
        }
    });

    // Register all commands
    context.subscriptions.push(askCommand, codeLensProvider);
}

class DevSageClient {
    private baseUrl: string;

    constructor() {
        this.baseUrl = vscode.workspace.getConfiguration('devsage').get('apiUrl') || 'http://localhost:8000';
    }

    async query(prompt: string, context: any): Promise<any> {
        try {
            const response = await fetch(`${this.baseUrl}/v1/agent/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getApiKey()}`
                },
                body: JSON.stringify({ prompt, context })
            });
            
            return await response.json();
        } catch (error) {
            vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
        }
    }

    private getApiKey(): string {
        return vscode.workspace.getConfiguration('devsage').get('apiKey') || '';
    }
}
```

---

## 🏆 Development Best Practices

### Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    
    # Database
    database_url: str = "sqlite:///./devsage.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # File paths
    workspace_dir: str = "./workspace"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Structured Logging

```python
# core/logger.py
import structlog
import logging
import sys

def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

# Usage
logger = structlog.get_logger()
```

### Error Handling

```python
# core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict

class DevSageException(HTTPException):
    """Base exception for DevSage"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}

class CodeExecutionException(DevSageException):
    """Raised when code execution fails"""
    pass

class SecurityException(DevSageException):
    """Raised for security violations"""
    pass

# Exception handler
@app.exception_handler(DevSageException)
async def devsage_exception_handler(request, exc: DevSageException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": exc.error_code,
            "metadata": exc.metadata
        }
    )
```

---

## 🎯 Additional Features

### 1. Real-time Collaboration

```python
# api/routes/collaboration.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time collaboration messages
            await manager.broadcast(f"Project {project_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 2. Project Templates

```python
# core/templates.py
PROJECT_TEMPLATES = {
    "python-fastapi": {
        "structure": {
            "app/__init__.py": "",
            "app/main.py": FASTAPI_BOILERPLATE,
            "requirements.txt": "fastapi\nuvicorn",
            "Dockerfile": DOCKERFILE_PYTHON
        },
        "dependencies": ["python", "fastapi"]
    },
    "react-typescript": {
        "structure": {
            "package.json": REACT_PACKAGE_JSON,
            "tsconfig.json": TSCONFIG_JSON,
            "src/App.tsx": REACT_APP_TSX
        },
        "dependencies": ["node", "typescript", "react"]
    }
}
```

### 3. Advanced Security Scanning

```python
# agents/security.py
import bandit
import safety
import ast

class SecurityAgent:
    async def scan_code(self, filepath: str) -> Dict[str, Any]:
        """Comprehensive security scan"""
        bandit_results = self._run_bandit(filepath)
        dependency_vulns = await self._check_dependencies(filepath)
        ast_analysis = self._analyze_ast(filepath)
        
        return {
            "bandit_issues": bandit_results,
            "dependency_vulnerabilities": dependency_vulns,
            "ast_analysis": ast_analysis,
            "security_score": self._calculate_security_score(bandit_results, dependency_vulns)
        }
```

---

## 🚀 Deployment & Scaling

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  devsage:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/devsage
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: devsage
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - devsage

volumes:
  ollama_data:
  redis_data:
  postgres_data:
```

### Monitoring & Observability

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
AGENT_REQUESTS = Counter('agent_requests_total', 'Agent requests by type', ['agent_type'])

# Middleware for metrics
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

---

## 📚 Recommended IDEs & Tools

### Primary Development Environment
- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - Ruff
  - Docker
  - GitLens

### Alternative IDEs
- **PyCharm Professional** - Excellent for Python development
- **Neovim** with LSP - For terminal-based development
- **JupyterLab** - For experimental and data science work

### Essential Development Tools
- **Docker & Docker Compose** - Containerization
- **Git** - Version control
- **Make** - Task automation
- **pre-commit** - Git hooks
- **pytest** - Testing framework
- **black & ruff** - Code formatting and linting

This comprehensive documentation provides a solid foundation for building, deploying, and scaling DevSage as a professional-grade AI coding assistant. The architecture supports multiple languages, includes robust security measures, and follows industry best practices for development and deployment.🔧 Key Directory Explanations:
agents/ - AI Agent System
Modular agents for different coding tasks

Coordinator for routing between agents

Specialized agents for reading, writing, executing code

api/ - Backend Services
FastAPI application with modular routes

WebSocket support for real-time features

Middleware for auth, logging, CORS

languages/ - Multi-language Support
Base class for language-agnostic operations

Language-specific implementations

Execution, analysis, formatting per language

gui/ - Desktop Application
Electron-based desktop app

React frontend with component architecture

State management and API integration

extensions/ - IDE Integrations
VS Code, Jupyter, IntelliJ extensions

Language server protocol implementations

IDE-specific functionality

tests/ - Comprehensive Testing
Unit tests for all components

Integration tests for workflows

Test fixtures and mock data

docker/ - Containerization
Multi-stage Docker builds

Development and production compose files

Nginx configuration for production

This structure supports:

Modular development - Work on one component without affecting others

Scalability - Easy to add new agents, languages, or features

Testing - Comprehensive test coverage

Deployment - Multiple deployment options

Documentation - Clear, organized documentation

Team collaboration - Clear separation of concerns

 Final Recommendation: USE ONLY VS CODE
Why VS Code Wins for DevSage:
✅ Single Environment - No context switching

✅ Best Extension Ecosystem - Everything you need

✅ Excellent Python Support - On par with PyCharm

✅ Superior JavaScript/TypeScript Support - Better than PyCharm

✅ Integrated Jupyter Support - No need for separate JupyterLab

✅ Docker Integration - Built-in container support

✅ Multi-Root Workspaces - Perfect for monorepo projects

✅ Lightweight - Uses fewer resources than multiple IDEs

✅ Consistent Experience - Same shortcuts, same workflow

✅ Team Friendly - Easy to share configurations

# One command to rule them all
git clone your-repo devsage
cd devsage
code devsage.code-workspace
# VS Code will prompt to install recommended extensions
# Then run in integrated terminal:
make install
make dev-fullstack


