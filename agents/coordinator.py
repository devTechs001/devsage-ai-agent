from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from datetime import datetime
from typing import Dict, Any

from core.config import settings
from core.logger import setup_logging
from agents.coordinator import EnhancedCoordinatorAgent

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting DevSage API")
    yield
    # Shutdown
    logger.info("Shutting down DevSage API")

app = FastAPI(
    title="DevSage API",
    description="Full-stack AI coding assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1"]
)

# Initialize coordinator
coordinator = EnhancedCoordinatorAgent()

# Pydantic models
class AgentRequest:
    prompt: str
    context: Dict[str, Any] = {}

class AgentResponse:
    success: bool
    result: Any = None
    error: str = None
    timestamp: datetime

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic
    # For now, simple token validation
    if not credentials.credentials.startswith("devsage_"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@app.post("/v1/agent/query")
async def query_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
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
        logger.error(f"Agent query failed: {e}")
        return AgentResponse(
            success=False,
            error=str(e),
            timestamp=datetime.utcnow()
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "DevSage API is running"}