"""
Virtual HR - Multi-Agent HR Automation System
FastAPI application with orchestrator integration.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, UTC
import uuid
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from agents.orchestrator.orchestrator import OrchestratorAgent

# Initialize FastAPI app
app = FastAPI(
    title="Virtual HR API",
    description="Multi-Agent HR Automation System with intelligent routing",
    version="1.0.0"
)

# Initialize orchestrator (lazy loaded on first request)
_orchestrator: Optional[OrchestratorAgent] = None


def get_orchestrator() -> OrchestratorAgent:
    """Get or create the orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator


# ============== Request/Response Models ==============

class ChatMessage(BaseModel):
    """Chat request model."""
    session_id: Optional[str] = Field(
        None, 
        description="Session ID for conversation tracking. Generated if not provided."
    )
    message: str = Field(..., description="The user's message")
    employee_id: Optional[str] = Field(
        None, 
        description="Employee ID for leave and HR operations"
    )
    employee_name: Optional[str] = Field(
        None, 
        description="Employee name"
    )
    is_hr: bool = Field(
        False, 
        description="Whether the user has HR privileges (for approvals/trends)"
    )


class ChatResponse(BaseModel):
    """Chat response model."""
    session_id: str
    response: str
    timestamp: str


class ConversationHistory(BaseModel):
    """Conversation history response model."""
    session_id: str
    messages: List[Dict[str, Any]]


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    message: str
    components: Dict[str, str]
    timestamp: str


# ============== Endpoints ==============

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def send_message(chat: ChatMessage):
    """
    Send a message to the Virtual HR assistant.
    
    The intelligent orchestrator will route your request to the appropriate agent:
    - **Policy questions** → RAG Agent (answers from company documents)
    - **Leave requests** → Leave Tracker Agent (manages leave via Google Sheets)
    - **Feedback** → Feedback Agent (anonymous feedback with sentiment analysis)
    
    **Examples:**
    - "What is the leave policy for marriage?"
    - "I want to apply for 2 days annual leave starting 2026-01-15"
    - "Check my leave balance. My employee ID is 123"
    - "I want to give feedback: The new office is great!"
    
    For HR operations (approvals, trends), set `is_hr=true`.
    """
    # Create or use existing session
    session_id = chat.session_id or str(uuid.uuid4())
    
    # Build user context
    user_context = {
        "employee_id": chat.employee_id,
        "employee_name": chat.employee_name,
        "is_hr": chat.is_hr
    }
    
    try:
        orchestrator = get_orchestrator()
        response = orchestrator.chat(
            message=chat.message,
            session_id=session_id,
            user_context=user_context
        )
        
        return ChatResponse(
            session_id=session_id,
            response=response,
            timestamp=datetime.now(UTC).isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@app.get("/chat/history/{session_id}", response_model=ConversationHistory, tags=["Chat"])
async def get_history(session_id: str):
    """
    Retrieve conversation history for a given session.
    """
    orchestrator = get_orchestrator()
    history = orchestrator.get_conversation_history(session_id)
    
    if not history:
        raise HTTPException(
            status_code=404, 
            detail="Session not found or no history available"
        )
    
    return ConversationHistory(
        session_id=session_id,
        messages=history
    )


@app.delete("/chat/history/{session_id}", tags=["Chat"])
async def clear_history(session_id: str):
    """
    Clear conversation history for a session.
    """
    orchestrator = get_orchestrator()
    orchestrator.clear_conversation(session_id)
    
    return {"message": f"Conversation history cleared for session {session_id}"}


@app.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check():
    """
    Health check endpoint to verify system status.
    """
    components = {}
    overall_status = "healthy"
    
    # Check configuration
    missing_config = Config.validate()
    if missing_config:
        components["config"] = f"missing: {', '.join(missing_config)}"
        overall_status = "degraded"
    else:
        components["config"] = "ok"
    
    # Check OpenAI connectivity
    try:
        from openai import OpenAI
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        # Just verify the client can be created
        components["openai"] = "ok"
    except Exception as e:
        components["openai"] = f"error: {str(e)}"
        overall_status = "degraded"
    
    # Check Google Sheets credentials
    if os.path.exists(Config.GOOGLE_SHEETS_CREDENTIALS_FILE):
        components["google_sheets"] = "credentials found"
    else:
        components["google_sheets"] = "credentials not found"
        overall_status = "degraded"
    
    return HealthStatus(
        status=overall_status,
        message="Virtual HR API is running" if overall_status == "healthy" else "Some components need attention",
        components=components,
        timestamp=datetime.now(UTC).isoformat()
    )


@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Virtual HR API",
        "version": "1.0.0",
        "description": "Multi-Agent HR Automation System",
        "docs": "/docs",
        "health": "/health"
    }


# ============== Startup/Shutdown Events ==============

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    print("🚀 Virtual HR API starting up...")
    
    # Validate configuration
    missing = Config.validate()
    if missing:
        print(f"⚠️  Missing configuration: {', '.join(missing)}")
    else:
        print("✅ Configuration validated")
    
    print("✅ Virtual HR API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 Virtual HR API shutting down...")


# ============== Run with uvicorn ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)