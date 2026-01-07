from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import uuid

app = FastAPI(title="Chat API", version="1.0.0")

# In-memory storage (use a database in production)
conversations: Dict[str, List[Dict]] = {}

# Request/Response models
class ChatMessage(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    timestamp: str

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict]

# Endpoints
@app.post("/chat", response_model=ChatResponse)
async def send_message(chat: ChatMessage):
    """
    Send a message and get a response.
    Creates a new session if session_id is not provided.
    """
    # Create or use existing session
    session_id = chat.session_id or str(uuid.uuid4())
    
    # Initialize conversation history if new session
    if session_id not in conversations:
        conversations[session_id] = []
    
    # Store user message
    user_msg = {
        "role": "user",
        "content": chat.message,
        "timestamp": datetime.utcnow().isoformat()
    }
    conversations[session_id].append(user_msg)
    
    # Generate response (replace with your actual logic)
    bot_response = f"Echo: {chat.message}"
    
    # Store bot response
    bot_msg = {
        "role": "assistant",
        "content": bot_response,
        "timestamp": datetime.utcnow().isoformat()
    }
    conversations[session_id].append(bot_msg)
    
    return ChatResponse(
        session_id=session_id,
        response=bot_response,
        timestamp=bot_msg["timestamp"]
    )

@app.get("/chat/history/{session_id}", response_model=ConversationHistory)
async def get_history(session_id: str):
    """
    Retrieve conversation history for a given session.
    """
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return ConversationHistory(
        session_id=session_id,
        messages=conversations[session_id]
    )

# Optional: Health check endpoint
@app.get("/")
async def root():
    return {"message": "Chat API is running", "status": "healthy"}