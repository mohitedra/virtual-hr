"""
Centralized configuration management for the Virtual HR system.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the Virtual HR multi-agent system."""
    
    # OpenAI Configuration (for embeddings and function calling)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    
    # Anthropic/Claude Configuration (for response generation)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    
    # Milvus Configuration
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "hr_policy_rag")
    
    # Google Sheets Configuration
    GOOGLE_SHEETS_CREDENTIALS_FILE = os.getenv(
        "GOOGLE_SHEETS_CREDENTIALS_FILE", 
        "credentials.json"
    )
    LEAVE_TRACKER_SHEET_ID = os.getenv("LEAVE_TRACKER_SHEET_ID")
    FEEDBACK_TRACKER_SHEET_ID = os.getenv("FEEDBACK_TRACKER_SHEET_ID")
    
    # Leave Policies (Default values)
    DEFAULT_ANNUAL_LEAVE_BALANCE = int(os.getenv("DEFAULT_ANNUAL_LEAVE_BALANCE", "20"))
    DEFAULT_SICK_LEAVE_BALANCE = int(os.getenv("DEFAULT_SICK_LEAVE_BALANCE", "10"))
    DEFAULT_PERSONAL_LEAVE_BALANCE = int(os.getenv("DEFAULT_PERSONAL_LEAVE_BALANCE", "5"))
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration. Returns list of missing configs."""
        missing = []
        
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.LEAVE_TRACKER_SHEET_ID:
            missing.append("LEAVE_TRACKER_SHEET_ID")
        if not cls.FEEDBACK_TRACKER_SHEET_ID:
            missing.append("FEEDBACK_TRACKER_SHEET_ID")
            
        return missing
    
    @classmethod
    def get_leave_balance_defaults(cls) -> dict:
        """Get default leave balances by type."""
        return {
            "Annual": cls.DEFAULT_ANNUAL_LEAVE_BALANCE,
            "Sick": cls.DEFAULT_SICK_LEAVE_BALANCE,
            "Personal": cls.DEFAULT_PERSONAL_LEAVE_BALANCE,
            "Maternity": 90,  # Fixed by policy
            "Paternity": 15,  # Fixed by policy
            "Marriage": 5,   # Fixed by policy
        }
