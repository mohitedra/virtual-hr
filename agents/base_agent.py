"""
Base agent interface for the Virtual HR multi-agent system.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentResponse:
    """Standard response format for all agents."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    agent_name: str = ""
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "agent_name": self.agent_name
        }


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.
    
    Each agent must implement the `handle` method to process
    queries specific to their domain.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def handle(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process a query and return a response.
        
        Args:
            query: The user's query or instruction
            context: Optional context including conversation history,
                    user info, extracted parameters, etc.
                    
        Returns:
            AgentResponse with the result
        """
        pass
    
    def _create_response(
        self, 
        success: bool, 
        message: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Helper to create standardized responses."""
        return AgentResponse(
            success=success,
            message=message,
            data=data,
            agent_name=self.name
        )
