"""
Main Orchestrator Agent for the Virtual HR system.
Uses OpenAI function calling to intelligently route requests to specialized sub-agents.
"""
import os
import sys
from typing import Dict, Any, Optional, List
import json

# Add parent path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.base_agent import BaseAgent, AgentResponse
from config import Config
from openai import OpenAI
from anthropic import Anthropic


class OrchestratorAgent:
    """
    Central orchestrator that manages conversation flow and routes
    requests to specialized sub-agents using OpenAI function calling.
    
    Sub-agents:
    - RAG Agent: Answers policy questions using document retrieval
    - Leave Tracker Agent: Manages leave requests, balances, approvals
    - Feedback Agent: Collects and analyzes anonymous feedback
    """
    
    # Function definitions for OpenAI function calling
    ROUTING_FUNCTIONS = [
        {
            "name": "handle_policy_question",
            "description": "Answer questions about company policies, HR rules, leave policies, "
                          "workplace conduct, harassment policies, travel policies, benefits, "
                          "holidays, referrals, or any other company documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The policy-related question to answer"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "handle_leave_management",
            "description": "Handle leave-related requests: submit leave applications, "
                          "check leave balance, view leave history, or approve/reject leave "
                          "(HR only). Use this for any leave or time-off related queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["submit_leave", "check_balance", "view_history", "update_status"],
                        "description": "The leave action to perform"
                    },
                    "employee_id": {
                        "type": "string",
                        "description": "Employee ID (if mentioned)"
                    },
                    "employee_name": {
                        "type": "string",
                        "description": "Employee name (if mentioned)"
                    },
                    "leave_type": {
                        "type": "string",
                        "enum": ["Annual", "Sick", "Personal", "Maternity", "Paternity", "Marriage", "Bereavement"],
                        "description": "Type of leave"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Leave start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Leave end date in YYYY-MM-DD format"
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "Number of leave days"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for leave or for approval/rejection"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["Approved", "Rejected"],
                        "description": "New status for leave (HR approval/rejection)"
                    }
                },
                "required": ["action"]
            }
        },
        {
            "name": "handle_feedback",
            "description": "Handle employee feedback: collect anonymous feedback about "
                          "workplace, management, culture, facilities, or any concerns. "
                          "Also handles viewing feedback trends (HR only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["submit_feedback", "view_trends"],
                        "description": "The feedback action to perform"
                    },
                    "feedback_text": {
                        "type": "string",
                        "description": "The feedback content (for submission)"
                    }
                },
                "required": ["action"]
            }
        },
        {
            "name": "handle_general_query",
            "description": "Handle general conversation, greetings, or queries that don't "
                          "fit into policy, leave, or feedback categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The general query"
                    }
                },
                "required": ["query"]
            }
        }
    ]
    
    def __init__(self):
        """Initialize the orchestrator with all sub-agents."""
        # OpenAI for function calling (routing)
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Claude for general responses
        self.claude = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.claude_model = Config.CLAUDE_MODEL
        
        # Lazy load sub-agents to avoid circular imports
        self._rag_agent = None
        self._leave_agent = None
        self._feedback_agent = None
        
        # Conversation context storage (in-memory, use Redis/DB for production)
        self.conversations: Dict[str, List[Dict]] = {}
    
    @property
    def rag_agent(self):
        """Lazy load RAG agent."""
        if self._rag_agent is None:
            from agents.rag_agent.rag_agent import RAGAgent
            data_dir = os.path.join(os.path.dirname(__file__), "..", "rag_agent", "data")
            self._rag_agent = RAGAgent(
                milvus_host=Config.MILVUS_HOST,
                milvus_port=Config.MILVUS_PORT,
                collection_name=Config.MILVUS_COLLECTION_NAME,
                data_dir=data_dir
            )
        return self._rag_agent
    
    @property
    def leave_agent(self):
        """Lazy load Leave agent."""
        if self._leave_agent is None:
            from agents.leave_agent.leave_agent import LeaveTrackerAgent
            self._leave_agent = LeaveTrackerAgent()
        return self._leave_agent
    
    @property
    def feedback_agent(self):
        """Lazy load Feedback agent."""
        if self._feedback_agent is None:
            from agents.feedback_agent.feedback_agent import FeedbackAgent
            self._feedback_agent = FeedbackAgent()
        return self._feedback_agent
    
    def chat(
        self, 
        message: str, 
        session_id: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process a user message and return a response.
        
        Args:
            message: The user's message
            session_id: Unique session identifier for conversation tracking
            user_context: Optional context including:
                - employee_id: User's employee ID
                - employee_name: User's name
                - is_hr: Boolean indicating HR role
                
        Returns:
            The assistant's response
        """
        user_context = user_context or {}
        
        # Initialize conversation if new session
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # Add user message to history
        self.conversations[session_id].append({
            "role": "user",
            "content": message
        })
        
        # Determine which agent to route to using function calling
        routing_response = self._route_message(message, session_id)
        
        # Execute the appropriate agent
        if routing_response.get("function"):
            response = self._execute_function(
                routing_response["function"],
                routing_response.get("arguments", {}),
                message,
                user_context
            )
        else:
            # Fallback to general response
            response = routing_response.get("content", "I'm here to help with HR queries!")
        
        # Add assistant response to history
        self.conversations[session_id].append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _route_message(self, message: str, session_id: str) -> dict:
        """
        Use OpenAI function calling to determine routing.
        
        Returns:
            Dict with 'function' name and 'arguments', or 'content' for direct response
        """
        # Build messages with conversation history for context
        messages = [
            {
                "role": "system",
                "content": """You are a helpful HR assistant. Analyze the user's message and 
determine which function to call. Consider the conversation context when making decisions.

Key routing guidelines:
- Policy questions (what is, how does, explain) → handle_policy_question
- Leave requests, balance checks, approvals → handle_leave_management
- Feedback submission, trends → handle_feedback
- Greetings, general chat → handle_general_query

Extract all relevant parameters from the user's message. For dates, convert 
relative dates like 'tomorrow' or 'next Monday' to YYYY-MM-DD format 
(today is the current date based on the system)."""
            }
        ]
        
        # Add conversation history (last 5 exchanges for context)
        history = self.conversations.get(session_id, [])[-10:]
        messages.extend(history)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                tools=[{"type": "function", "function": f} for f in self.ROUTING_FUNCTIONS],
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            
            # Check if a function was called
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                return {
                    "function": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                }
            
            # No function call, return content directly
            return {"content": response_message.content or "How can I help you today?"}
            
        except Exception as e:
            print(f"Routing error: {e}")
            return {"content": "I'm having trouble understanding your request. Could you please rephrase?"}
    
    def _execute_function(
        self, 
        function_name: str, 
        arguments: dict,
        original_message: str,
        user_context: Dict[str, Any]
    ) -> str:
        """Execute the routed function and return response."""
        
        if function_name == "handle_policy_question":
            return self._handle_policy(arguments, original_message)
        
        elif function_name == "handle_leave_management":
            return self._handle_leave(arguments, original_message, user_context)
        
        elif function_name == "handle_feedback":
            return self._handle_feedback(arguments, original_message, user_context)
        
        elif function_name == "handle_general_query":
            return self._handle_general(arguments, original_message)
        
        else:
            return "I'm not sure how to handle that request. Could you please clarify?"
    
    def _handle_policy(self, arguments: dict, original_message: str) -> str:
        """Route to RAG agent for policy questions."""
        query = arguments.get("query", original_message)
        
        try:
            response = self.rag_agent.query(query)
            return response
        except Exception as e:
            print(f"RAG Agent error: {e}")
            return ("I'm having trouble accessing the policy documents right now. "
                   "Please try again later or contact HR directly.")
    
    def _handle_leave(
        self, 
        arguments: dict, 
        original_message: str,
        user_context: Dict[str, Any]
    ) -> str:
        """Route to Leave agent for leave management."""
        
        # Build context for leave agent
        context = {
            "action": arguments.get("action", ""),
            "employee_id": arguments.get("employee_id") or user_context.get("employee_id"),
            "employee_name": arguments.get("employee_name") or user_context.get("employee_name"),
            "is_hr": user_context.get("is_hr", False),
            "extracted_params": {
                "leave_type": arguments.get("leave_type", "Annual"),
                "start_date": arguments.get("start_date"),
                "end_date": arguments.get("end_date"),
                "num_days": arguments.get("num_days"),
                "reason": arguments.get("reason"),
                "status": arguments.get("status"),
                "approval_reason": arguments.get("reason"),
                "employee_id": arguments.get("employee_id") or user_context.get("employee_id"),
            }
        }
        
        try:
            response = self.leave_agent.handle(original_message, context)
            return response.message
        except Exception as e:
            print(f"Leave Agent error: {e}")
            return ("I'm having trouble processing your leave request. "
                   "Please try again or contact HR directly.")
    
    def _handle_feedback(
        self, 
        arguments: dict, 
        original_message: str,
        user_context: Dict[str, Any]
    ) -> str:
        """Route to Feedback agent for feedback handling."""
        
        context = {
            "action": arguments.get("action", "submit_feedback"),
            "feedback_text": arguments.get("feedback_text", original_message),
            "is_hr": user_context.get("is_hr", False)
        }
        
        try:
            response = self.feedback_agent.handle(original_message, context)
            return response.message
        except Exception as e:
            print(f"Feedback Agent error: {e}")
            return ("I'm having trouble processing your feedback. "
                   "Please try again later.")
    
    def _handle_general(self, arguments: dict, original_message: str) -> str:
        """Handle general queries with conversational response using Claude."""
        
        try:
            response = self.claude.messages.create(
                model=self.claude_model,
                max_tokens=512,
                system="""You are a friendly HR assistant named Virtual HR. 
You help employees with:
- Company policies and documentation
- Leave requests and balance
- Anonymous feedback submission

Be warm and helpful. If the user seems lost, briefly explain what you can help with.""",
                messages=[
                    {"role": "user", "content": original_message}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return "Hello! I'm Virtual HR. I can help you with company policies, leave management, and collecting feedback. How can I assist you today?"
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        return self.conversations.get(session_id, [])
    
    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]


# For direct testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    orchestrator = OrchestratorAgent()
    
    # Test conversation
    session_id = "test-session"
    
    test_messages = [
        "Hello!",
        "What is the leave policy for marriage?",
        "I want to check my leave balance. My employee ID is 123.",
        "I want to submit feedback: The new office layout is great!",
    ]
    
    for msg in test_messages:
        print(f"\n👤 User: {msg}")
        response = orchestrator.chat(msg, session_id)
        print(f"🤖 HR Bot: {response}")
