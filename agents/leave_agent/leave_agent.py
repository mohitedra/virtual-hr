"""
Leave Tracker Agent for the Virtual HR system.
Handles leave requests, balance checking, and status updates via Google Sheets.
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json

# Add parent path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.base_agent import BaseAgent, AgentResponse
from utils.sheets_client import SheetsClient, LeaveTrackerSheet
from config import Config
from openai import OpenAI


class LeaveTrackerAgent(BaseAgent):
    """
    Agent responsible for leave management operations.
    
    Capabilities:
    - Submit new leave requests
    - Check leave balance
    - View leave history
    - Update leave status (HR only - requires reason)
    """
    
    LEAVE_TYPES = ["Annual", "Sick", "Personal", "Maternity", "Paternity", "Marriage", "Bereavement"]
    
    def __init__(self):
        super().__init__("Leave Tracker Agent")
        
        # Initialize OpenAI for natural language understanding
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Initialize Google Sheets
        self.sheets_client = SheetsClient(Config.GOOGLE_SHEETS_CREDENTIALS_FILE)
        self.leave_sheet = LeaveTrackerSheet(
            self.sheets_client, 
            Config.LEAVE_TRACKER_SHEET_ID
        )
        
        # Default leave balances
        self.leave_balances = Config.get_leave_balance_defaults()
    
    def handle(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process leave-related queries.
        
        Context may contain:
        - action: The specific action (submit_leave, check_balance, view_history, update_status)
        - employee_id: The employee's ID
        - employee_name: The employee's name
        - is_hr: Boolean indicating if user is HR (required for status updates)
        - extracted_params: Pre-extracted parameters from orchestrator
        """
        context = context or {}
        action = context.get("action", "").lower()
        
        # If no action specified, use LLM to understand intent
        if not action:
            action, extracted_params = self._understand_intent(query)
            context["extracted_params"] = extracted_params
        
        # Route to appropriate handler
        if action == "submit_leave":
            return self._handle_submit_leave(query, context)
        elif action == "check_balance":
            return self._handle_check_balance(query, context)
        elif action == "view_history":
            return self._handle_view_history(query, context)
        elif action == "update_status":
            return self._handle_update_status(query, context)
        else:
            return self._create_response(
                success=False,
                message="I'm not sure what you'd like to do with leave management. "
                       "I can help you submit a leave request, check your balance, "
                       "view your leave history, or (for HR) approve/reject leave requests."
            )
    
    def _understand_intent(self, query: str) -> tuple[str, dict]:
        """Use LLM to understand the user's intent and extract parameters."""
        
        system_prompt = """You are a leave management assistant. Analyze the user's query and extract:
1. The action they want to perform
2. Any relevant parameters

Respond with JSON only:
{
    "action": "submit_leave" | "check_balance" | "view_history" | "update_status",
    "params": {
        "employee_id": "string or null",
        "employee_name": "string or null",
        "leave_type": "Annual|Sick|Personal|Maternity|Paternity|Marriage|Bereavement or null",
        "start_date": "YYYY-MM-DD or null",
        "end_date": "YYYY-MM-DD or null",
        "num_days": "number or null",
        "reason": "string or null",
        "status": "Approved|Rejected or null (for update_status)",
        "approval_reason": "string or null (reason for approval/rejection)"
    }
}

Examples:
- "I want to take 2 days off next week" -> action: submit_leave
- "How many leaves do I have left?" -> action: check_balance
- "Show my leave history" -> action: view_history
- "Approve leave for employee 123 due to medical emergency" -> action: update_status
"""
        
        response = self.client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("action", ""), result.get("params", {})
        except (json.JSONDecodeError, KeyError):
            return "", {}
    
    def _handle_submit_leave(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Handle leave request submission."""
        params = context.get("extracted_params", {})
        
        # Get employee info from context or params
        employee_id = context.get("employee_id") or params.get("employee_id")
        employee_name = context.get("employee_name") or params.get("employee_name")
        
        if not employee_id or not employee_name:
            return self._create_response(
                success=False,
                message="I need your employee ID and name to submit a leave request. "
                       "Please provide them like: 'I am John Doe (ID: 123) and I want to apply for leave...'"
            )
        
        # Extract leave details
        leave_type = params.get("leave_type", "Annual")
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        num_days = params.get("num_days")
        reason = params.get("reason", "")
        
        # Validate dates
        if not start_date:
            return self._create_response(
                success=False,
                message="Please specify when you'd like to start your leave. "
                       "For example: 'I want leave from 2026-01-15 to 2026-01-17'"
            )
        
        # Calculate end_date if only num_days provided
        if not end_date and num_days:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = start + timedelta(days=int(num_days) - 1)
                end_date = end.strftime("%Y-%m-%d")
            except ValueError:
                pass
        
        # Calculate num_days if not provided
        if not num_days and end_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                num_days = (end - start).days + 1
            except ValueError:
                num_days = 1
        
        if not end_date:
            end_date = start_date
            num_days = 1
        
        # Check balance before submitting
        balance = self._get_leave_balance(str(employee_id), leave_type)
        if balance is not None and int(num_days) > balance:
            return self._create_response(
                success=False,
                message=f"Insufficient {leave_type} leave balance. "
                       f"You have {balance} days remaining but requested {num_days} days.",
                data={"balance": balance, "requested": num_days}
            )
        
        # Submit to Google Sheets
        try:
            result = self.leave_sheet.add_leave_request(
                employee_id=str(employee_id),
                employee_name=employee_name,
                leave_type=leave_type,
                start_date=start_date,
                end_date=end_date,
                num_days=int(num_days),
                reason=reason
            )
            
            return self._create_response(
                success=True,
                message=f"✅ Your {leave_type} leave request has been submitted successfully!\n\n"
                       f"📅 Dates: {start_date} to {end_date} ({num_days} days)\n"
                       f"📋 Status: Pending approval\n"
                       f"💬 Reason: {reason or 'Not specified'}\n\n"
                       f"You'll be notified once HR reviews your request.",
                data=result
            )
        except Exception as e:
            return self._create_response(
                success=False,
                message=f"Failed to submit leave request: {str(e)}"
            )
    
    def _handle_check_balance(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Handle leave balance inquiries."""
        params = context.get("extracted_params", {})
        employee_id = context.get("employee_id") or params.get("employee_id")
        leave_type = params.get("leave_type")
        
        if not employee_id:
            return self._create_response(
                success=False,
                message="Please provide your employee ID to check your leave balance. "
                       "Example: 'Check leave balance for employee ID 123'"
            )
        
        # Get balances
        balances = {}
        if leave_type:
            balance = self._get_leave_balance(str(employee_id), leave_type)
            balances[leave_type] = balance
        else:
            # Get all types
            for lt in self.LEAVE_TYPES:
                balances[lt] = self._get_leave_balance(str(employee_id), lt)
        
        # Format response
        balance_text = "\n".join([
            f"• {lt}: {bal} days remaining" 
            for lt, bal in balances.items() 
            if bal is not None
        ])
        
        return self._create_response(
            success=True,
            message=f"📊 **Leave Balance for Employee {employee_id}**\n\n{balance_text}",
            data={"employee_id": employee_id, "balances": balances}
        )
    
    def _handle_view_history(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Handle leave history inquiries."""
        params = context.get("extracted_params", {})
        employee_id = context.get("employee_id") or params.get("employee_id")
        
        if not employee_id:
            return self._create_response(
                success=False,
                message="Please provide your employee ID to view leave history. "
                       "Example: 'Show leave history for employee 123'"
            )
        
        try:
            history = self.leave_sheet.get_leave_history(str(employee_id))
            
            if not history:
                return self._create_response(
                    success=True,
                    message=f"No leave records found for employee {employee_id}.",
                    data={"employee_id": employee_id, "history": []}
                )
            
            # Format history
            history_text = ""
            for record in history[-10:]:  # Show last 10
                status_emoji = {
                    "Approved": "✅",
                    "Rejected": "❌",
                    "Pending": "⏳"
                }.get(record.get("Leave Status", ""), "❓")
                
                history_text += (
                    f"\n{status_emoji} **{record.get('Leave Type', 'N/A')}** - "
                    f"{record.get('Start Date', 'N/A')} to {record.get('End Date', 'N/A')} "
                    f"({record.get('Number of Days', 0)} days)\n"
                    f"   Status: {record.get('Leave Status', 'N/A')}\n"
                )
            
            return self._create_response(
                success=True,
                message=f"📜 **Leave History for Employee {employee_id}**\n{history_text}",
                data={"employee_id": employee_id, "history": history}
            )
        except Exception as e:
            return self._create_response(
                success=False,
                message=f"Failed to retrieve leave history: {str(e)}"
            )
    
    def _handle_update_status(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Handle leave approval/rejection (HR only)."""
        params = context.get("extracted_params", {})
        
        # Check HR permission
        is_hr = context.get("is_hr", False)
        if not is_hr:
            return self._create_response(
                success=False,
                message="⚠️ Only HR personnel can approve or reject leave requests. "
                       "Please contact your HR department."
            )
        
        employee_id = params.get("employee_id")
        status = params.get("status")
        approval_reason = params.get("approval_reason") or params.get("reason")
        start_date = params.get("start_date")
        
        if not employee_id:
            return self._create_response(
                success=False,
                message="Please specify the employee ID for the leave to update. "
                       "Example: 'Approve leave for employee 123. Reason: Medical emergency'"
            )
        
        if not status or status not in ["Approved", "Rejected"]:
            return self._create_response(
                success=False,
                message="Please specify whether to Approve or Reject the leave. "
                       "Example: 'Approve leave for employee 123. Reason: Medical emergency'"
            )
        
        if not approval_reason:
            return self._create_response(
                success=False,
                message="⚠️ A reason is required for approval/rejection. "
                       "Example: 'Approve leave for employee 123. Reason: Approved as per policy'"
            )
        
        try:
            result = self.leave_sheet.update_leave_status(
                employee_id=str(employee_id),
                status=status,
                reason=approval_reason,
                start_date=start_date
            )
            
            if result.get("error"):
                return self._create_response(
                    success=False,
                    message=result["error"]
                )
            
            status_emoji = "✅" if status == "Approved" else "❌"
            return self._create_response(
                success=True,
                message=f"{status_emoji} Leave for employee {employee_id} has been **{status}**.\n\n"
                       f"📝 Reason: {approval_reason}",
                data=result
            )
        except Exception as e:
            return self._create_response(
                success=False,
                message=f"Failed to update leave status: {str(e)}"
            )
    
    def _get_leave_balance(self, employee_id: str, leave_type: str) -> Optional[int]:
        """Calculate remaining leave balance for an employee."""
        default_balance = self.leave_balances.get(leave_type)
        if default_balance is None:
            return None
        
        used = self.leave_sheet.calculate_used_leaves(employee_id, leave_type)
        return max(0, default_balance - used)


# For direct testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = LeaveTrackerAgent()
    
    # Test with sample context
    test_context = {
        "employee_id": "TEST001",
        "employee_name": "Test User",
        "is_hr": True
    }
    
    # Test balance check
    response = agent.handle("Check my leave balance", test_context)
    print(f"\nBalance Check:\n{response.message}")
