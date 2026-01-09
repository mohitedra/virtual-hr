"""
Google Sheets client wrapper for the Virtual HR system.
Uses gspread with service account authentication.
"""
import gspread
from google.oauth2.service_account import Credentials
from typing import List, Dict, Any, Optional
from datetime import datetime
import os


class SheetsClient:
    """
    A wrapper for Google Sheets operations using gspread.
    Provides CRUD operations for the HR system sheets.
    """
    
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    def __init__(self, credentials_file: str):
        """
        Initialize the Sheets client with service account credentials.
        
        Args:
            credentials_file: Path to the service account JSON file
        """
        if not os.path.exists(credentials_file):
            raise FileNotFoundError(
                f"Credentials file not found: {credentials_file}. "
                "Please download your service account credentials from Google Cloud Console."
            )
        
        self.credentials = Credentials.from_service_account_file(
            credentials_file, 
            scopes=self.SCOPES
        )
        self.client = gspread.authorize(self.credentials)
        self._sheet_cache: Dict[str, gspread.Spreadsheet] = {}
    
    def get_spreadsheet(self, sheet_id: str) -> gspread.Spreadsheet:
        """Get a spreadsheet by ID with caching."""
        if sheet_id not in self._sheet_cache:
            self._sheet_cache[sheet_id] = self.client.open_by_key(sheet_id)
        return self._sheet_cache[sheet_id]
    
    def get_worksheet(
        self, 
        sheet_id: str, 
        worksheet_name: str = "Sheet1"
    ) -> gspread.Worksheet:
        """Get a specific worksheet from a spreadsheet."""
        spreadsheet = self.get_spreadsheet(sheet_id)
        try:
            return spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            # Create the worksheet if it doesn't exist
            return spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)
    
    def append_row(
        self, 
        sheet_id: str, 
        row_data: List[Any], 
        worksheet_name: str = "Sheet1"
    ) -> int:
        """
        Append a row to the worksheet.
        
        Returns:
            The row number of the newly added row
        """
        worksheet = self.get_worksheet(sheet_id, worksheet_name)
        worksheet.append_row(row_data, value_input_option='USER_ENTERED')
        return worksheet.row_count
    
    def get_all_records(
        self, 
        sheet_id: str, 
        worksheet_name: str = "Sheet1"
    ) -> List[Dict[str, Any]]:
        """Get all records from the worksheet as a list of dictionaries."""
        worksheet = self.get_worksheet(sheet_id, worksheet_name)
        return worksheet.get_all_records()
    
    def find_rows_by_value(
        self, 
        sheet_id: str, 
        column: int, 
        value: str, 
        worksheet_name: str = "Sheet1"
    ) -> List[gspread.Cell]:
        """
        Find all cells in a column matching the given value.
        
        Args:
            sheet_id: The spreadsheet ID
            column: Column number (1-indexed)
            value: Value to search for
            worksheet_name: Name of the worksheet
            
        Returns:
            List of matching cells
        """
        worksheet = self.get_worksheet(sheet_id, worksheet_name)
        try:
            return worksheet.findall(value, in_column=column)
        except gspread.CellNotFound:
            return []
    
    def update_cell(
        self, 
        sheet_id: str, 
        row: int, 
        col: int, 
        value: Any, 
        worksheet_name: str = "Sheet1"
    ) -> None:
        """Update a specific cell."""
        worksheet = self.get_worksheet(sheet_id, worksheet_name)
        worksheet.update_cell(row, col, value)
    
    def update_row(
        self, 
        sheet_id: str, 
        row: int, 
        values: List[Any], 
        start_col: int = 1, 
        worksheet_name: str = "Sheet1"
    ) -> None:
        """Update multiple cells in a row starting from start_col."""
        worksheet = self.get_worksheet(sheet_id, worksheet_name)
        for i, value in enumerate(values):
            worksheet.update_cell(row, start_col + i, value)
    
    def get_row(
        self, 
        sheet_id: str, 
        row: int, 
        worksheet_name: str = "Sheet1"
    ) -> List[Any]:
        """Get all values in a specific row."""
        worksheet = self.get_worksheet(sheet_id, worksheet_name)
        return worksheet.row_values(row)
    
    def ensure_headers(
        self, 
        sheet_id: str, 
        headers: List[str], 
        worksheet_name: str = "Sheet1"
    ) -> None:
        """Ensure the worksheet has the correct headers in the first row."""
        worksheet = self.get_worksheet(sheet_id, worksheet_name)
        existing_headers = worksheet.row_values(1)
        
        if existing_headers != headers:
            # Clear and set headers
            if not existing_headers:
                worksheet.append_row(headers, value_input_option='USER_ENTERED')
            else:
                for i, header in enumerate(headers, 1):
                    worksheet.update_cell(1, i, header)


class LeaveTrackerSheet:
    """
    Specialized sheet handler for Leave Tracker.
    """
    
    HEADERS = [
        "Employee ID",
        "Employee Name", 
        "Leave Type",
        "Start Date",
        "End Date",
        "Number of Days",
        "Leave Status",
        "Requested On",
        "Approval Date",
        "Comments/Reason"
    ]
    
    # Column indices (1-indexed for gspread)
    COL_EMPLOYEE_ID = 1
    COL_EMPLOYEE_NAME = 2
    COL_LEAVE_TYPE = 3
    COL_START_DATE = 4
    COL_END_DATE = 5
    COL_NUM_DAYS = 6
    COL_STATUS = 7
    COL_REQUESTED_ON = 8
    COL_APPROVAL_DATE = 9
    COL_COMMENTS = 10
    
    def __init__(self, sheets_client: SheetsClient, sheet_id: str):
        self.client = sheets_client
        self.sheet_id = sheet_id
        self._ensure_setup()
    
    def _ensure_setup(self):
        """Ensure the sheet has proper headers."""
        self.client.ensure_headers(self.sheet_id, self.HEADERS)
    
    def add_leave_request(
        self,
        employee_id: str,
        employee_name: str,
        leave_type: str,
        start_date: str,
        end_date: str,
        num_days: int,
        reason: str = ""
    ) -> dict:
        """Add a new leave request."""
        row_data = [
            employee_id,
            employee_name,
            leave_type,
            start_date,
            end_date,
            num_days,
            "Pending",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "",  # Approval date - empty initially
            reason
        ]
        self.client.append_row(self.sheet_id, row_data)
        return {
            "employee_id": employee_id,
            "employee_name": employee_name,
            "leave_type": leave_type,
            "start_date": start_date,
            "end_date": end_date,
            "num_days": num_days,
            "status": "Pending",
            "reason": reason
        }
    
    def get_leave_history(self, employee_id: str) -> List[Dict[str, Any]]:
        """Get all leave records for an employee."""
        all_records = self.client.get_all_records(self.sheet_id)
        return [r for r in all_records if str(r.get("Employee ID", "")) == str(employee_id)]
    
    def get_pending_leaves(self, employee_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pending leave requests, optionally filtered by employee."""
        all_records = self.client.get_all_records(self.sheet_id)
        pending = [r for r in all_records if r.get("Leave Status") == "Pending"]
        
        if employee_id:
            pending = [r for r in pending if str(r.get("Employee ID", "")) == str(employee_id)]
        
        return pending
    
    def update_leave_status(
        self, 
        employee_id: str, 
        status: str, 
        reason: str,
        start_date: Optional[str] = None
    ) -> dict:
        """
        Update the status of a leave request.
        
        Args:
            employee_id: The employee's ID
            status: New status (Approved/Rejected)
            reason: Reason for approval/rejection
            start_date: Optional - to identify specific leave if multiple pending
            
        Returns:
            Updated leave record or error info
        """
        worksheet = self.client.get_worksheet(self.sheet_id)
        all_values = worksheet.get_all_values()
        
        # Find pending leave for this employee
        found_row = None
        for i, row in enumerate(all_values[1:], start=2):  # Skip header, 1-indexed
            if (str(row[self.COL_EMPLOYEE_ID - 1]) == str(employee_id) and 
                row[self.COL_STATUS - 1] == "Pending"):
                # If start_date specified, match it
                if start_date and row[self.COL_START_DATE - 1] != start_date:
                    continue
                found_row = i
                break
        
        if not found_row:
            return {"error": f"No pending leave found for employee {employee_id}"}
        
        # Update status
        worksheet.update_cell(found_row, self.COL_STATUS, status)
        worksheet.update_cell(found_row, self.COL_APPROVAL_DATE, 
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Append reason to comments
        current_comment = worksheet.cell(found_row, self.COL_COMMENTS).value or ""
        new_comment = f"{current_comment} | HR: {reason}" if current_comment else f"HR: {reason}"
        worksheet.update_cell(found_row, self.COL_COMMENTS, new_comment)
        
        return {
            "employee_id": employee_id,
            "status": status,
            "reason": reason,
            "updated": True
        }
    
    def calculate_used_leaves(self, employee_id: str, leave_type: str) -> int:
        """Calculate total approved leaves used by employee for a leave type."""
        history = self.get_leave_history(employee_id)
        total_used = 0
        
        for record in history:
            if (record.get("Leave Type") == leave_type and 
                record.get("Leave Status") == "Approved"):
                try:
                    total_used += int(record.get("Number of Days", 0))
                except (ValueError, TypeError):
                    pass
        
        return total_used


class FeedbackTrackerSheet:
    """
    Specialized sheet handler for Feedback Tracker.
    """
    
    HEADERS = [
        "Feedback",
        "Sentiment",
        "Action Items",
        "Submitted On"
    ]
    
    def __init__(self, sheets_client: SheetsClient, sheet_id: str):
        self.client = sheets_client
        self.sheet_id = sheet_id
        self._ensure_setup()
    
    def _ensure_setup(self):
        """Ensure the sheet has proper headers."""
        self.client.ensure_headers(self.sheet_id, self.HEADERS)
    
    def add_feedback(
        self,
        feedback_text: str,
        sentiment: str,
        action_items: str
    ) -> dict:
        """Add new feedback entry."""
        row_data = [
            feedback_text,
            sentiment,
            action_items,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        self.client.append_row(self.sheet_id, row_data)
        return {
            "feedback": feedback_text,
            "sentiment": sentiment,
            "action_items": action_items,
            "submitted": True
        }
    
    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """Get all feedback records."""
        return self.client.get_all_records(self.sheet_id)
    
    def get_feedback_by_sentiment(self, sentiment: str) -> List[Dict[str, Any]]:
        """Get feedback filtered by sentiment."""
        all_records = self.get_all_feedback()
        return [r for r in all_records if r.get("Sentiment", "").lower() == sentiment.lower()]
