"""
Feedback Agent for the Virtual HR system.
Collects anonymous employee feedback, performs sentiment analysis using Claude,
and generates action items. All data is stored in Google Sheets.
"""
import os
import sys
from typing import Dict, Any, Optional
import json

# Add parent path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.base_agent import BaseAgent, AgentResponse
from utils.sheets_client import SheetsClient, FeedbackTrackerSheet
from config import Config
from anthropic import Anthropic


class FeedbackAgent(BaseAgent):
    """
    Agent responsible for anonymous employee feedback collection and analysis.
    
    Capabilities:
    - Collect anonymous feedback
    - Analyze sentiment (Positive/Neutral/Negative) using Claude
    - Generate actionable items from feedback
    - Retrieve feedback trends (for HR)
    """
    
    def __init__(self):
        super().__init__("Feedback Agent")
        
        # Initialize Claude for sentiment analysis and action item generation
        self.claude = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.claude_model = Config.CLAUDE_MODEL
        
        # Initialize Google Sheets
        self.sheets_client = SheetsClient(Config.GOOGLE_SHEETS_CREDENTIALS_FILE)
        self.feedback_sheet = FeedbackTrackerSheet(
            self.sheets_client, 
            Config.FEEDBACK_TRACKER_SHEET_ID
        )
    
    def handle(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process feedback-related queries.
        
        Context may contain:
        - action: The specific action (submit_feedback, view_trends)
        - is_hr: Boolean indicating if user is HR (required for viewing trends)
        - feedback_text: Pre-extracted feedback text
        """
        context = context or {}
        action = context.get("action", "").lower()
        
        # Determine action from query if not specified
        if not action:
            action = self._determine_action(query)
        
        if action == "submit_feedback":
            return self._handle_submit_feedback(query, context)
        elif action == "view_trends":
            return self._handle_view_trends(context)
        else:
            # Default to treating the query as feedback submission
            return self._handle_submit_feedback(query, context)
    
    def _determine_action(self, query: str) -> str:
        """Determine if user wants to submit feedback or view trends."""
        query_lower = query.lower()
        
        trend_keywords = ["trend", "summary", "report", "all feedback", "show feedback", "analytics"]
        if any(keyword in query_lower for keyword in trend_keywords):
            return "view_trends"
        
        return "submit_feedback"
    
    def _handle_submit_feedback(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Handle anonymous feedback submission."""
        
        # Extract feedback text from context or use query directly
        feedback_text = context.get("feedback_text") or query
        
        # Clean up the feedback if it contains submission phrases
        feedback_text = self._clean_feedback(feedback_text)
        
        if len(feedback_text.strip()) < 10:
            return self._create_response(
                success=False,
                message="Please provide more detailed feedback. "
                       "Your input helps us improve the workplace!"
            )
        
        # Analyze sentiment and generate action items using Claude
        try:
            analysis = self._analyze_feedback(feedback_text)
            sentiment = analysis.get("sentiment", "Neutral")
            action_items = analysis.get("action_items", "No specific actions identified")
            
            # Store in Google Sheets
            result = self.feedback_sheet.add_feedback(
                feedback_text=feedback_text,
                sentiment=sentiment,
                action_items=action_items
            )
            
            # Create friendly response
            sentiment_emoji = {
                "Positive": "😊",
                "Neutral": "😐",
                "Negative": "😟"
            }.get(sentiment, "📝")
            
            return self._create_response(
                success=True,
                message=f"{sentiment_emoji} **Thank you for your anonymous feedback!**\n\n"
                       f"Your feedback has been recorded and will be reviewed by HR.\n\n"
                       f"📊 **Sentiment Detected:** {sentiment}\n\n"
                       f"We value your input and are committed to improving the workplace.",
                data={
                    "feedback": feedback_text,
                    "sentiment": sentiment,
                    "action_items": action_items
                }
            )
            
        except Exception as e:
            return self._create_response(
                success=False,
                message=f"Failed to submit feedback: {str(e)}"
            )
    
    def _handle_view_trends(self, context: Dict[str, Any]) -> AgentResponse:
        """Handle feedback trends viewing (HR only)."""
        
        is_hr = context.get("is_hr", False)
        if not is_hr:
            return self._create_response(
                success=False,
                message="⚠️ Feedback trends are only accessible to HR personnel. "
                       "If you'd like to submit feedback, please share your thoughts!"
            )
        
        try:
            all_feedback = self.feedback_sheet.get_all_feedback()
            
            if not all_feedback:
                return self._create_response(
                    success=True,
                    message="📊 **Feedback Summary**\n\nNo feedback has been collected yet.",
                    data={"total": 0, "feedback": []}
                )
            
            # Aggregate by sentiment
            positive = [f for f in all_feedback if f.get("Sentiment") == "Positive"]
            neutral = [f for f in all_feedback if f.get("Sentiment") == "Neutral"]
            negative = [f for f in all_feedback if f.get("Sentiment") == "Negative"]
            
            # Get recent action items
            recent_actions = []
            for f in all_feedback[-5:]:  # Last 5 feedback items
                if f.get("Action Items"):
                    recent_actions.append(f.get("Action Items"))
            
            summary = (
                f"📊 **Feedback Summary**\n\n"
                f"**Total Feedback:** {len(all_feedback)}\n\n"
                f"**Sentiment Breakdown:**\n"
                f"• 😊 Positive: {len(positive)} ({len(positive)*100//max(len(all_feedback),1)}%)\n"
                f"• 😐 Neutral: {len(neutral)} ({len(neutral)*100//max(len(all_feedback),1)}%)\n"
                f"• 😟 Negative: {len(negative)} ({len(negative)*100//max(len(all_feedback),1)}%)\n\n"
            )
            
            if recent_actions:
                summary += "**Recent Action Items:**\n"
                for i, action in enumerate(recent_actions[:3], 1):
                    summary += f"{i}. {action}\n"
            
            return self._create_response(
                success=True,
                message=summary,
                data={
                    "total": len(all_feedback),
                    "positive": len(positive),
                    "neutral": len(neutral),
                    "negative": len(negative),
                    "recent_feedback": all_feedback[-5:]
                }
            )
            
        except Exception as e:
            return self._create_response(
                success=False,
                message=f"Failed to retrieve feedback trends: {str(e)}"
            )
    
    def _clean_feedback(self, text: str) -> str:
        """Remove common prefixes from feedback text."""
        prefixes_to_remove = [
            "i want to give feedback:",
            "i want to submit feedback:",
            "my feedback is:",
            "feedback:",
            "i'd like to share:",
            "here's my feedback:",
        ]
        
        text_lower = text.lower().strip()
        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        return text
    
    def _analyze_feedback(self, feedback_text: str) -> dict:
        """
        Analyze feedback for sentiment and generate action items using Claude.
        
        Returns:
            Dict with 'sentiment' and 'action_items' keys
        """
        prompt = """You are an HR feedback analyst. Analyze the employee feedback and provide:
1. Sentiment classification (exactly one of: Positive, Neutral, Negative)
2. Actionable items for HR to consider

Respond with JSON only:
{
    "sentiment": "Positive" | "Neutral" | "Negative",
    "action_items": "Concise actionable recommendations for HR (1-2 sentences max)"
}

Guidelines:
- Positive: Appreciation, satisfaction, praise
- Negative: Complaints, concerns, frustrations
- Neutral: Suggestions, observations, mixed feelings
- Action items should be specific and actionable
- If no clear action needed, suggest "Monitor and acknowledge"

Analyze this feedback:

""" + feedback_text
        
        response = self.claude.messages.create(
            model=self.claude_model,
            max_tokens=256,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        try:
            # Parse JSON from Claude's response
            response_text = response.content[0].text
            # Try to extract JSON if wrapped in markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text.strip())
            return {
                "sentiment": result.get("sentiment", "Neutral"),
                "action_items": result.get("action_items", "Review and acknowledge feedback")
            }
        except (json.JSONDecodeError, KeyError, IndexError):
            return {
                "sentiment": "Neutral",
                "action_items": "Review feedback manually"
            }


# For direct testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = FeedbackAgent()
    
    # Test feedback submission
    test_feedback = "The new coffee machine in the break room is amazing! It really makes the mornings better."
    response = agent.handle(test_feedback)
    print(f"\nFeedback Submission:\n{response.message}")
    
    # Test trends (as HR)
    hr_context = {"is_hr": True, "action": "view_trends"}
    response = agent.handle("Show me feedback trends", hr_context)
    print(f"\nFeedback Trends:\n{response.message}")
