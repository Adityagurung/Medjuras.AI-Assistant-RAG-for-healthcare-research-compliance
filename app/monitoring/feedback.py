import logging
import uuid
from typing import Optional

from monitoring.database import db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def store_feedback(
    conversation_id: str,
    message_id: str,
    user_query: str,
    assistant_response: str,
    feedback: int,  # 1 for thumbs up, -1 for thumbs down
    user_type: Optional[str] = None,
    response_detail: Optional[str] = None,
    tool_used: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    """
    Store user feedback in the database.

    Args:
        conversation_id: Unique conversation identifier
        message_id: Unique message identifier
        user_query: The original user question
        assistant_response: The assistant's response
        feedback: 1 for thumbs up, -1 for thumbs down
        user_type: Healthcare Provider, Medical Researcher, or Patient
        response_detail: Simple, Detailed, or Technical
        tool_used: Which tool was used to generate the response
        session_id: Browser/streamlit session identifier

    Returns:
        bool: True if feedback was stored successfully, False otherwise
    """

    insert_sql = """
        INSERT INTO conversation_feedback
        (conversation_id, message_id, user_query, assistant_response,
         feedback, user_type, response_detail, tool_used, session_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        with db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    insert_sql,
                    (
                        conversation_id,
                        message_id,
                        user_query,
                        assistant_response,
                        feedback,
                        user_type,
                        response_detail,
                        tool_used,
                        session_id,
                    ),
                )
                conn.commit()

                feedback_type = "👍 positive" if feedback == 1 else "👎 negative"
                logger.info(f"Stored {feedback_type} feedback for message {message_id}")
                return True

    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        return False


def generate_message_id() -> str:
    """Generate a unique message ID"""
    return str(uuid.uuid4())


if __name__ == "__main__":
    # Test feedback storage
    print("Testing feedback storage...")

    test_message_id = generate_message_id()

    success = store_feedback(
        conversation_id="test-conv-123",
        message_id=test_message_id,
        user_query="What are the symptoms of diabetes?",
        assistant_response="Common symptoms of diabetes include increased thirst, frequent urination, and fatigue.",
        feedback=1,  # thumbs up
        user_type="Patient",
        response_detail="Simple",
        tool_used="direct_response",
        session_id="test-session",
    )

    if success:
        print("✅ Test feedback stored successfully!")
    else:
        print("❌ Test feedback storage failed!")
