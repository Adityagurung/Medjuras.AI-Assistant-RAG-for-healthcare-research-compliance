from __future__ import annotations
import logging
from typing import Optional
from monitoring.database import db_connection, migrate_feedback_schema

logger = logging.getLogger(__name__)

def primary_tool_name(used_tools) -> str:
    if not used_tools:
        return "none"
    first = used_tools[0]
    if isinstance(first, (list, tuple)) and first:
        return str(first[0])
    return "unknown"

def store_conversation_turn(**kwargs) -> bool:
    migrate_feedback_schema()
    sql = """INSERT INTO conversation_feedback
        (conversation_id, message_id, user_query, assistant_response, feedback,
         user_type, response_detail, tool_used, session_id, model, prompt_tokens,
         completion_tokens, total_tokens, response_time_sec, estimated_cost_usd, llm_provider)
        VALUES (%(conversation_id)s, %(message_id)s, %(user_query)s, %(assistant_response)s, 0,
         %(user_type)s, %(response_detail)s, %(tool_used)s, %(session_id)s, %(model)s, %(prompt_tokens)s,
         %(completion_tokens)s, %(total_tokens)s, %(response_time_sec)s, %(estimated_cost_usd)s, %(llm_provider)s)
        ON CONFLICT (message_id) DO UPDATE SET
         assistant_response=EXCLUDED.assistant_response, model=EXCLUDED.model,
         prompt_tokens=EXCLUDED.prompt_tokens, completion_tokens=EXCLUDED.completion_tokens,
         total_tokens=EXCLUDED.total_tokens, response_time_sec=EXCLUDED.response_time_sec,
         estimated_cost_usd=EXCLUDED.estimated_cost_usd, llm_provider=EXCLUDED.llm_provider,
         tool_used=EXCLUDED.tool_used"""
    try:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, kwargs)
            conn.commit()
        return True
    except Exception as e:
        logger.error("store_conversation_turn failed: %s", e)
        return False

def update_star_rating(message_id: str, rating: int, **kwargs) -> bool:
    if rating < 1 or rating > 5:
        return False
    migrate_feedback_schema()
    sql = """UPDATE conversation_feedback SET feedback=%s, user_type=COALESCE(%s,user_type),
        response_detail=COALESCE(%s,response_detail), tool_used=COALESCE(%s,tool_used),
        session_id=COALESCE(%s,session_id) WHERE message_id=%s"""
    try:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (rating, kwargs.get("user_type"), kwargs.get("response_detail"), kwargs.get("tool_used"), kwargs.get("session_id"), message_id))
            conn.commit()
        return True
    except Exception as e:
        logger.error("update_star_rating failed: %s", e)
        return False
