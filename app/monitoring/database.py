import logging
import os
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get PostgreSQL connection using environment variables"""
    try:
        # Use postgres service name for Docker, localhost for local testing
        host = "postgres" if os.getenv("DOCKER_CONTAINER") else "localhost"
        port = (
            "5432"
            if os.getenv("DOCKER_CONTAINER")
            else os.getenv("POSTGRES_PORT", "5433")
        )

        conn = psycopg2.connect(
            host=host,
            database=os.getenv("POSTGRES_DB", "ragdb"),
            user=os.getenv("POSTGRES_USER", ""),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            port=port,
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise


@contextmanager
def db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = get_db_connection()
        yield conn
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def create_feedback_table():
    """Create the conversation_feedback table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS conversation_feedback (
        id SERIAL PRIMARY KEY,
        conversation_id VARCHAR(255) NOT NULL,
        message_id VARCHAR(255) NOT NULL UNIQUE,
        user_query TEXT NOT NULL,
        assistant_response TEXT NOT NULL,
        feedback INTEGER NOT NULL CHECK (feedback IN (1, -1)),
        user_type VARCHAR(50),
        response_detail VARCHAR(50),
        tool_used VARCHAR(100),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        session_id VARCHAR(255)
    );
    """

    create_indexes_sql = [
        "CREATE INDEX IF NOT EXISTS idx_feedback_conversation ON conversation_feedback(conversation_id);",
        "CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON conversation_feedback(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_feedback_tool ON conversation_feedback(tool_used);",
        "CREATE INDEX IF NOT EXISTS idx_feedback_message ON conversation_feedback(message_id);",
    ]

    try:
        with db_connection() as conn:
            with conn.cursor() as cursor:
                # Create table
                cursor.execute(create_table_sql)
                logger.info("Feedback table created successfully")

                # Create indexes
                for index_sql in create_indexes_sql:
                    cursor.execute(index_sql)
                    logger.info(f"Index created: {index_sql}")

                conn.commit()
                logger.info("Database schema setup completed")

    except psycopg2.Error as e:
        logger.error(f"Error creating feedback table: {e}")
        raise


def check_feedback_exists(message_id: str) -> bool:
    """Check if feedback already exists for a message ID"""
    try:
        with db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM conversation_feedback WHERE message_id = %s LIMIT 1",
                    (message_id,),
                )
                return cursor.fetchone() is not None
    except psycopg2.Error as e:
        logger.error(f"Error checking feedback existence: {e}")
        return False


if __name__ == "__main__":
    # Check if the postgres container is running
    # docker ps | grep postgres

    # If it's not running, start it:
    # docker compose up postgres -d

    # Check the logs to see what happened:
    # docker compose logs postgres

    print("Testing database connection...")
    try:
        create_feedback_table()
        print("✅ Database connection successful and table created!")

        # Test connection
        with db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                print(f"PostgreSQL version: {version[0]}")

    except Exception as e:
        print(f"❌ Database connection failed: {e}")


# SELECT *
# FROM conversation_feedback

def create_conversations_table():
    sql = 'CREATE TABLE IF NOT EXISTS conversations (id SERIAL PRIMARY KEY, conversation_id VARCHAR(255))'
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()

def create_compliance_violations_table():
    sql = 'CREATE TABLE IF NOT EXISTS compliance_violations (id SERIAL PRIMARY KEY, violation_type VARCHAR(100))'
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()

def init_db_schema():
    create_feedback_table()
    create_conversations_table()
    create_compliance_violations_table()


def log_compliance_violation(violation_type, query_excerpt='', jurisdiction_hint='unknown', details=''):
    create_compliance_violations_table()
    q = 'INSERT INTO compliance_violations (violation_type) VALUES (%s)'
    try:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(q, (violation_type,))
            conn.commit()
    except psycopg2.Error as e:
        logger.error('compliance log failed: %s', e)
