#!/bin/sh
set -e
export PYTHONPATH=/app/app:/app
python -c "from monitoring.database import create_feedback_table; create_feedback_table()" 2>/dev/null || true
exec streamlit run /app/app/main.py --server.port=8501 --server.address=0.0.0.0
