#!/bin/sh
set -e
export PYTHONPATH=/app/app:/app
exec streamlit run /app/app/main.py --server.port=8501 --server.address=0.0.0.0
