#!/bin/sh
set -e
echo wait backends
sleep 8
python -c "import sys; sys.path.insert(0,'app'); from monitoring.database import init_db_schema; init_db_schema()" || true
python app/warmup.py || true
exec streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0


