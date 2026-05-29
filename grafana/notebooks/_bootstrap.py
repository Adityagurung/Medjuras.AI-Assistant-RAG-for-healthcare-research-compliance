import os
import sys
from pathlib import Path

# Fix macOS / corporate-proxy SSL before any HTTPS clients import.
try:
    import certifi

    ca = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE") or certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "app"))
