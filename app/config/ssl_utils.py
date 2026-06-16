from __future__ import annotations
import os
from pathlib import Path
import certifi

def _is_valid_ca_bundle(path):
    if not path: return False
    bundle=Path(path)
    if not bundle.is_file() or bundle.stat().st_size==0: return False
    try: text=bundle.read_text(encoding="utf-8", errors="ignore")
    except OSError: return False
    return "BEGIN CERTIFICATE" in text

def configure_ssl_environment():
    cafile=certifi.where()
    for env_name in ("CORPORATE_CA_BUNDLE","SSL_CERT_FILE","REQUESTS_CA_BUNDLE"):
        candidate=os.getenv(env_name)
        if _is_valid_ca_bundle(candidate):
            cafile=str(Path(candidate).resolve()); break
    os.environ["SSL_CERT_FILE"]=cafile
    os.environ["REQUESTS_CA_BUNDLE"]=cafile
    return cafile

def httpx_verify_for_url(url):
    if (url or "").lower().startswith("http://"): return False
    return configure_ssl_environment()
