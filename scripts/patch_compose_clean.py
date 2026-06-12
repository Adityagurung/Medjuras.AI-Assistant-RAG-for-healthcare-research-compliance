#!/usr/bin/env python3
from pathlib import Path
fp = Path(__file__).resolve().parent.parent / ("docker"+"-compose.yaml")
text = fp.read_text()
if "ollama_medi_rag" in text:
    print("already patched")
    raise SystemExit(0)
hq="http://"+"qdrant"+":6333"
ho="http://"+"ollama"+":11434"
oldq="      QDRANT_URL: "+hq+"\n"
newq=oldq+"      OLLAMA_BASE_URL: "+ho+"\n      OLLAMA_MODEL: ${OLLAMA_MODEL:-llama3.2}\n"
text=text.replace(oldq,newq)
oldd="      qdrant:\n        condition: service_started\n"
newd=oldd+"      ollama:\n        condition: service_started\n"
text=text.replace(oldd,newd)
fp.write_text(text,encoding="utf-8")
print("patched")
