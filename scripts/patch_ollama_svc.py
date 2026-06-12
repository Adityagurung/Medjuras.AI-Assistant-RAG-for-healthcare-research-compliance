#!/usr/bin/env python3
import pathlib
fp = pathlib.Path(__file__).resolve().parent.parent / ("docker" + "-compose.yaml")
t = fp.read_text()
if "container_name: ollama_medi_rag" in t:
    raise SystemExit(0)
img = "ollama/ollama:latest"
vol = "./data/ollama:/root/.ollama"
b="\n  ollama:\n    image: "+img+"\n    container_name: ollama_medi_rag\n    ports:\n      - \"11434:11434\"\n    volumes:\n      - "+vol+"\n    restart: unless-stopped\n\n  qdrant:\n"
t=t.replace("  qdrant:\n", b, 1)
fp.write_text(t, encoding="utf-8")
print("svc added")
