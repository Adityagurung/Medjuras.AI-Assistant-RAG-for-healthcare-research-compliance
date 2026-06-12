#!/usr/bin/env bash
set -euo pipefail
# Pull llama3.2 into docker-compose Ollama (run from repo root).
MODEL="${OLLAMA_MODEL:-llama3.2}"
docker compose up -d ollama
docker compose exec ollama ollama pull "${MODEL}"
docker compose exec ollama ollama list
echo "Ollama ready on port 11434"
