import json, pathlib
nb_path = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "6_llm_evaluation.ipynb"
nb = json.loads(nb_path.read_text())
if any(c.get("id") == "ollama-setup-md" for c in nb["cells"]):
    raise SystemExit(0)
host = "http://" + "localhost" + ":11434"
src=[]
src.append("## Ollama setup (required for MODEL_COMPARE)\n")
src.append("\nModel comparison uses local llama3.2 via Ollama.\n\n")
src.append("### Option A — Docker (docker-compose)\n\n")
src.append("1. `docker compose up -d ollama`\n")
src.append("2. `docker compose exec ollama ollama pull llama3.2` (first time, ~2 GB)\n")
md={"cell_type":"markdown","id":"ollama-setup-md","metadata":{},"source":src}
idx=next(i for i,c in enumerate(nb["cells"]) if c.get("id")=="model-compare-cell")
nb["cells"].insert(idx,md)
nb_path.write_text(json.dumps(nb,indent=1),encoding="utf-8")
print("notebook updated")
