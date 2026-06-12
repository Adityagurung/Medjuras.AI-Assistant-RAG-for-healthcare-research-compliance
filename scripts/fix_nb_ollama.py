import json, pathlib
nb_path = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "6_llm_evaluation.ipynb"
nb = json.loads(nb_path.read_text())
host = "http://" + "localhost" + ":11434"
src=["## Ollama setup (required for MODEL_COMPARE)\n","\n","Uses local llama3.2 for model comparison.\n","\n","### Option A - Docker\n","\n","1. docker compose up -d ollama\n","2. docker compose exec ollama ollama pull llama3.2\n","3. curl -s "+host+"/api/tags\n","\n","Jupyter .env: OLLAMA_BASE_URL="+host+", OLLAMA_MODEL=llama3.2\n","\n","### Option B - Manual\n","\n","1. Install Ollama from ollama.com\n","2. ollama serve\n","3. ollama pull llama3.2\n","4. curl -s "+host+"/api/tags\n","\n","Run MODEL_COMPARE cell below.\n"]
for c in nb["cells"]:
    if c.get("id")=="ollama-setup-md":
        c["source"]=src
        break
nb_path.write_text(json.dumps(nb,indent=1),encoding="utf-8")
print("fixed")
