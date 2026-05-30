# MedJuras.AI — Medical RAG Assistant

Agentic retrieval-augmented generation for healthcare research using **MedRAG** corpora ([textbooks](https://huggingface.co/datasets/MedRAG/textbooks), [pubmed](https://huggingface.co/datasets/MedRAG/pubmed), [wikipedia](https://huggingface.co/datasets/MedRAG/wikipedia)), **OpenAI** (`gpt-4o-mini`, `text-embedding-3-small`), optional **Ollama** (`llama3.2` in Streamlit), **Elasticsearch + Qdrant** hybrid RRF, and **LangGraph** (corpus + live PubMed).

Project layout follows [Brahman.ai](https://github.com/Adityagurung/Brahman.ai): `app/`, `notebooks/`, `data/`, `results/`, `transformers/`, `config/`, `grafana/`.

## Quick start

### 1. Virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Environment

```powershell
copy .env.example .env
# Set OPENAI_API_KEY, ENTREZ_EMAIL, Postgres/Grafana passwords
```

### 3. Docker services (required for notebook 1 indexing + eval 3–6)

```powershell
docker compose up -d elasticsearch qdrant
# optional: postgres grafana
```

If Elasticsearch indices stay **red**, recreate the stack after pulling latest `docker-compose.yaml` (disk threshold disabled for local dev).

### 4. Notebooks (run in order)

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `notebooks/1_ingest_medrag_huggingface.ipynb` | HF download → chunk → ES + Qdrant index |
| 2 | `notebooks/2_ground_truth_data.ipynb` | 200 corpus-only Q&A pairs |
| 3 | `notebooks/3_keyword_search_evaluation_minsearch.ipynb` | ES BM25 metrics → `results/` |
| 4 | `notebooks/4_semantic_search_evaluation_qdrant.ipynb` | Qdrant dense metrics |
| 5 | `notebooks/5_hybrid_search_evaluation_qdrant.ipynb` | Hybrid RRF metrics |
| 6 | `notebooks/offline-rag-evaluation.ipynb` | RAG + LLM judge |

Set kernel working directory to `notebooks/` (for `_bootstrap.py`).

**Row limits:** textbooks 700, pubmed 1000, wikipedia 300 (medical keyword filter). Config: `config/config.yaml`.

### 5. Scripts (no extra notebooks)

```powershell
$env:PYTHONPATH="app"
python scripts/index_all.py          # ES + Qdrant ingest after notebook 1 part A
python scripts/run_streamlit.py      # UI — OpenAI default, Ollama optional
python scripts/run_agent.py "What is hypertension?"   # LangGraph agent
```

## Architecture

```
HF MedRAG → records.json → chunks.jsonl → ES (BM25 + dense) + Qdrant (dense)
                ↓
         ground_truth.json (200)
                ↓
    retrieval eval (BM25 / dense / RRF) → results/*.json
                ↓
         Streamlit / LangGraph + PubMed API
```

- **Embeddings:** OpenAI `text-embedding-3-small` (1536d) for ES and Qdrant  
- **Chat:** `gpt-4o-mini` (default); Streamlit can switch to **Ollama `llama3.2`**  
- **Agent:** `app/agent/langgraph_agent.py` — hybrid corpus retrieval → live PubMed → answer  
- **Jurisdiction metadata:** `GLOBAL` (generic compliance prompts)

## Project structure

```
├── app/                 # Application code (ingestion, search, llm, agent, interface)
├── config/              # config.yaml, medical_keywords.txt
├── data/
│   ├── raw/medrag/      # textbooks.json, pubmed.json, wikipedia.json
│   ├── processed/       # records.json, chunks/chunks.jsonl
│   └── evaluation/      # ground_truth.json
├── notebooks/           # Six Brahman-style pipelines
├── results/             # Notebook outputs (metrics, plots, copies)
├── transformers/        # Pointer to app/ingestion chunking
├── grafana/             # Dashboards + provisioning only
├── scripts/             # run_streamlit, run_agent, index_all, entrypoint
├── docker-compose.yaml
└── .env.example
```

## Grafana & monitoring

- Grafana: http://localhost:3010 (see `.env` admin password)  
- Postgres logs feedback from Streamlit when DB is configured (`app/monitoring/`)

## License

See [LICENSE](LICENSE).
