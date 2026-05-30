"""Index chunks into Elasticsearch and Qdrant (requires Docker services up)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from ingestion.es_ingest import ingest_chunks as ingest_es
from ingestion.qdrant_ingest import ingest_chunks as ingest_qdrant


def main():
    """Run ES then Qdrant ingest with wipe."""
    n_es = ingest_es(wipe=True)
    print(f"Elasticsearch: indexed {n_es} docs")
    n_qd = ingest_qdrant(wipe=True)
    print(f"Qdrant: upserted {n_qd} points")


if __name__ == "__main__":
    main()
