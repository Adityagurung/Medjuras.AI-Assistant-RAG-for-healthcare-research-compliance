import json
import os
import sys
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from ingestion.paths import MEDRAG_RAW_DIR, ensure_data_dirs
from update_docs import record_milestones

datasets.config.HF_HUB_TIMEOUT = 30



def _with_ext(path, fmt):
    ext = ".json" if fmt == "json" else ".parquet"
    return path if path.endswith(ext) else path + ext


def save_sample(sample_data, output_path, output_format):
    out = _with_ext(output_path, output_format)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if output_format == "json":
        with open(out, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
    elif output_format == "parquet":
        table = pa.Table.from_pylist(sample_data)
        pq.write_table(table, out, compression="snappy")
    else:
        raise ValueError(f"Unknown output_format: {output_format}")

    return out


def _has_any(text_chunk, token_set):
    text_lower = text_chunk.lower()
    return any(tok in text_lower for tok in token_set)


def passes_human_clinical_filter(text_chunk):
    """Keep documents that match at least 2 of 4 relevance scopes."""
    scopes = {
        "human_clinical": [
            "human",
            "patient",
            "clinical",
            "hospital",
            "cohort",
            "case report",
            "participants",
            "subject",
            "trial",
        ],
        "disease_symptom": [
            "disease",
            "disorder",
            "syndrome",
            "pathology",
            "diagnosis",
            "symptom",
            "infection",
            "illness",
        ],
        "anatomy": [
            "anatomy",
            "anatomical",
            "organ",
            "tissue",
            "heart",
            "cardiac",
            "brain",
            "liver",
            "kidney",
            "lung",
            "pulmonary",
            "vascular",
            "nerve",
        ],
        "treatment_outcome": [
            "treatment",
            "therapy",
            "therapeutic",
            "intervention",
            "drug",
            "medication",
            "surgery",
            "outcome",
            "prognosis",
            "response",
            "recovery",
            "cure",
        ],
    }
    matched = sum(1 for tokset in scopes.values() if _has_any(text_chunk, tokset))
    return matched >= 2


def create_medical_seed(
    dataset_path,
    seed_size=None,
    min_content_len=100,
    output_path="data/medical_wikipedia_seed.json",
    output_format="json",
    source="textbook",
):
    print(f"Loading MedRAG {dataset_path} dataset...")
    ds = load_dataset(dataset_path, streaming=True)

    sample_data = []
    desc = f"Collecting {source} ({dataset_path})"
    pbar = tqdm(desc=desc, unit="row", dynamic_ncols=True)

    for i, item in enumerate(ds["train"]):
        pbar.update(1)
        if seed_size is not None and len(sample_data) >= seed_size:
            break

        text = item.get("content", "")
        if len(text) <= min_content_len:
            pbar.set_postfix(collected=len(sample_data), scanned=i + 1)
            continue
        if not passes_human_clinical_filter(text):
            pbar.set_postfix(collected=len(sample_data), scanned=i + 1)
            continue

        sample_data.append(
            {
                "id": item.get("id", str(i)),
                "title": item.get("title", ""),
                "text": text,
                "wiki_id": item.get("wiki_id", ""),
                "source": source,
            }
        )
        pbar.set_postfix(collected=len(sample_data), scanned=i + 1)

    pbar.close()

    full_out_path = save_sample(sample_data, output_path, output_format)

    print(f"Created medical seed dataset with {len(sample_data)} articles")
    print(f"Saved to: {full_out_path}")
    print(f"File size: {os.path.getsize(full_out_path) / (1024 * 1024):.1f} MB")

    return sample_data


if __name__ == "__main__":
    ensure_data_dirs()
    record_milestones(
        [
            "Seed collection uses a 2-of-4 relevance filter: human clinical, disease symptom, anatomy, and treatment outcome.",
            "PubMed and textbook sources use the same filter.",
            "Animal terms are not auto-excluded when relevance scope is met.",
            "Notebook 01 target is 500 curated documents per source.",
        ]
    )
    create_medical_seed(
        dataset_path="MedRAG/textbooks",
        seed_size=500,
        output_path=str(MEDRAG_RAW_DIR / "medical_textbook_seed.json"),
        output_format="json",
        source="textbook",
    )
    create_medical_seed(
        dataset_path="MedRAG/pubmed",
        seed_size=500,
        output_path=str(MEDRAG_RAW_DIR / "medical_pubmed_seed.json"),
        output_format="json",
        source="pubmed",
    )