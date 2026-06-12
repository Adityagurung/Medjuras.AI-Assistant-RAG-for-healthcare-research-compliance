from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
from llm.mygenassist_client import get_aux_model, get_chat_model, use_mygenassist
from llm.ollama_client import OLLAMA_MODEL, ollama_chat
from llm.openai_client import DEFAULT_MODEL, get_openai_client
from search.hybrid_search import hybrid_search
from search.hydrate import hydrate_hits
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from evaluation.config_paths import load_ground_truth
from evaluation.model_comparison_charts import save_model_comparison_dashboard
from ingestion.paths import RESULTS_DIR, ensure_data_dirs

def _build_context(hits, max_chars=2500):
    parts, used = [], 0
    for h in hits:
        block = f"{h.title}\n{h.text[:500]}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)

def _ground_truth_text(doc_id):
    if not doc_id:
        return ""
    hits = hydrate_hits([doc_id])
    if not hits:
        return ""
    h = hits[0]
    return f"{h.title}\n{h.text}".strip()

def _resolve_models():
    chat = get_chat_model() if use_mygenassist() else os.getenv("OPENAI_DEFAULT_MODEL", DEFAULT_MODEL)
    aux = get_aux_model() if use_mygenassist() else os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    ollama_name = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    return {
        chat: {"provider": "openai", "model": chat, "temperature": 0.1},
        aux: {"provider": "openai", "model": aux, "temperature": 0.1},
        f"ollama:{ollama_name}": {"provider": "ollama", "model": ollama_name, "temperature": 0.1},
    }

def _generate_answer(client, q, context, cfg):
    prompt = "Answer using only the context. Be concise.\n\nQ: " + q + "\n\nContext:\n" + context
    if cfg["provider"] == "ollama":
        resp = ollama_chat(prompt, system="Answer from context only.", model=cfg["model"], temperature=cfg.get("temperature", 0.1))
        return (resp.text or "").strip()
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "system", "content": "Answer from context only."}, {"role": "user", "content": prompt}],
        temperature=cfg.get("temperature", 0.1),
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()

def _cosine_sim(embedder, a, b):
    if not a or not b:
        return 0.0
    v1, v2 = embedder.encode(a), embedder.encode(b)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return 0.0 if denom <= 0 else float(np.dot(v1, v2) / denom)

def run_model_comparison(max_samples=25, local=True, embedding_model_name="multi-qa-MiniLM-L6-cos-v1", save_json=True, save_chart=True):
    rows_gt = load_ground_truth()[:max_samples]
    model_cfgs = _resolve_models()
    model_names = list(model_cfgs.keys())
    embedder = SentenceTransformer(embedding_model_name)
    client = get_openai_client()
    per_q, scores_by_model = [], {m: [] for m in model_names}
    for row in tqdm(rows_gt, desc="Model comparison", unit="q"):
        qtext = row["question"]
        doc_id = row.get("doc_id", "")
        gt_text = _ground_truth_text(doc_id)
        hits = hybrid_search(qtext, top_k=5, local=local)
        context = _build_context(hits)
        entry = {"question": qtext, "doc_id": doc_id, "models": {}}
        for label, cfg in model_cfgs.items():
            ans = _generate_answer(client, qtext, context, cfg)
            sim = _cosine_sim(embedder, ans, gt_text)
            scores_by_model[label].append(sim)
            entry["models"][label] = {"answer": ans, "cosine_similarity": sim, "provider": cfg["provider"], "model_id": cfg["model"]}
        per_q.append(entry)
    summary = []
    for label in model_names:
        sc = scores_by_model[label]
        summary.append({"model": label, "mean_cosine": float(np.mean(sc)) if sc else 0.0, "std_cosine": float(np.std(sc)) if sc else 0.0, "n": len(sc)})
    result = {"n_questions": len(per_q), "local": local, "embedding_model": embedding_model_name, "summary": summary, "scores_by_model": scores_by_model, "per_question": per_q}
    json_p, chart_p = None, None
    if save_json:
        ensure_data_dirs()
        jp = RESULTS_DIR / "model_comparison.json"
        jp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        json_p = str(jp.resolve())
    if save_chart:
        ensure_data_dirs()
        img = RESULTS_DIR / "images" / "model_comparison_dashboard.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        chart_p = str(save_model_comparison_dashboard(scores_by_model, out_path=img))
    result["paths"] = {"json": json_p, "chart": chart_p}
    return result

if __name__ == "__main__":
    out = run_model_comparison(max_samples=5, local=True)
    print(json.dumps(out["summary"], indent=2))
    print("Saved:", out.get("paths"))
