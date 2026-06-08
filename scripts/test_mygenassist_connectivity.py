"""Connectivity checks for MyGenAssist."""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

load_dotenv(ROOT / ".env")

from llm.mygenassist_client import (
    get_aux_model,
    get_chat_model,
    get_embedding_model,
    get_llm_client,
    get_mygenassist_base_url,
    structured_chat_parse,
    use_mygenassist,
)


class PingResult(BaseModel):
    status: str
    message: str


def _fail(label, exc):
    print("FAIL ", label, ":", exc, sep="")
    if "chat.int.bayer.com" in str(exc) or "Connect" in type(exc).__name__:
        print("      Hint: connect to Bayer VPN and re-run this script.")


def main():
    print("MyGenAssist connectivity test")
    print("  base_url:", get_mygenassist_base_url())

    if not use_mygenassist():
        print("FAIL  MYGENASSIST_API_KEY is not set in .env")
        return 1

    client = get_llm_client()
    chat_model = get_chat_model()
    aux_model = get_aux_model()
    passed = 0
    total = 4

    try:
        resp = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
        )
        text = (resp.choices[0].message.content or "").strip()
        print("PASS  chat model (" + chat_model + "):", repr(text[:80]))
        passed += 1
    except Exception as exc:
        _fail("chat model (" + chat_model + ")", exc)

    try:
        resp = client.chat.completions.create(
            model=aux_model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
        )
        text = (resp.choices[0].message.content or "").strip()
        print("PASS  aux model (" + aux_model + "):", repr(text[:80]))
        passed += 1
    except Exception as exc:
        _fail("aux model (" + aux_model + ")", exc)

    try:
        parsed = structured_chat_parse(
            client,
            model=aux_model,
            messages=[{"role": "user", "content": "Return JSON with status ok and message structured works."}],
            response_format=PingResult,
            temperature=0,
        )
        print("PASS  structured parse (" + aux_model + "):", parsed.model_dump())
        passed += 1
    except Exception as exc:
        _fail("structured parse (" + aux_model + ")", exc)

    embedding_model = get_embedding_model()
    try:
        resp = client.embeddings.create(
            model=embedding_model,
            input="MyGenAssist embedding connectivity check",
        )
        dims = len(resp.data[0].embedding)
        print("PASS  embeddings (" + embedding_model + "): vector dims =", dims)
        passed += 1
    except Exception as exc:
        _fail("embeddings (" + embedding_model + ")", exc)


    print("")
    print("Result:", passed, "of", total, "checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
