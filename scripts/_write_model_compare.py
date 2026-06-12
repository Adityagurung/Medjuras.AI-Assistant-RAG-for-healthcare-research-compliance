#!/usr/bin/env python3
from pathlib import Path
import base64

def main():
    Path(__file__).resolve().parents[1].joinpath("app","evaluation","model_comparison_eval.py").write_bytes(base64.b64decode(Path(__file__).resolve().parents[1].joinpath("scripts","model_comparison_eval.b64").read_text()))
    print("ok")

if __name__ == "__main__":
    main()
