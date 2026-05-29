def ground_truth_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "data/evaluation/ground_truth.json"
