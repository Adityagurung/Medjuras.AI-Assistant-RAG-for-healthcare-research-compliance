"""CLI demo for the LangGraph agent (corpus + live PubMed)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from agent.langgraph_agent import run_agent


def main():
    """Parse args and print agent answer."""
    parser = argparse.ArgumentParser(description="MedJuras LangGraph agent")
    parser.add_argument("question", nargs="?", default="What is hypertension?")
    args = parser.parse_args()
    state = run_agent(args.question, local=True)
    print(state.get("answer", ""))


if __name__ == "__main__":
    main()
