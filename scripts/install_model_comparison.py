#!/usr/bin/env python3
from pathlib import Path
def main():
    p=Path(__file__).resolve().parents[1]
    print(p)
