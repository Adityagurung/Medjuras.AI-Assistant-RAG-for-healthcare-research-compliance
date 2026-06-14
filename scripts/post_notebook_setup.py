"""Post-notebook setup: verify stack, init Postgres feedback table."""
from __future__ import annotations
import json, os, subprocess, sys
import urllib.error, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
