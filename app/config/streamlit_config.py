import base64
import os
from pathlib import Path

import streamlit as st


def _header_image_path() -> Path:
    app_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    docker_app = Path("/app") / "app" / "images" / "healthcareai.jpg"
    candidates = [
        app_root / "images" / "healthcareai.jpg",
        repo_root / "images" / "healthcareai.jpg",
        docker_app,
        Path("/app/images/healthcareai.jpg"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def apply_custom_styling():
    header_img_path = _header_image_path()
    header_width = 800
    if header_img_path.is_file():
        st.image(str(header_img_path), width=header_width)
    else:
        st.error(f"Header image not found: {header_img_path}")
