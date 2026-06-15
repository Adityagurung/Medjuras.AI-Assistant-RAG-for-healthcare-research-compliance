import base64
import os

import streamlit as st


def apply_custom_styling():

    header_img_path = "images/healthcareai.jpg"
    header_width = 800

    if os.path.exists(header_img_path):
        st.image(header_img_path, width=header_width)
    else:
        st.error(f"Header image not found: {header_img_path}")
