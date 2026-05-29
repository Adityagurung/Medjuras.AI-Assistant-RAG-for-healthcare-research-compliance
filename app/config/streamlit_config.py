import base64
import os

import streamlit as st


def apply_custom_styling():

    header_img_path = "app/images/streamlit/54316015-blood-pressure-health-check.jpg"
    header_width = 800
    # header_img_path = 'app/images/4bf18256b2fe075fac25bd9c87e4ee7c.jpg'
    # header_img_path = 'app/images/video_image.jpeg'
    # header_width = 400

    if os.path.exists(header_img_path):
        st.image(header_img_path, width=header_width)
    else:
        st.error(f"Header image not found: {header_img_path}")
