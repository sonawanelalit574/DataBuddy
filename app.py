"""ASGI entry point.

Vercel loads a top-level ``app`` from this file. Streamlit's UI stays in
``streamlit_app.py`` so this module can be imported without starting the UI.
"""

from pathlib import Path

import streamlit as st

app = st.App(Path(__file__).resolve().with_name("streamlit_app.py"))

if __name__ == "__main__":
    app.run()
