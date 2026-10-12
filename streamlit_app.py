from __future__ import annotations

import logging

import streamlit as st

from ui.components import init_state, render_section, render_sidebar
from ui.styles import inject_styles


def configure_logging() -> None:
    logger = logging.getLogger("databuddy")
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def main() -> None:
    st.set_page_config(
        page_title="DataBuddy — Your Data, Explained",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    configure_logging()
    inject_styles()
    init_state()
    render_sidebar()
    flash = st.session_state.get("flash")
    if flash:
        st.success(flash)
        st.session_state["flash"] = None
    render_section(st.session_state["section"])


if __name__ == "__main__":
    main()
