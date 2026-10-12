"""Light, quiet styling for the analyst workspace."""

CSS = """
:root {
  --db-ink: #1c2230;
  --db-muted: #5c6578;
  --db-line: #e4e7f0;
  --db-blue: #3451d1;
  --db-purple: #6d5bd0;
  --db-card: #ffffff;
  --db-bg: #f4f6fb;
  --db-ok: #0f7b5a;
  --db-warn: #8a5a00;
}
.stApp {
  background: var(--db-bg);
  color: var(--db-ink);
}
.block-container {
  padding-top: 1.4rem;
  padding-bottom: 3rem;
  max-width: 1120px;
}
section[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid var(--db-line);
}
section[data-testid="stSidebar"] .block-container {
  padding-top: 1.2rem;
}
h1, h2, h3 {
  letter-spacing: -0.02em;
  color: var(--db-ink);
}
.db-hero {
  background: linear-gradient(180deg, #ffffff 0%, #f7f8fd 100%);
  border: 1px solid var(--db-line);
  border-radius: 20px;
  padding: 2rem 2rem 1.6rem;
  box-shadow: 0 10px 30px rgba(28, 34, 48, 0.05);
  margin-bottom: 1.2rem;
}
.db-kicker {
  color: var(--db-blue);
  font-weight: 650;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.78rem;
  margin-bottom: 0.6rem;
}
.db-hero h1 {
  font-size: 2.35rem;
  line-height: 1.15;
  margin: 0 0 0.7rem;
}
.db-hero p {
  color: var(--db-muted);
  font-size: 1.08rem;
  margin: 0;
  max-width: 42rem;
}
.db-pill {
  display: inline-block;
  border-radius: 999px;
  padding: 0.28rem 0.7rem;
  background: #eef1fb;
  color: var(--db-blue);
  font-size: 0.82rem;
  font-weight: 650;
  border: 1px solid #d9e0f7;
}
.db-pill.warn {
  background: #fff6e8;
  color: var(--db-warn);
  border-color: #f3e1c2;
}
.db-card {
  background: var(--db-card);
  border: 1px solid var(--db-line);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  box-shadow: 0 8px 24px rgba(28, 34, 48, 0.04);
  margin-bottom: 0.8rem;
}
.db-note {
  color: var(--db-muted);
  font-size: 0.95rem;
}
div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid var(--db-line);
  border-radius: 14px;
  padding: 0.7rem 0.8rem;
}
.stButton > button {
  border-radius: 12px;
  border: 1px solid #d5dced;
  font-weight: 600;
}
@media (max-width: 800px) {
  .db-hero {
    padding: 1.3rem 1rem 1.1rem;
  }
  .db-hero h1 {
    font-size: 1.7rem;
  }
  div[data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
}
"""


def inject_styles() -> None:
    import streamlit as st

    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
