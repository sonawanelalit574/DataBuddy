"""Streamlit views for DataBuddy."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from agent.orchestrator import Orchestrator
from analytics.data_quality import IQR_METHOD, quality_table
from analytics.dataset_loader import DatasetBundle, DatasetError, load_csv_bytes, memory_label
from analytics.export import (
    analysis_markdown,
    cleaned_frame,
    csv_bytes,
    figure_html,
    figure_png,
    quality_csv_bytes,
)
from analytics.metrics import categorical_summary, correlation_matrix, date_ranges, dataset_overview, numeric_summary
from agent.session_cache import is_current, replace_dataset
from agent.tool_registry import execute
from aws.bedrock_client import BedrockClient, BedrockError, credentials_available
from config import ROOT, get_settings

logger = logging.getLogger("databuddy.ui")

SECTIONS = [
    "Welcome",
    "Upload Dataset",
    "Dataset Overview",
    "AI Analyst Chat",
    "Visualizations",
    "Data Quality",
    "Export",
    "Settings and AWS",
]

EXAMPLE_PROMPTS = [
    "Give me a quick overview of this dataset.",
    "Find the top-performing products.",
    "Show monthly sales trends.",
    "Check for missing values and duplicates.",
    "Find unusual values and potential outliers.",
    "What should I investigate next?",
]

SAMPLE_PATH = ROOT / "data" / "sample_sales.csv"


def init_state() -> None:
    defaults = {
        "section": "Welcome",
        "bundle": None,
        "dataset_fingerprint": None,
        "messages": [],
        "last_context": {},
        "explained_ids": [],
        "pending_prompt": None,
        "use_bedrock": False,
        "include_preview": False,
        "model_id_override": "",
        "region_override": "",
        "bedrock_verified": False,
        "bedrock_status": "Not verified in this session.",
        "upload_token": None,
        "visual_result": None,
        "flash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    settings = get_settings()
    region, model, enabled = _resolved_bedrock()
    with st.sidebar:
        st.markdown("### DataBuddy")
        st.caption("Your Data, Explained")
        if enabled:
            label = "Amazon Bedrock is on"
            css = "db-pill"
        elif st.session_state["use_bedrock"]:
            label = "Bedrock was requested, but it is not configured"
            css = "db-pill warn"
        else:
            label = "Demo Mode · local calculations"
            css = "db-pill"
        st.markdown(f"<span class='{css}'>{label}</span>", unsafe_allow_html=True)
        if not enabled:
            st.caption("No cloud model is being called.")
        bundle = st.session_state.get("bundle")
        st.markdown("#### Dataset")
        if bundle is None:
            st.caption("No dataset loaded yet.")
        else:
            st.write(bundle.source_name)
            st.caption(f"{len(bundle.frame):,} rows · {bundle.frame.shape[1]} columns")
        if st.button("Load sample sales", width="stretch", key="sidebar_sample"):
            _load_sample()
        if st.button("Clear conversation", width="stretch", key="sidebar_clear"):
            st.session_state["messages"] = []
            st.session_state["last_context"] = {}
            st.session_state["explained_ids"] = []
            st.session_state["flash"] = "Conversation cleared. The dataset is still loaded."
            st.rerun()
        if st.button("Remove dataset", width="stretch", key="sidebar_reset"):
            st.session_state["bundle"] = None
            st.session_state["dataset_fingerprint"] = None
            st.session_state["messages"] = []
            st.session_state["last_context"] = {}
            st.session_state["explained_ids"] = []
            st.session_state["visual_result"] = None
            st.session_state["upload_token"] = None
            st.session_state["flash"] = "Dataset removed from this session."
            st.rerun()
        st.markdown("#### Navigate")
        for section in SECTIONS:
            selected = section == st.session_state["section"]
            if st.button(
                section,
                key=f"nav_{section}",
                width="stretch",
                type="primary" if selected else "secondary",
            ):
                st.session_state["section"] = section
                st.rerun()
        st.caption(
            "Files stay in this session unless you turn on Amazon Bedrock in Settings. "
            f"Configured in the environment: {'yes' if settings.bedrock_configured else 'no'}."
        )
        del region, model


def render_section(section: str) -> None:
    pages = {
        "Welcome": render_welcome,
        "Upload Dataset": render_upload,
        "Dataset Overview": render_overview,
        "AI Analyst Chat": render_chat,
        "Visualizations": render_visualizations,
        "Data Quality": render_quality,
        "Export": render_export,
        "Settings and AWS": render_settings,
    }
    pages[section]()


def render_welcome() -> None:
    st.markdown(
        """
        <div class="db-hero">
          <div class="db-kicker">DataBuddy</div>
          <h1>Your data has answers. Let's find them.</h1>
          <p>Upload a dataset, ask a question in plain English, and turn raw data into useful decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.markdown(
            "<div class='db-card'>No dataset is loaded. Ask a sample question and I will open the bundled sales file, "
            "or upload your own CSV first.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='db-card'>Working with <strong>{_escape(bundle.source_name)}</strong> "
            f"· {len(bundle.frame):,} rows · {bundle.frame.shape[1]} columns.</div>",
            unsafe_allow_html=True,
        )
    st.subheader("Try a question")
    st.caption("Choosing a card sends that question to the analyst. It is not a static example.")
    left, right = st.columns(2)
    for index, prompt in enumerate(EXAMPLE_PROMPTS):
        column = left if index % 2 == 0 else right
        with column:
            if st.button(prompt, key=f"welcome_prompt_{index}", width="stretch"):
                _queue_question(prompt)


def render_upload() -> None:
    st.header("Upload Dataset")
    st.write(
        "CSV files stay in this session. DataBuddy does not send the file anywhere unless you later "
        "turn on Amazon Bedrock, and even then the default is column names only."
    )
    upload = st.file_uploader("Choose a CSV", type=["csv"], accept_multiple_files=False)
    if upload is not None:
        token = f"{upload.name}:{upload.size}"
        if token != st.session_state.get("upload_token"):
            try:
                bundle = load_csv_bytes(upload.getvalue(), upload.name)
            except DatasetError as exc:
                st.error(exc.user_message)
            else:
                st.session_state["upload_token"] = token
                _adopt(bundle)
                st.session_state["flash"] = (
                    f"Loaded {bundle.source_name}: {len(bundle.frame):,} rows and {bundle.frame.shape[1]} columns."
                )
                st.rerun()
    if st.button("Load the bundled sample sales CSV", key="upload_sample"):
        _load_sample()
    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.info("Upload a CSV or load the sample to preview it here.")
        return
    _show_warnings(bundle)
    _preview(bundle)


def render_overview() -> None:
    st.header("Dataset Overview")
    bundle = _require_dataset()
    if bundle is None:
        return
    frame = bundle.frame
    overview = dataset_overview(frame)
    _show_warnings(bundle)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['rows']:,}")
    c2.metric("Columns", f"{overview['columns']:,}")
    c3.metric("Memory", overview["memory_label"])
    c4.metric("Extra duplicate rows", f"{overview['duplicate_extra']:,}")
    st.caption(f"Missing cells: {overview['missing_cells']:,}. {memory_label(frame)} includes the text in each cell.")
    details = []
    for name, kind in bundle.kinds.items():
        series = frame[name]
        details.append(
            {
                "column": name,
                "kind": kind,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
            }
        )
    st.subheader("Columns")
    st.dataframe(pd.DataFrame(details), width="stretch", hide_index=True)
    st.subheader("Preview")
    _preview(bundle)
    numeric = numeric_summary(frame)
    st.subheader("Numeric summary")
    if numeric.empty:
        st.info("This dataset has no numeric columns.")
    else:
        st.dataframe(numeric, width="stretch", hide_index=True)
    categories = categorical_summary(frame)
    st.subheader("Categories")
    if categories.empty:
        st.info("This dataset has no categorical or boolean columns.")
    else:
        st.dataframe(categories, width="stretch", hide_index=True)
    dates = date_ranges(frame)
    st.subheader("Dates")
    if dates.empty:
        st.info("This dataset has no date columns.")
    else:
        st.dataframe(dates, width="stretch", hide_index=True)
    matrix = correlation_matrix(frame)
    st.subheader("Correlations")
    if matrix.empty:
        st.info("A correlation matrix needs at least two numeric columns and a few complete rows.")
    else:
        st.caption("Pearson correlation. This is association, not cause.")
        st.dataframe(matrix.round(3), width="stretch")


def render_chat() -> None:
    st.header("AI Analyst Chat")
    _mode_banner()
    prompt = st.chat_input("Ask about this dataset")
    if prompt:
        st.session_state["pending_prompt"] = prompt
        st.rerun()
    _consume_pending()
    messages = st.session_state.get("messages") or []
    if not messages:
        st.info("Ask a question, or start from one of the cards on the Welcome page.")
        return
    fingerprint = None
    bundle = st.session_state.get("bundle")
    if bundle is not None:
        fingerprint = bundle.fingerprint
    for index, message in enumerate(messages):
        _render_message(index, message, fingerprint)


def render_visualizations() -> None:
    st.header("Interactive Visualizations")
    bundle = _require_dataset()
    if bundle is None:
        return
    frame = bundle.frame
    chart_type = st.selectbox(
        "Chart",
        ["bar", "line", "histogram", "scatter", "box", "heatmap"],
        key="viz_type",
    )
    columns = [""] + [str(column) for column in frame.columns]
    x_column = ""
    y_column = ""
    if chart_type != "heatmap":
        x_column = st.selectbox("X column", columns, key="viz_x")
        if chart_type != "histogram":
            y_column = st.selectbox("Y column", columns, key="viz_y")
    col1, col2 = st.columns(2)
    draw = col1.button("Draw chart", type="primary", key="viz_draw")
    suggest = col2.button("Use a suggested chart", key="viz_suggest")
    if suggest:
        x_column, y_column, chart_type = _suggested_chart(bundle)
        draw = True
    if draw:
        result = execute(
            "create_visualization",
            {"chart_type": chart_type, "x_column": x_column, "y_column": y_column},
            frame,
        )
        st.session_state["visual_result"] = {
            "fingerprint": bundle.fingerprint,
            "result": result,
        }
    stored = st.session_state.get("visual_result")
    if not stored:
        st.info("Choose columns and draw a chart. The chart is calculated from the current dataset.")
        return
    if stored["fingerprint"] != bundle.fingerprint:
        st.warning("The last chart belonged to a previous dataset, so it was cleared.")
        st.session_state["visual_result"] = None
        return
    result = stored["result"]
    if not result.ok:
        st.error(result.message)
        return
    if result.figure is not None:
        st.plotly_chart(result.figure, width="stretch")
    for evidence in result.evidences:
        with st.expander("Show how it was calculated", expanded=False):
            _evidence_block(evidence)
    for note in result.limitations:
        st.caption(note)


def render_quality() -> None:
    st.header("Data Quality Report")
    bundle = _require_dataset()
    if bundle is None:
        return
    st.info(IQR_METHOD + " A flagged value can be a real extreme. This screen does not decide that it is an error.")
    _show_warnings(bundle)
    table = quality_table(bundle.frame)
    if table.empty:
        st.info("There are no columns to review.")
        return
    st.dataframe(table, width="stretch", hide_index=True)
    gaps = table[table["missing_count"] > 0]
    if gaps.empty:
        st.success("No missing values in the loaded columns.")
    else:
        st.warning(f"{len(gaps)} column(s) have missing values. They are listed in the table.")


def render_export() -> None:
    st.header("Export Results")
    bundle = _require_dataset()
    if bundle is None:
        return
    try:
        cleaned, removed = cleaned_frame(bundle.frame)
        cleaned_payload = csv_bytes(cleaned)
        quality_payload = quality_csv_bytes(bundle.frame)
    except Exception:
        logger.error("export failed")
        st.error("I couldn't build that export. The dataset is still loaded.")
        return
    st.write(
        f"The cleaned CSV drops {removed:,} exact duplicate row(s) and keeps missing values as blanks. "
        "It does not invent replacements."
    )
    st.download_button(
        "Download cleaned dataset (CSV)",
        data=cleaned_payload,
        file_name="databuddy_cleaned.csv",
        mime="text/csv",
        key="export_clean",
    )
    st.download_button(
        "Download data-quality report (CSV)",
        data=quality_payload,
        file_name="databuddy_quality.csv",
        mime="text/csv",
        key="export_quality",
    )
    entries = _analysis_entries(bundle.fingerprint)
    try:
        markdown = analysis_markdown(bundle.source_name, bundle.fingerprint, entries)
    except Exception:
        logger.error("markdown export failed")
        st.error("I couldn't build the analysis summary.")
        markdown = ""
    if markdown:
        st.download_button(
            "Download analysis summary (Markdown)",
            data=markdown.encode("utf-8"),
            file_name="databuddy_analysis.md",
            mime="text/markdown",
            key="export_md",
        )
    if not entries:
        st.caption("The summary is empty until you ask a question in the chat.")
    charts = _current_charts(bundle.fingerprint)
    if not charts:
        st.caption("Charts you draw in chat or on the Visualizations page can be downloaded here.")
        return
    st.subheader("Charts")
    png_missing = False
    for index, (title, figure) in enumerate(charts):
        st.write(title or f"Chart {index + 1}")
        try:
            html = figure_html(figure)
        except Exception:
            logger.error("chart html export failed")
            st.error("I couldn't turn that chart into an HTML file.")
            continue
        st.download_button(
            "Download chart (HTML)",
            data=html.encode("utf-8"),
            file_name=f"databuddy_chart_{index + 1}.html",
            mime="text/html",
            key=f"export_html_{index}",
        )
        png = figure_png(figure)
        if png is None:
            png_missing = True
        else:
            st.download_button(
                "Download chart (PNG)",
                data=png,
                file_name=f"databuddy_chart_{index + 1}.png",
                mime="image/png",
                key=f"export_png_{index}",
            )
    if png_missing:
        st.info("PNG export needs the optional kaleido package. The HTML download still works without it.")


def render_settings() -> None:
    st.header("Settings and AWS Configuration")
    settings = get_settings()
    st.toggle(
        "Use Amazon Bedrock to interpret questions",
        key="use_bedrock",
        help="Off by default. Numbers are still calculated locally.",
    )
    st.toggle(
        "Send a 3-row preview with the question",
        key="include_preview",
        help="Leave this off if the rows contain anything private.",
    )
    st.text_input(
        "AWS region",
        key="region_override",
        placeholder="Set the region where your model is enabled",
    )
    st.text_input(
        "Bedrock model ID",
        key="model_id_override",
        placeholder="Paste a model ID that is enabled in your account",
    )
    region, model, enabled = _resolved_bedrock()
    st.markdown("#### What this session will do")
    st.write(f"Environment region: `{settings.region or 'not set'}`")
    st.write(f"Environment model ID: `{settings.model_id or 'not set'}`")
    st.write(f"Effective region: `{region or 'not set'}`")
    st.write(f"Effective model ID: `{model or 'not set'}`")
    st.write("AWS credentials detected: " + ("yes" if credentials_available() else "no"))
    st.write("Bedrock call on the next question: " + ("yes" if enabled else "no"))
    if st.session_state["bedrock_verified"]:
        st.success(st.session_state["bedrock_status"])
    else:
        st.info(st.session_state["bedrock_status"])
    if st.button("Test Bedrock connection", key="test_bedrock"):
        _test_bedrock(region, model)
    st.markdown("#### How your data is handled")
    st.markdown(
        """
- With Bedrock off, the CSV stays in this session. Demo Mode uses Pandas on this machine.
- With Bedrock on, the request includes your question, column names, column kinds, and missing-value rates.
- The full dataset is not sent unless you opt in to a 3-row preview.
- Bedrock may choose a tool and rephrase the draft. If that rephrase adds a number the tool did not return, DataBuddy keeps the verified wording.
- Do not put access keys in the app, in source control, or in a screenshot.
        """
    )
    st.markdown("#### Enable Bedrock when you are ready")
    st.markdown(
        """
1. In the Amazon Bedrock console, open model access for one model in your region. Copy that model's ID. DataBuddy does not pick one for you.
2. Give your IAM principal `bedrock:InvokeModel` for that model. Do not grant broad `bedrock:*` access for a local demo.
3. Configure credentials with the standard provider chain (`aws configure`, environment variables, or a role). Leave them out of `.env` if you can.
4. Set `AWS_REGION` and `BEDROCK_MODEL_ID` in a local `.env`, or type them above for this session only.
5. Restart Streamlit if you edited `.env`, then press **Test Bedrock connection**.
6. Turn on **Use Amazon Bedrock** only after you accept the data-handling note above.

A failed test stays labeled as not verified. Demo Mode continues to work either way.
        """
    )


def _queue_question(prompt: str) -> None:
    st.session_state["section"] = "AI Analyst Chat"
    st.session_state["pending_prompt"] = prompt
    st.rerun()


def _consume_pending() -> None:
    prompt = st.session_state.get("pending_prompt")
    if not prompt:
        return
    st.session_state["pending_prompt"] = None
    if not _ensure_dataset():
        return
    bundle = st.session_state["bundle"]
    orchestrator, warning = _orchestrator_for_request()
    with st.spinner("DataBuddy is checking the columns and running the calculation..."):
        try:
            response = orchestrator.answer(
                prompt,
                bundle.frame,
                bundle.fingerprint,
                prior=st.session_state.get("last_context") or None,
                dataset_warnings=bundle.warnings,
                include_preview=bool(st.session_state.get("include_preview")),
            )
        except Exception:
            logger.error("analysis failed")
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "question": prompt,
                    "fingerprint": bundle.fingerprint,
                    "content": (
                        "I couldn't finish that request. The dataset is unchanged. "
                        "Try a more specific question, such as a column summary or a missing-value check."
                    ),
                    "error": True,
                }
            )
            st.session_state["messages"].insert(
                -1,
                {"role": "user", "content": prompt, "fingerprint": bundle.fingerprint},
            )
            return
    if warning:
        response.warnings.insert(0, warning)
    st.session_state["last_context"] = response.context
    st.session_state["messages"].append(
        {"role": "user", "content": prompt, "fingerprint": bundle.fingerprint}
    )
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "question": prompt,
            "fingerprint": bundle.fingerprint,
            "content": response.answer,
            "response": response,
        }
    )


def _render_message(index: int, message: dict, fingerprint: str | None) -> None:
    role = message.get("role")
    if role == "system":
        st.info(message.get("content", ""))
        return
    if role == "user":
        with st.chat_message("user"):
            st.write(message.get("content", ""))
        return
    with st.chat_message("assistant"):
        response = message.get("response")
        current = fingerprint is not None and is_current(message, fingerprint)
        if message.get("error"):
            st.error(message.get("content", "That request did not finish."))
            return
        if response is None:
            st.write(message.get("content", ""))
            return
        if not current:
            st.warning("This reply belongs to a previous dataset, so its chart and table are hidden.")
            st.write(response.answer)
            return
        st.caption(response.mode_label)
        for warning in response.warnings:
            st.warning(warning)
        st.write(response.answer)
        if response.findings:
            st.markdown("**Findings**")
            for finding in response.findings:
                st.write(f"• {finding}")
        for title, table in response.tables:
            if title:
                st.caption(title)
            st.dataframe(table, width="stretch", hide_index=True)
        for title, figure in response.figures:
            if title:
                st.caption(title)
            st.plotly_chart(figure, width="stretch")
        explained = index in st.session_state["explained_ids"]
        if st.button("Explain this result", key=f"explain_{index}"):
            ids = list(st.session_state["explained_ids"])
            if index in ids:
                ids.remove(index)
            else:
                ids.append(index)
            st.session_state["explained_ids"] = ids
            st.rerun()
        if explained:
            st.info(response.explanation)
        with st.expander("Show how it was calculated"):
            if not response.evidences:
                st.write("No calculation was run for this question.")
            for evidence in response.evidences:
                _evidence_block(evidence)
            if response.assumptions:
                st.write("Assumptions: " + " ".join(response.assumptions))
            st.caption(f"Dataset fingerprint {response.fingerprint}")
        if response.limitations:
            st.markdown("**Limitations**")
            for item in response.limitations:
                st.write(f"• {item}")
        if response.next_questions:
            st.write("Want to investigate further?")
            for next_index, question in enumerate(response.next_questions):
                if st.button(question, key=f"next_{index}_{next_index}"):
                    _queue_question(question)


def _evidence_block(evidence) -> None:
    columns = ", ".join(evidence.columns) if evidence.columns else "none"
    st.write(f"Columns: {columns}")
    st.write(f"Formula: {evidence.formula}")
    st.write(f"Records analyzed: {evidence.records_analyzed:,}")
    st.write(f"Filters and assumptions: {evidence.filters}")
    if evidence.method_notes:
        st.write(f"Method: {evidence.method_notes}")


def _load_sample() -> None:
    try:
        bundle = _read_sample()
    except DatasetError as exc:
        st.session_state["flash"] = None
        st.error(exc.user_message)
        return
    _adopt(bundle)
    st.session_state["flash"] = (
        f"Loaded {bundle.source_name}: {len(bundle.frame):,} rows and {bundle.frame.shape[1]} columns."
    )
    st.rerun()


def _read_sample() -> DatasetBundle:
    if not SAMPLE_PATH.exists():
        raise DatasetError(
            "The bundled sample file is missing. Restore data/sample_sales.csv and try again."
        )
    return load_csv_bytes(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name)


def _ensure_dataset() -> bool:
    if st.session_state.get("bundle") is not None:
        return True
    try:
        bundle = _read_sample()
    except DatasetError as exc:
        st.error(exc.user_message)
        return False
    _adopt(bundle)
    st.session_state["messages"].append(
        {
            "role": "system",
            "content": (
                "I loaded the bundled sample sales dataset so this question has real rows to use. "
                "You can replace it from Upload Dataset."
            ),
            "fingerprint": bundle.fingerprint,
        }
    )
    return True


def _adopt(bundle: DatasetBundle) -> None:
    state = {
        "dataset_fingerprint": st.session_state.get("dataset_fingerprint"),
        "messages": st.session_state.get("messages", []),
        "last_context": st.session_state.get("last_context", {}),
        "explained_ids": st.session_state.get("explained_ids", []),
    }
    replace_dataset(state, bundle.fingerprint)
    st.session_state["bundle"] = bundle
    st.session_state["dataset_fingerprint"] = state["dataset_fingerprint"]
    st.session_state["messages"] = state["messages"]
    st.session_state["last_context"] = state["last_context"]
    st.session_state["explained_ids"] = state["explained_ids"]
    if state["messages"] == [] and st.session_state.get("visual_result"):
        stored = st.session_state["visual_result"]
        if stored and stored.get("fingerprint") != bundle.fingerprint:
            st.session_state["visual_result"] = None


def _require_dataset() -> DatasetBundle | None:
    bundle = st.session_state.get("bundle")
    if bundle is not None:
        return bundle
    st.info("Load the sample sales file or upload a CSV to use this page.")
    if st.button("Load sample sales", key=f"empty_{st.session_state['section']}"):
        _load_sample()
    return None


def _preview(bundle: DatasetBundle) -> None:
    st.dataframe(bundle.frame.head(15), width="stretch", hide_index=True)
    if len(bundle.frame) > 15:
        st.caption(f"Showing the first 15 of {len(bundle.frame):,} rows.")


def _show_warnings(bundle: DatasetBundle) -> None:
    for warning in bundle.warnings:
        st.warning(warning)


def _mode_banner() -> None:
    _region, _model, enabled = _resolved_bedrock()
    if enabled:
        preview = " A 3-row preview is included." if st.session_state["include_preview"] else " Cell values stay local."
        st.caption("Amazon Bedrock is on. Your question and column schema will be sent." + preview)
    else:
        st.caption("Demo Mode is on. Answers come from local Pandas calculations. Amazon Bedrock is not being called.")


def _resolved_bedrock() -> tuple[str, str, bool]:
    settings = get_settings()
    region = (st.session_state.get("region_override") or settings.region).strip()
    model = (st.session_state.get("model_id_override") or settings.model_id).strip()
    enabled = bool(st.session_state.get("use_bedrock") and region and model)
    return region, model, enabled


def _orchestrator_for_request() -> tuple[Orchestrator, str | None]:
    region, model, enabled = _resolved_bedrock()
    if st.session_state.get("use_bedrock") and not enabled:
        return Orchestrator(use_bedrock=False), (
            "Bedrock is switched on, but the region or model ID is blank, so this answer stayed in Demo Mode."
        )
    return Orchestrator(use_bedrock=enabled, region=region, model_id=model), None


def _test_bedrock(region: str, model: str) -> None:
    if not region or not model:
        st.session_state["bedrock_verified"] = False
        st.session_state["bedrock_status"] = (
            "Add a region and a model ID that is enabled in your account before testing."
        )
        st.rerun()
    try:
        result = BedrockClient(region, model).test_connection()
    except BedrockError as exc:
        st.session_state["bedrock_verified"] = False
        st.session_state["bedrock_status"] = exc.user_message
        st.rerun()
    st.session_state["bedrock_verified"] = True
    st.session_state["bedrock_status"] = (
        "A test request succeeded in this session. Model reply: " + result["reply"]
    )
    st.rerun()


def _suggested_chart(bundle: DatasetBundle) -> tuple[str, str, str]:
    kinds = bundle.kinds
    dates = [name for name, kind in kinds.items() if kind == "datetime"]
    numeric = [name for name, kind in kinds.items() if kind == "numeric"]
    categorical = [name for name, kind in kinds.items() if kind == "categorical"]
    metric = numeric[0] if numeric else ""
    from analytics.metrics import default_metric_name

    preferred = default_metric_name(bundle.frame)
    if preferred:
        metric = preferred
    if dates and metric:
        return dates[0], metric, "line"
    if categorical and metric:
        return categorical[0], metric, "bar"
    if metric:
        return metric, "", "histogram"
    return "", "", "bar"


def _analysis_entries(fingerprint: str) -> list[dict]:
    entries = []
    for message in st.session_state.get("messages") or []:
        response = message.get("response")
        if message.get("role") != "assistant" or response is None:
            continue
        if message.get("fingerprint") != fingerprint:
            continue
        entries.append(
            {
                "question": message.get("question") or "",
                "answer": response.answer,
                "mode_label": response.mode_label,
                "findings": response.findings,
                "limitations": response.limitations,
                "evidences": [
                    {
                        "columns": evidence.columns,
                        "formula": evidence.formula,
                        "records_analyzed": evidence.records_analyzed,
                        "filters": evidence.filters,
                        "method_notes": evidence.method_notes,
                    }
                    for evidence in response.evidences
                ],
            }
        )
    return entries


def _current_charts(fingerprint: str) -> list[tuple[str, object]]:
    charts: list[tuple[str, object]] = []
    for message in st.session_state.get("messages") or []:
        response = message.get("response")
        if message.get("fingerprint") != fingerprint or response is None:
            continue
        charts.extend(response.figures)
    stored = st.session_state.get("visual_result")
    if stored and stored.get("fingerprint") == fingerprint:
        result = stored["result"]
        if result.ok and result.figure is not None:
            charts.append((result.figure_title or "Visualization", result.figure))
    return charts


def _escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
