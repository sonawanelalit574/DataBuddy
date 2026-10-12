"""Exports built from calculations that already ran."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from analytics.data_quality import quality_table


def cleaned_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows only.

    Missing values stay missing. Filling them would invent data.
    """

    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned, before - len(cleaned)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def quality_csv_bytes(frame: pd.DataFrame) -> bytes:
    return csv_bytes(quality_table(frame))


def analysis_markdown(dataset_name: str, fingerprint: str, entries: list[dict[str, Any]]) -> str:
    lines = [
        "# DataBuddy analysis summary",
        "",
        f"Dataset: {dataset_name}",
        f"Fingerprint: {fingerprint}",
        "",
        "Figures in this file come from the calculations recorded in the session. "
        "They are not estimates from a language model.",
        "",
    ]
    if not entries:
        lines.append("No questions have been answered for this dataset yet.")
        lines.append("")
        return "\n".join(lines)
    for index, entry in enumerate(entries, start=1):
        lines.append(f"## {index}. {entry.get('question') or 'Question'}")
        lines.append("")
        lines.append(f"Mode: {entry.get('mode_label') or 'Unknown'}")
        lines.append("")
        lines.append(entry.get("answer") or "")
        lines.append("")
        findings = entry.get("findings") or []
        if findings:
            lines.append("Findings:")
            lines.append("")
            for finding in findings:
                lines.append(f"- {finding}")
            lines.append("")
        for evidence in entry.get("evidences") or []:
            lines.append("How it was calculated:")
            lines.append("")
            columns = ", ".join(evidence.get("columns") or []) or "none"
            lines.append(f"- Columns: {columns}")
            lines.append(f"- Formula: {evidence.get('formula') or 'n/a'}")
            lines.append(f"- Records analyzed: {evidence.get('records_analyzed')}")
            lines.append(f"- Filters and assumptions: {evidence.get('filters') or 'none'}")
            if evidence.get("method_notes"):
                lines.append(f"- Method: {evidence.get('method_notes')}")
            lines.append("")
        limitations = entry.get("limitations") or []
        if limitations:
            lines.append("Limitations:")
            lines.append("")
            for item in limitations:
                lines.append(f"- {item}")
            lines.append("")
    return "\n".join(lines)


def figure_html(figure: Any) -> str:
    return figure.to_html(include_plotlyjs=True, full_html=True)


def figure_png(figure: Any) -> bytes | None:
    try:
        payload = figure.to_image(format="png", scale=2)
    except Exception:
        return None
    if isinstance(payload, bytes):
        return payload
    return bytes(payload)
