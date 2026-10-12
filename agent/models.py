"""Response objects shared by the router, narrator, and interface."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from agent.tool_registry import Evidence

MODE_LABELS = {
    "demo": "Demo Mode · local calculations only",
    "bedrock": "Amazon Bedrock chose the checks · figures calculated locally",
    "local_fallback": "Local fallback · Amazon Bedrock did not complete this request",
}


@dataclass
class AgentResponse:
    answer: str
    findings: list[str]
    evidences: list[Evidence]
    tables: list[tuple[str, pd.DataFrame]]
    figures: list[tuple[str, Any]]
    limitations: list[str]
    next_questions: list[str]
    explanation: str
    mode: str
    tools_used: list[str]
    warnings: list[str]
    fingerprint: str
    context: dict[str, str] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)

    @property
    def mode_label(self) -> str:
        return MODE_LABELS.get(self.mode, self.mode)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "findings": self.findings,
            "limitations": self.limitations,
            "next_questions": self.next_questions,
            "explanation": self.explanation,
            "mode": self.mode,
            "mode_label": self.mode_label,
            "tools_used": self.tools_used,
            "warnings": self.warnings,
            "fingerprint": self.fingerprint,
            "assumptions": self.assumptions,
            "evidences": [
                {
                    "columns": evidence.columns,
                    "formula": evidence.formula,
                    "records_analyzed": evidence.records_analyzed,
                    "filters": evidence.filters,
                    "method_notes": evidence.method_notes,
                }
                for evidence in self.evidences
            ],
            "tables": [
                {"title": title, "rows": _records(table)}
                for title, table in self.tables
            ],
            "charts": [title for title, _figure in self.figures],
        }


def _records(frame: pd.DataFrame, limit: int = 30) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    view = frame.head(limit).copy()
    for column in view.columns:
        if pd.api.types.is_datetime64_any_dtype(view[column]):
            view[column] = view[column].astype(str)
    payload = view.to_dict(orient="records")
    return [_json_ready(row) for row in payload]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value
