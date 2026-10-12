"""Choose a check, run it, and explain only what came back."""

from __future__ import annotations

import re

import pandas as pd

from agent.intent import ToolCall, _direction, route
from agent.models import AgentResponse
from agent.narration import clarify, compose
from agent.tool_registry import execute
from aws.bedrock_client import BedrockClient, BedrockError

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


class Orchestrator:
    def __init__(
        self,
        use_bedrock: bool = False,
        llm: object | None = None,
        region: str = "",
        model_id: str = "",
    ):
        self.use_bedrock = use_bedrock
        self.llm = llm
        self.region = region
        self.model_id = model_id

    def answer(
        self,
        question: str,
        frame: pd.DataFrame,
        fingerprint: str,
        prior: dict[str, str] | None = None,
        dataset_warnings: list[str] | None = None,
        include_preview: bool = False,
    ) -> AgentResponse:
        if self.use_bedrock:
            try:
                model = self.llm or BedrockClient(self.region, self.model_id)
                return self._with_model(
                    model,
                    question,
                    frame,
                    fingerprint,
                    prior,
                    dataset_warnings,
                    include_preview,
                )
            except BedrockError as exc:
                local = self._local(question, frame, fingerprint, prior, dataset_warnings)
                local.mode = "local_fallback"
                local.warnings.insert(
                    0,
                    f"{exc.user_message} I used local Demo Mode for this answer instead.",
                )
                return local
        return self._local(question, frame, fingerprint, prior, dataset_warnings)

    def _local(
        self,
        question: str,
        frame: pd.DataFrame,
        fingerprint: str,
        prior: dict[str, str] | None,
        dataset_warnings: list[str] | None,
    ) -> AgentResponse:
        plan = route(question, frame, prior)
        if plan.clarification and not plan.calls:
            return clarify(question, plan.clarification, frame, fingerprint, dataset_warnings)
        results = [execute(call.name, call.arguments, frame) for call in plan.calls]
        return compose(question, plan, results, frame, fingerprint, dataset_warnings)

    def _with_model(
        self,
        model: object,
        question: str,
        frame: pd.DataFrame,
        fingerprint: str,
        prior: dict[str, str] | None,
        dataset_warnings: list[str] | None,
        include_preview: bool,
    ) -> AgentResponse:
        schema = dataset_schema_text(frame, include_preview=include_preview)
        calls = model.select_tools(question, schema, prior_text(prior))  # type: ignore[attr-defined]
        if not calls:
            raise BedrockError(
                "malformed",
                "Amazon Bedrock did not return a usable tool request.",
            )
        typed = [call if isinstance(call, ToolCall) else ToolCall(call.name, dict(call.arguments)) for call in calls[:4]]
        results = [execute(call.name, call.arguments, frame) for call in typed]
        if not any(result.ok for result in results):
            raise BedrockError(
                "malformed",
                "Amazon Bedrock asked for a calculation that does not fit this dataset.",
            )
        from agent.intent import RoutePlan

        plan = RoutePlan(calls=typed, direction=_direction(question.lower()))
        response = compose(question, plan, results, frame, fingerprint, dataset_warnings)
        source = response.answer + "\n" + "\n".join(response.findings)
        try:
            rewritten = model.explain(question, source)  # type: ignore[attr-defined]
        except BedrockError as exc:
            response.mode = "bedrock"
            response.warnings.append(
                "Amazon Bedrock selected the checks, and the wording below is the verified local explanation. "
                + exc.user_message
            )
            return response
        if rewritten and numbers_supported(rewritten, source):
            follow = ""
            if response.next_questions and "Want to investigate further?" not in rewritten:
                follow = f"\n\nWant to investigate further? Try: {response.next_questions[0]}"
            response.answer = rewritten.strip() + follow
        else:
            response.warnings.append(
                "Amazon Bedrock rewrote the explanation with figures that were not in the calculation, "
                "so DataBuddy kept the verified wording."
            )
        response.mode = "bedrock"
        return response


def dataset_schema_text(frame: pd.DataFrame, include_preview: bool = False) -> str:
    """Column names and kinds only, unless the user opted into a 3-row preview."""

    from analytics.dataset_loader import classify_frame

    kinds = classify_frame(frame)
    lines = [f"Rows: {len(frame)}", "Columns:"]
    rows = max(len(frame), 1)
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        percent = (missing / rows * 100) if len(frame) else 0
        lines.append(f"- {column} | kind={kinds[str(column)]} | missing={missing} ({percent:.1f}%)")
    if include_preview and len(frame):
        preview = frame.head(3).copy()
        for column in preview.columns:
            preview[column] = preview[column].map(lambda value: "" if pd.isna(value) else str(value)[:80])
        lines.append("Preview of 3 rows, included because the user opted in:")
        lines.append(preview.to_csv(index=False))
    else:
        lines.append("Cell values were not included.")
    return "\n".join(lines)


def prior_text(prior: dict[str, str] | None) -> str:
    if not prior:
        return "No previous calculation."
    allowed = ("group_column", "metric_column", "aggregation", "date_column", "frequency", "direction")
    bits = [f"{key}={prior[key]}" for key in allowed if prior.get(key)]
    if not bits:
        return "No previous calculation."
    return "Previous calculation used " + ", ".join(bits) + "."


def numbers_supported(rewritten: str, source: str) -> bool:
    """Every number in the rewrite must already appear in the verified draft."""

    allowed = _canonical_numbers(source)
    for token in _NUMBER.findall(rewritten or ""):
        if _canonical_numbers(token).isdisjoint(allowed):
            return False
    return True


def _canonical_numbers(text: str) -> set[str]:
    found: set[str] = set()
    for token in _NUMBER.findall(text or ""):
        raw = token.replace("%", "").replace(",", "")
        found.add(raw)
        if re.fullmatch(r"\d+\.0+", raw):
            found.add(raw.split(".")[0])
        if re.fullmatch(r"\d+", raw):
            found.update({raw + ".0", raw + ".00", raw + ".000"})
    return found
