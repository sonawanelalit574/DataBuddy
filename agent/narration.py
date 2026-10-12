"""Turn verified tool results into plain language.

The wording is allowed to be friendly. The numbers have to come from the tool.
"""

from __future__ import annotations

import re
from typing import Any

from agent.intent import RoutePlan, suggested_questions
from agent.models import AgentResponse
from agent.tool_registry import Evidence, ToolResult


def compose(
    question: str,
    plan: RoutePlan,
    results: list[ToolResult],
    frame: pd.DataFrame,
    fingerprint: str,
    dataset_warnings: list[str] | None = None,
) -> AgentResponse:
    paragraphs: list[str] = []
    findings: list[str] = []
    tables: list[tuple[str, pd.DataFrame]] = []
    figures: list[tuple[str, Any]] = []
    evidences: list[Evidence] = []
    limitations = list(plan.limitations)
    assumptions = list(plan.assumptions)
    context: dict[str, str] = {}
    tools_used: list[str] = []

    if assumptions:
        paragraphs.append(" ".join(assumptions))

    for result in results:
        tools_used.append(result.name)
        context.update(result.context_update)
        if not result.ok:
            paragraphs.append(result.message)
            limitations.append("That part of the question did not produce a calculation.")
            continue
        paragraphs.append(_prose(result, plan.direction))
        findings.extend(_findings(result))
        if result.table is not None and not result.table.empty:
            tables.append((result.table_title or result.name, result.table))
        if result.figure is not None:
            figures.append((result.figure_title or result.name, result.figure))
        evidences.extend(result.evidences)
        limitations.extend(result.limitations)

    if "calculate_group_metrics" in tools_used:
        context["direction"] = plan.direction

    suggestions = _suggestions_from(results, frame)
    next_questions = _next_questions(suggestions, question, tools_used, context)
    answer = "\n\n".join(part for part in paragraphs if part).strip()
    if next_questions and not _is_suggestion_only(results):
        answer = f"{answer}\n\nWant to investigate further? Try: {next_questions[0]}"
    if dataset_warnings:
        limitations.extend(dataset_warnings)
    # Preserve order while dropping repeated limitation lines.
    seen: list[str] = []
    for item in limitations:
        if item and item not in seen:
            seen.append(item)

    return AgentResponse(
        answer=answer or "I could not turn that request into a calculation.",
        findings=findings,
        evidences=evidences,
        tables=tables,
        figures=figures,
        limitations=seen,
        next_questions=next_questions,
        explanation=explain_plain(evidences),
        mode="demo",
        tools_used=tools_used,
        warnings=[],
        fingerprint=fingerprint,
        context=context,
        assumptions=assumptions,
    )


def explain_plain(evidences: list[Evidence]) -> str:
    if not evidences:
        return (
            "I did not apply a formula to the dataset for this reply. "
            "Nothing in the message is a new measurement."
        )
    sentences = ["Here is what I actually did, in everyday language."]
    for index, evidence in enumerate(evidences, start=1):
        columns = ", ".join(evidence.columns) if evidence.columns else "the columns listed with the result"
        prefix = f"Step {index}: " if len(evidences) > 1 else ""
        note = f" {evidence.method_notes}" if evidence.method_notes else ""
        sentences.append(
            f"{prefix}I looked at {evidence.records_analyzed:,} rows. "
            f"I used {columns}. "
            f"The calculation was: {evidence.formula}. "
            f"I treated the data this way: {evidence.filters}.{note}"
        )
    sentences.append(
        "The figures above come from that calculation. I did not fill in blanks or add a number that the tool did not return."
    )
    return " ".join(sentences)


def clarify(
    question: str,
    message: str,
    frame: pd.DataFrame,
    fingerprint: str,
    dataset_warnings: list[str] | None = None,
) -> AgentResponse:
    suggestions = suggested_questions(frame)
    next_questions = [item for item in suggestions if _normalize(item) != _normalize(question)][:3]
    limitations = [
        "No calculation was run because the question was outside the checks I can do, or a required column was not clear."
    ]
    if dataset_warnings:
        limitations.extend(dataset_warnings)
    return AgentResponse(
        answer=message,
        findings=[],
        evidences=[],
        tables=[],
        figures=[],
        limitations=limitations,
        next_questions=next_questions,
        explanation=explain_plain([]),
        mode="demo",
        tools_used=[],
        warnings=[],
        fingerprint=fingerprint,
        context={},
        assumptions=[],
    )


def _prose(result: ToolResult, direction: str) -> str:
    data = result.data
    if result.name == "dataset_overview":
        counts = data["kind_counts"]
        return (
            f"This dataset has {data['rows']:,} rows and {data['columns']:,} columns, "
            f"using about {data['memory_label']}. "
            f"I see {counts.get('numeric', 0)} numeric, {counts.get('categorical', 0)} categorical, "
            f"{counts.get('datetime', 0)} date, and {counts.get('boolean', 0)} boolean columns. "
            f"There are {data['missing_cells']:,} missing cells and "
            f"{data['duplicate_extra']:,} extra duplicate rows."
            + (f" {data.get('chart_note')}" if data.get("chart_note") else "")
        )
    if result.name == "calculate_group_metrics":
        return _group_prose(data, direction)
    if result.name == "calculate_time_trend":
        return _trend_prose(data)
    if result.name == "find_missing_values":
        if data["missing_cells"] == 0:
            return "I did not find missing values in this dataset."
        return (
            f"I found {data['missing_cells']:,} missing cells across "
            f"{data['columns_with_gaps']:,} columns. The table lists every column with a gap."
        )
    if result.name == "find_duplicates":
        extra = data["extra_duplicate_rows"]
        if extra == 0:
            return "I did not find exact duplicate rows."
        return (
            f"I found {extra:,} extra duplicate rows. "
            f"{data['rows_in_duplicate_sets']:,} rows belong to a duplicated set, "
            "counting every copy."
        )
    if result.name == "detect_outliers":
        return _outlier_prose(data)
    if result.name == "calculate_correlation":
        value = data["correlation"]
        strength = _strength(abs(float(value)))
        return (
            f"The Pearson correlation between {data['column_a']} and {data['column_b']} "
            f"is {float(value):.3f}, based on {data['records']:,} complete pairs. "
            f"That is a {strength} straight-line association by a common rule of thumb, not evidence of cause."
        )
    if result.name == "get_column_summary":
        return _column_prose(data)
    if result.name == "create_visualization":
        return (
            f"I drew a {data['chart_type']} chart using {data['formula']}. "
            f"It is based on {data['records']:,} rows."
        )
    if result.name == "suggest_next_questions":
        lines = "\n".join(f"- {item}" for item in data.get("questions", []))
        return "These are questions I can answer from the columns you have:\n\n" + lines
    return result.message or "The calculation finished."


def _findings(result: ToolResult) -> list[str]:
    data = result.data
    if result.name == "dataset_overview":
        kinds = data.get("kinds", {})
        lines = []
        for kind in ("numeric", "categorical", "datetime", "boolean"):
            names = kinds.get(kind) or []
            if names:
                lines.append(f"{kind.capitalize()} columns: {', '.join(names)}.")
        if data["duplicate_extra"]:
            lines.append(
                f"Removing exact duplicates would leave {data['rows'] - data['duplicate_extra']:,} rows."
            )
        return lines
    if result.name == "calculate_group_metrics" and data.get("highest") and data.get("lowest"):
        return [
            f"Highest: {data['highest']['group']} at {fmt(data['highest']['value'])}.",
            f"Lowest: {data['lowest']['group']} at {fmt(data['lowest']['value'])}.",
            f"Groups compared: {data['groups']:,}. Rows used: {data['records']:,}.",
        ]
    if result.name == "calculate_time_trend" and data.get("peak"):
        lines = [
            f"Highest period: {data['peak']['period']} at {fmt(data['peak']['value'])}.",
        ]
        if data.get("trough"):
            lines.append(f"Lowest period: {data['trough']['period']} at {fmt(data['trough']['value'])}.")
        lines.append(f"Rows used: {data['records']:,}.")
        return lines
    if result.name == "detect_outliers":
        if data.get("lower") is not None and data.get("upper") is not None:
            return [
                f"Fence for {data['column']}: {fmt(data['lower'])} to {fmt(data['upper'])}.",
                f"Potential outliers: {data['outlier_count']:,} of {data['records']:,} numeric values.",
            ]
        return [str(data.get("note") or "The outlier screen did not produce fences.")]
    if result.name == "get_column_summary":
        return [
            f"{data['column']} is {data['kind']} with {data['unique']:,} distinct values "
            f"and {data['missing']:,} missing."
        ]
    if result.name == "suggest_next_questions":
        return ["No metric was calculated for this reply."]
    return []


def _group_prose(data: dict[str, Any], direction: str) -> str:
    label = {
        "sum": "total",
        "mean": "average",
        "median": "median",
        "min": "minimum",
        "max": "maximum",
        "count": "count",
    }.get(data["aggregation"], data["aggregation"])
    if not data.get("highest"):
        return (
            f"I grouped {data['group_column']} but there were no usable {data['metric_column']} values to rank."
        )
    focus = data["lowest"] if direction == "bottom" else data["highest"]
    word = "lowest" if direction == "bottom" else "highest"
    return (
        f"Here is how {data['group_column']} compares on the {label} of {data['metric_column']}. "
        f"The {word} result in this dataset is {focus['group']} at {fmt(focus['value'])}. "
        f"I ranked {data['groups']:,} groups from {data['records']:,} rows, and the table has the rest."
    )


def _trend_prose(data: dict[str, Any]) -> str:
    if not data.get("peak"):
        return (
            f"I could not build a {data['frequency_label']} trend because no rows had both "
            f"a readable {data['date_column']} and a numeric {data['metric_column']}."
        )
    if data.get("periods") == 1:
        return (
            f"Only one {data['frequency_label']} is present ({data['peak']['period']}), "
            f"so this is a total rather than a trend. The sum of {data['metric_column']} is "
            f"{fmt(data['peak']['value'])}."
        )
    return (
        f"From {data['start']} to {data['end']}, the {data['frequency_label']} sum of "
        f"{data['metric_column']} is highest in {data['peak']['period']} at {fmt(data['peak']['value'])} "
        f"and lowest in {data['trough']['period']} at {fmt(data['trough']['value'])}. "
        "Months or periods with no rows are left out, not treated as zero."
    )


def _outlier_prose(data: dict[str, Any]) -> str:
    if data.get("note") and not data.get("outlier_count") and data.get("lower") is None:
        return str(data["note"])
    count = int(data.get("outlier_count") or 0)
    if count == 0:
        return (
            f"I did not find potential outliers in {data['column']} using Tukey fences "
            f"on {data['records']:,} numeric values."
        )
    listed = ""
    flagged = data.get("flagged_values") or []
    if count <= 5 and flagged:
        listed = " Flagged value(s): " + ", ".join(fmt(value) for value in flagged) + "."
    return (
        f"I found {count:,} potential outlier(s) in {data['column']}. "
        f"The Tukey fences run from {fmt(data['lower'])} to {fmt(data['upper'])}."
        f"{listed} The box plot shows the spread, and the table lists the flagged rows."
    )


def _column_prose(data: dict[str, Any]) -> str:
    base = (
        f"{data['column']} is a {data['kind']} column with {data['unique']:,} distinct values "
        f"and {data['missing']:,} missing out of {data['records']:,} rows."
    )
    stats = data.get("stats") or {}
    if "mean" in stats and stats["mean"] is not None:
        return (
            f"{base} The average is {fmt(stats['mean'])}, "
            f"the minimum is {fmt(stats['min'])}, and the maximum is {fmt(stats['max'])}."
        )
    if data.get("minimum") or data.get("maximum"):
        return f"{base} The dates run from {data.get('minimum')} to {data.get('maximum')}."
    top = data.get("top_values") or []
    if top:
        lead = top[0]
        return f"{base} The most common value is {lead['value']} ({lead['count']:,} rows)."
    return base


def _next_questions(
    suggestions: list[str],
    question: str,
    tools_used: list[str],
    context: dict[str, str],
) -> list[str]:
    def keep(item: str, avoid_repeat: bool) -> bool:
        if _normalize(item) == _normalize(question):
            return False
        if not avoid_repeat:
            return True
        group = context.get("group_column", "")
        metric = context.get("metric_column", "")
        date_column = context.get("date_column", "")
        if (
            "calculate_group_metrics" in tools_used
            and group
            and metric
            and _mentions(group, item)
            and _mentions(metric, item)
        ):
            return False
        if (
            "calculate_time_trend" in tools_used
            and date_column
            and metric
            and _mentions(date_column, item)
            and _mentions(metric, item)
        ):
            return False
        return True

    chosen = [item for item in suggestions if keep(item, True)]
    if not chosen:
        chosen = [item for item in suggestions if keep(item, False)]
    return chosen[:3]


def _mentions(column: str, text: str) -> bool:
    token = column.lower().replace("_", " ")
    haystack = text.lower().replace("_", " ")
    return re.search(rf"\b{re.escape(token)}\b", haystack) is not None


def _suggestions_from(results: list[ToolResult], frame: pd.DataFrame) -> list[str]:
    for result in results:
        if result.ok and result.name == "suggest_next_questions":
            return list(result.data.get("questions") or [])
    return suggested_questions(frame)


def _is_suggestion_only(results: list[ToolResult]) -> bool:
    names = [result.name for result in results if result.ok]
    return names == ["suggest_next_questions"]


def _strength(value: float) -> str:
    if value >= 0.7:
        return "strong"
    if value >= 0.4:
        return "moderate"
    return "weak"


def fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
