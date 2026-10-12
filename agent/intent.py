"""Local question routing for Demo Mode.

This module never calculates a metric. It only chooses a validated tool
and the columns that tool should see. If a column or an operation is
unclear, it asks for clarification instead of guessing a result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from analytics.dataset_loader import classify_frame
from analytics.metrics import columns_of_kind, default_metric_name, strongest_pair

SYNONYMS: dict[str, tuple[str, ...]] = {
    "product": ("product", "products", "item", "items", "sku"),
    "category": ("category", "categories"),
    "region": ("region", "regions"),
    "revenue": ("revenue", "sales"),
    "profit": ("profit", "profits", "profitability"),
    "quantity": ("quantity", "quantities", "units"),
    "unit_price": ("unit price", "price"),
    "order_date": ("order date", "date"),
}

GROUP_KEYS = ("product", "category", "region")
METRIC_KEYS = ("revenue", "profit", "quantity", "unit_price")


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, str]


@dataclass
class RoutePlan:
    calls: list[ToolCall] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    clarification: str | None = None
    direction: str = "top"
    limitations: list[str] = field(default_factory=list)


def route(question: str, frame: pd.DataFrame, prior: dict[str, str] | None = None) -> RoutePlan:
    text = (question or "").strip()
    if not text:
        return RoutePlan(clarification="Ask a question about the dataset you loaded.")
    if frame.empty:
        return RoutePlan(
            clarification="This dataset has column names but no rows, so there is nothing to calculate yet."
        )

    lowered = text.lower()
    kinds = classify_frame(frame)
    plan = RoutePlan()
    wants_missing = bool(re.search(r"\b(missing|nulls?|blank values?|empty values?|n/?a)\b", lowered))
    wants_dupes = "duplicate" in lowered
    wants_outliers = bool(re.search(r"\b(outliers?|unusual|anomal(?:y|ies))\b", lowered))
    wants_corr = "correlat" in lowered
    wants_heatmap = "heatmap" in lowered or "heat map" in lowered
    wants_trend = bool(
        re.search(r"\b(trend|over time|time series|monthly|weekly|daily|quarterly|yearly)\b", lowered)
    )
    wants_overview = bool(
        re.search(r"\b(overview|quick look)\b", lowered)
        or bool(re.search(r"\b(summarize|describe|tell me about)\b.{0,40}\b(dataset|data|file|table)\b", lowered))
    )
    wants_next = bool(re.search(r"\b(investigate next|what should i|what next|next questions?)\b", lowered))
    wants_chart = bool(
        re.search(r"\b(chart|plot|histogram|scatter|box plot|boxplot|bar chart|line chart|heatmap|heat map)\b", lowered)
    )
    wants_group = bool(
        re.search(
            r"\b(top|bottom|best|worst|underperform\w*|highest|lowest|rank|compare|average|avg|mean|total|which|most|least)\b",
            lowered,
        )
    ) or bool(re.search(r"\bby\s+[a-z0-9_]", lowered))
    forecast = bool(re.search(r"\b(predict|forecast|machine learning|regression)\b", lowered))

    specific = any(
        [wants_missing, wants_dupes, wants_outliers, wants_corr, wants_trend, wants_group, wants_chart, wants_next]
    )
    if wants_overview and not specific:
        plan.calls.append(ToolCall("dataset_overview", {}))

    if wants_missing:
        plan.calls.append(ToolCall("find_missing_values", {}))
    if wants_dupes:
        plan.calls.append(ToolCall("find_duplicates", {}))

    if wants_outliers:
        targets, notes = _outlier_targets(text, frame, kinds)
        if targets:
            plan.assumptions.extend(notes)
            for column in targets:
                plan.calls.append(ToolCall("detect_outliers", {"column_name": column}))
        elif not plan.calls:
            return RoutePlan(clarification=_need_kind(frame, "numeric", "screen for outliers"))

    if wants_corr or wants_heatmap:
        added, error, notes = _plan_correlation(text, frame, kinds, wants_heatmap or wants_corr)
        plan.assumptions.extend(notes)
        if added:
            plan.calls.extend(added)
        elif error and not plan.calls:
            return RoutePlan(clarification=error)
        elif error:
            plan.assumptions.append(error)

    if wants_trend:
        call, error, notes, freq_limit = _plan_trend(lowered, text, frame, kinds)
        plan.assumptions.extend(notes)
        if call:
            plan.calls.append(call)
            plan.limitations.extend(freq_limit)
        elif error and not plan.calls:
            return RoutePlan(clarification=error)
        elif error:
            plan.assumptions.append(error)

    if wants_group and not (wants_trend and not _group_explicit(lowered)):
        call, direction, error, notes = _plan_group(lowered, text, frame, kinds)
        if call:
            plan.calls.append(call)
            plan.direction = direction
            plan.assumptions.extend(notes)
        elif error and not plan.calls:
            return RoutePlan(clarification=error, direction=direction)
        elif error:
            plan.assumptions.append(error)

    if wants_chart and not _chart_already_covered(wants_trend, wants_heatmap, lowered):
        call, error, notes = _plan_chart(lowered, text, frame, kinds)
        plan.assumptions.extend(notes)
        if call:
            plan.calls.append(call)
        elif error and not plan.calls:
            return RoutePlan(clarification=error)
        elif error:
            plan.assumptions.append(error)

    if wants_next and not any(call.name != "suggest_next_questions" for call in plan.calls):
        plan.calls = [ToolCall("suggest_next_questions", {})]

    if not plan.calls:
        summary = _plan_column_summary(lowered, text, frame)
        if summary:
            plan.calls.append(summary)

    if not plan.calls:
        follow = _plan_follow_up(lowered, text, frame, kinds, prior or {})
        if follow is not None:
            plan = follow

    if not plan.calls:
        if forecast:
            return RoutePlan(
                clarification=(
                    "I don't forecast or train models. I can show the historical trend, "
                    "rank categories, or check correlations using the columns in this file.\n\n"
                    + clarification_message(frame, "")
                )
            )
        return RoutePlan(clarification=clarification_message(frame))

    if forecast and wants_trend:
        plan.limitations.append(
            "You asked about the future. I am only describing the dates already in the file."
        )
    plan.calls = plan.calls[:4]
    return plan


def suggested_questions(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["Load a CSV with at least one data row, then ask for an overview."]
    kinds = classify_frame(frame)
    metric = default_metric_name(frame)
    group = _preferred_group(list(frame.columns), kinds)
    dates = columns_of_kind(frame, "datetime")
    numeric = columns_of_kind(frame, "numeric")
    questions: list[str] = []
    if group and metric:
        questions.append(f"Which {group} has the highest total {metric}?")
        questions.append(f"Which {group} has the lowest total {metric}?")
    if dates and metric:
        questions.append(f"Show monthly {metric} using {dates[0]}.")
    if len(numeric) >= 2:
        left = metric or numeric[0]
        right = next(column for column in numeric if column != left)
        questions.append(f"What is the correlation between {left} and {right}?")
    if int(frame.isna().sum().sum()):
        questions.append("Check for missing values and duplicates.")
    else:
        questions.append("Find unusual values and potential outliers.")
    questions.append("Give me a quick overview of this dataset.")
    unique: list[str] = []
    for question in questions:
        if question not in unique:
            unique.append(question)
    return unique[:4]


def clarification_message(frame: pd.DataFrame, lead: str | None = None) -> str:
    columns = ", ".join(map(str, frame.columns)) or "none"
    opening = lead if lead is not None else (
        "I can run a fixed set of checks, and that question does not match one of them "
        "closely enough for me to guess."
    )
    body = (
        "I can do these things when the columns exist:\n"
        "- Summarize the dataset\n"
        "- Rank a category by a numeric column\n"
        "- Show a time trend\n"
        "- Check missing values and duplicates\n"
        "- Screen a numeric column for outliers\n"
        "- Correlate two numeric columns\n"
        "- Draw a bar, line, histogram, scatter, box, or heatmap\n\n"
        f"Columns in this dataset: {columns}."
    )
    return f"{opening}\n\n{body}" if opening else body


def _plan_group(lowered: str, question: str, frame: pd.DataFrame, kinds: dict[str, str]):
    direction = _direction(lowered)
    aggregation = _aggregation(lowered)
    assumptions: list[str] = []
    mentioned = mentioned_columns(question, list(frame.columns))
    group_column = next((column for column in mentioned if kinds.get(column) in {"categorical", "boolean"}), None)
    metric_column = next((column for column in mentioned if kinds.get(column) == "numeric"), None)
    if group_column is None:
        group_column = _synonym_column(lowered, list(frame.columns), kinds, GROUP_KEYS, {"categorical", "boolean"})
    if metric_column is None:
        metric_column = _synonym_column(lowered, list(frame.columns), kinds, METRIC_KEYS, {"numeric"})
    if metric_column is None and aggregation != "count":
        metric_column = default_metric_name(frame)
        if metric_column:
            assumptions.append(
                f"The question did not name a numeric measure, so I used {metric_column}."
            )
    if group_column is None:
        categorical = [name for name, kind in kinds.items() if kind in {"categorical", "boolean"}]
        if len(categorical) == 1:
            group_column = categorical[0]
            assumptions.append(f"I grouped by {group_column} because it is the only category column.")
        else:
            listed = ", ".join(categorical) if categorical else "none"
            named = re.search(r"\bby\s+([a-z0-9_]+(?:\s+[a-z0-9_]+){0,2})", lowered)
            if named:
                return None, direction, (
                    f"I can't find a category column matching '{named.group(1).strip()}'. "
                    f"Category columns in this dataset: {listed}."
                ), assumptions
            return None, direction, (
                "I need a category column to rank. "
                f"Category columns in this dataset: {listed}."
            ), assumptions
    if metric_column is None:
        numeric = ", ".join(columns_of_kind(frame, "numeric")) or "none"
        return None, direction, (
            f"I need a numeric column to aggregate. Numeric columns: {numeric}."
        ), assumptions
    if group_column == metric_column:
        return None, direction, "Choose a category column and a different numeric column.", assumptions
    return (
        ToolCall(
            "calculate_group_metrics",
            {
                "group_column": group_column,
                "metric_column": metric_column,
                "aggregation": aggregation,
            },
        ),
        direction,
        None,
        assumptions,
    )


def _plan_trend(lowered: str, question: str, frame: pd.DataFrame, kinds: dict[str, str]):
    assumptions: list[str] = []
    frequency = "ME"
    if re.search(r"\b(daily|per day|by day)\b", lowered):
        frequency = "D"
    elif re.search(r"\b(weekly|per week|by week)\b", lowered):
        frequency = "W"
    elif "quarter" in lowered:
        frequency = "QE"
    elif re.search(r"\b(yearly|annual|per year|by year)\b", lowered):
        frequency = "YE"
    dates = [name for name, kind in kinds.items() if kind == "datetime"]
    mentioned = [column for column in mentioned_columns(question, list(frame.columns)) if kinds.get(column) == "datetime"]
    if mentioned:
        date_column = mentioned[0]
    elif len(dates) == 1:
        date_column = dates[0]
        assumptions.append(f"I used {date_column} as the date column.")
    elif not dates:
        return None, _need_kind(frame, "datetime", "build a time trend"), assumptions, []
    else:
        return None, (
            "This dataset has more than one date column. Tell me which to use: " + ", ".join(dates) + "."
        ), assumptions, []
    metric = next(
        (column for column in mentioned_columns(question, list(frame.columns)) if kinds.get(column) == "numeric"),
        None,
    )
    if metric is None:
        metric = _synonym_column(lowered, list(frame.columns), kinds, METRIC_KEYS, {"numeric"})
    if metric is None:
        metric = default_metric_name(frame)
        if metric:
            assumptions.append(f"The question did not name a measure, so I summed {metric}.")
    if metric is None:
        return None, _need_kind(frame, "numeric", "sum over time"), assumptions, []
    return (
        ToolCall(
            "calculate_time_trend",
            {"date_column": date_column, "metric_column": metric, "frequency": frequency},
        ),
        None,
        assumptions,
        [],
    )


def _plan_correlation(question: str, frame: pd.DataFrame, kinds: dict[str, str], include_matrix: bool):
    numeric_mentioned = [
        column for column in mentioned_columns(question, list(frame.columns)) if kinds.get(column) == "numeric"
    ]
    numeric = columns_of_kind(frame, "numeric")
    notes: list[str] = []
    if len(numeric_mentioned) >= 2:
        return (
            [ToolCall("calculate_correlation", {"column_a": numeric_mentioned[0], "column_b": numeric_mentioned[1]})],
            None,
            notes,
        )
    if len(numeric_mentioned) == 1:
        listed = ", ".join(column for column in numeric if column != numeric_mentioned[0]) or "none"
        return [], (
            f"Name a second numeric column to correlate with {numeric_mentioned[0]}. Other numeric columns: {listed}."
        ), notes
    if len(numeric) < 2:
        return [], _need_kind(frame, "numeric", "calculate a correlation") + " A correlation needs two.", notes
    pair = strongest_pair(frame)
    calls = []
    if include_matrix:
        calls.append(ToolCall("create_visualization", {"chart_type": "heatmap", "x_column": "", "y_column": ""}))
    if pair:
        calls.append(ToolCall("calculate_correlation", {"column_a": pair[0], "column_b": pair[1]}))
        notes.append(
            f"No pair was named, so I compared the numeric columns and highlighted {pair[0]} and {pair[1]}."
        )
    elif not calls:
        return [], "The numeric columns do not have enough complete rows for a correlation.", notes
    return calls, None, notes


def _plan_chart(lowered: str, question: str, frame: pd.DataFrame, kinds: dict[str, str]):
    chart_type = "bar"
    if "histogram" in lowered or "distribution" in lowered:
        chart_type = "histogram"
    elif "scatter" in lowered:
        chart_type = "scatter"
    elif "box" in lowered:
        chart_type = "box"
    elif "heatmap" in lowered or "heat map" in lowered:
        chart_type = "heatmap"
    elif "line" in lowered:
        chart_type = "line"
    mentioned = mentioned_columns(question, list(frame.columns))
    numeric = [column for column in mentioned if kinds.get(column) == "numeric"]
    categorical = [column for column in mentioned if kinds.get(column) in {"categorical", "boolean"}]
    dates = [column for column in mentioned if kinds.get(column) == "datetime"]
    assumptions: list[str] = []
    x_column = ""
    y_column = ""
    if chart_type == "heatmap":
        pass
    elif chart_type == "histogram":
        y_source = numeric or columns_of_kind(frame, "numeric")
        if not y_source:
            return None, _need_kind(frame, "numeric", "draw a histogram"), assumptions
        x_column = y_source[0]
        if not numeric:
            assumptions.append(f"I used {x_column} for the histogram.")
    elif chart_type == "scatter":
        pool = numeric or columns_of_kind(frame, "numeric")
        if len(pool) < 2:
            return None, "A scatter plot needs two numeric columns.", assumptions
        x_column, y_column = pool[0], pool[1]
    elif chart_type == "box":
        y_source = numeric or columns_of_kind(frame, "numeric")
        if not y_source:
            return None, _need_kind(frame, "numeric", "draw a box plot"), assumptions
        y_column = y_source[0]
        if categorical:
            x_column = categorical[0]
    elif chart_type == "line":
        x_source = dates or columns_of_kind(frame, "datetime") or categorical
        y_source = numeric or columns_of_kind(frame, "numeric")
        if not x_source or not y_source:
            return None, "A line chart needs an x column and a numeric y column.", assumptions
        x_column, y_column = x_source[0], y_source[0]
    else:
        x_source = categorical or _preferred_group(list(frame.columns), kinds)
        y_source = numeric[0] if numeric else default_metric_name(frame)
        x_column = x_source if isinstance(x_source, str) else (x_source[0] if x_source else "")
        y_column = y_source or ""
        if not x_column or not y_column:
            return None, "A bar chart needs a category column and a numeric column.", assumptions
        if not categorical:
            assumptions.append(f"I used {x_column} and {y_column} for the bar chart.")
    return ToolCall(
        "create_visualization",
        {"chart_type": chart_type, "x_column": x_column, "y_column": y_column},
    ), None, assumptions


def _plan_column_summary(lowered: str, question: str, frame: pd.DataFrame) -> ToolCall | None:
    if not re.search(r"\b(summarize|describe|summary of|profile|tell me about)\b", lowered):
        return None
    mentioned = mentioned_columns(question, list(frame.columns))
    if len(mentioned) == 1:
        return ToolCall("get_column_summary", {"column_name": mentioned[0]})
    return None


def _plan_follow_up(
    lowered: str,
    question: str,
    frame: pd.DataFrame,
    kinds: dict[str, str],
    prior: dict[str, str],
) -> RoutePlan | None:
    if not prior:
        return None
    if not re.search(r"\b(what about|how about|instead|break that|split that|same for|break this down)\b", lowered):
        if not re.match(r"^(and |now )?(by|for)\s+", lowered):
            return None
    mentioned = mentioned_columns(question, list(frame.columns))
    synonym_group = _synonym_column(lowered, list(frame.columns), kinds, GROUP_KEYS, {"categorical", "boolean"})
    synonym_metric = _synonym_column(lowered, list(frame.columns), kinds, METRIC_KEYS, {"numeric"})
    named = mentioned[0] if len(mentioned) == 1 else None
    group_column = prior.get("group_column", "")
    metric_column = prior.get("metric_column", "")
    aggregation = prior.get("aggregation", "sum")
    direction = prior.get("direction", "top")
    assumptions = ["I reused the previous calculation and changed only what this follow-up names."]
    if synonym_group or (named and kinds.get(named) in {"categorical", "boolean"}):
        group_column = synonym_group or named or group_column
    elif synonym_metric or (named and kinds.get(named) == "numeric"):
        metric_column = synonym_metric or named or metric_column
    elif not group_column or not metric_column:
        return None
    if not group_column or not metric_column:
        return None
    if group_column not in frame.columns or metric_column not in frame.columns:
        return None
    return RoutePlan(
        calls=[
            ToolCall(
                "calculate_group_metrics",
                {
                    "group_column": group_column,
                    "metric_column": metric_column,
                    "aggregation": aggregation if aggregation in {"sum", "mean", "median", "min", "max", "count"} else "sum",
                },
            )
        ],
        assumptions=assumptions,
        direction=direction if direction in {"top", "bottom"} else "top",
    )


def _outlier_targets(question: str, frame: pd.DataFrame, kinds: dict[str, str]):
    mentioned = [
        column for column in mentioned_columns(question, list(frame.columns)) if kinds.get(column) == "numeric"
    ]
    if mentioned:
        return mentioned[:3], []
    scored: list[tuple[float, str]] = []
    for column, kind in kinds.items():
        if kind != "numeric":
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(series) < 4 or float(series.std()) == 0:
            continue
        center = abs(float(series.mean())) or 1.0
        scored.append((float(series.std()) / center, column))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = [column for _score, column in scored[:1]]
    if not chosen:
        chosen = columns_of_kind(frame, "numeric")[:1]
    notes = []
    if chosen:
        others = [column for column in columns_of_kind(frame, "numeric") if column not in chosen]
        note = f"No numeric column was named, so I screened {chosen[0]}."
        if others:
            note += " You can name another numeric column: " + ", ".join(others) + "."
        notes.append(note)
    return chosen, notes


def _group_explicit(lowered: str) -> bool:
    return bool(
        re.search(r"\b(top|bottom|best|worst|underperform\w*|highest|lowest|rank|compare|which)\b", lowered)
    )


def _chart_already_covered(wants_trend: bool, wants_heatmap: bool, lowered: str) -> bool:
    if wants_heatmap:
        return True
    explicit = "histogram" in lowered or "scatter" in lowered or "box" in lowered or "bar chart" in lowered
    if wants_trend and not explicit:
        return True
    return False


def _direction(lowered: str) -> str:
    bottom = re.search(r"\b(bottom|worst|underperform\w*|lowest|least)\b", lowered)
    top = re.search(r"\b(top|best|highest|most)\b", lowered)
    if bottom and (top is None or bottom.start() < top.start()):
        return "bottom"
    return "top"


def _aggregation(lowered: str) -> str:
    if re.search(r"\b(average|avg|mean)\b", lowered):
        return "mean"
    if "median" in lowered:
        return "median"
    if re.search(r"\b(how many|count of|number of)\b", lowered):
        return "count"
    return "sum"


def mentioned_columns(question: str, columns: list[str]) -> list[str]:
    normalized_question = _normalize(question)
    found: list[str] = []
    ordered = sorted((str(column) for column in columns), key=lambda item: len(_normalize(item)), reverse=True)
    for column in ordered:
        token = _normalize(column)
        if token and re.search(rf"\b{re.escape(token)}\b", normalized_question):
            found.append(column)
    return found


def _preferred_group(columns: list[str], kinds: dict[str, str]) -> str | None:
    for key in GROUP_KEYS:
        match = _find_role(columns, kinds, key, {"categorical", "boolean"})
        if match:
            return match
    categorical = [column for column in columns if kinds.get(column) in {"categorical", "boolean"}]
    return categorical[0] if categorical else None


def _synonym_column(
    lowered: str,
    columns: list[str],
    kinds: dict[str, str],
    keys: tuple[str, ...],
    expected: set[str],
) -> str | None:
    normalized = _normalize(lowered)
    for key in keys:
        phrases = SYNONYMS.get(key, (key,))
        if any(re.search(rf"\b{re.escape(_normalize(phrase))}\b", normalized) for phrase in phrases):
            match = _find_role(columns, kinds, key, expected)
            if match:
                return match
            for column in columns:
                if kinds.get(column) in expected and key in _normalize(column):
                    return column
    return None


def _find_role(columns: list[str], kinds: dict[str, str], role: str, expected: set[str]) -> str | None:
    for column in columns:
        token = _normalize(column).replace(" ", "_")
        if token == role or role in token.split("_"):
            if kinds.get(column) in expected:
                return column
    return None


def _need_kind(frame: pd.DataFrame, kind: str, action: str) -> str:
    available = ", ".join(map(str, frame.columns)) or "none"
    pretty = {"numeric": "numeric", "datetime": "date", "categorical": "category"}[kind]
    return (
        f"I can't {action} because this dataset has no {pretty} column I can use. "
        f"Available columns: {available}."
    )


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
