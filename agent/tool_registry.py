"""The only operations the analyst is allowed to run."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from agent.guard import CalculationTimeout, run_with_timeout
from analytics.data_quality import IQR_METHOD, detect_outliers, duplicate_report, missing_report
from analytics.dataset_loader import classify_frame
from analytics.metrics import (
    AGGREGATIONS,
    FREQUENCIES,
    column_summary,
    correlation_pair,
    dataset_overview,
    default_metric_name,
    group_metrics,
    time_trend,
)
from analytics.visualizations import ChartError, bar_from_table, build_chart, line_from_table
from config import get_settings

logger = logging.getLogger("databuddy.tools")


@dataclass
class Evidence:
    columns: list[str]
    formula: str
    records_analyzed: int
    filters: str
    method_notes: str = ""


@dataclass
class ToolResult:
    ok: bool
    name: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    table_title: str = ""
    figure: Any = None
    figure_title: str = ""
    evidences: list[Evidence] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    context_update: dict[str, str] = field(default_factory=dict)

    @classmethod
    def fail(cls, name: str, message: str) -> "ToolResult":
        return cls(ok=False, name=name, message=message)


@dataclass(frozen=True)
class Param:
    description: str
    required: bool = True
    enum: tuple[str, ...] | None = None
    allow_blank: bool = False
    column: bool = False
    kind: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, Param]
    required: tuple[str, ...]
    handler: Callable[[pd.DataFrame, dict[str, str]], ToolResult]


def execute(name: str, arguments: Any, frame: pd.DataFrame) -> ToolResult:
    spec = TOOLS.get(name)
    if spec is None:
        return ToolResult.fail(
            str(name),
            f"I don't have an operation named '{name}'. I only run a fixed list of data checks.",
        )
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return ToolResult.fail(name, "The tool arguments were not a JSON object, so I did not run anything.")
    cleaned, error = _validate(spec, arguments, frame)
    if error:
        return ToolResult.fail(name, error)
    settings = get_settings()
    try:
        return run_with_timeout(lambda: spec.handler(frame, cleaned), settings.tool_timeout_seconds)
    except CalculationTimeout as exc:
        logger.error("tool timed out name=%s", name)
        return ToolResult.fail(name, exc.user_message)
    except Exception:
        logger.error("tool failed name=%s", name)
        return ToolResult.fail(
            name,
            "That calculation hit an unexpected problem and was stopped. Try another column or a smaller question.",
        )


def bedrock_tool_config() -> dict[str, Any]:
    tools = []
    for spec in TOOLS.values():
        properties: dict[str, Any] = {}
        for param_name, param in spec.params.items():
            schema: dict[str, Any] = {"type": "string", "description": param.description}
            if param.enum:
                schema["enum"] = list(param.enum)
            properties[param_name] = schema
        tools.append(
            {
                "toolSpec": {
                    "name": spec.name,
                    "description": spec.description,
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": properties,
                            "required": list(spec.required),
                        }
                    },
                }
            }
        )
    return {"tools": tools}


def _validate(spec: ToolSpec, arguments: dict[str, Any], frame: pd.DataFrame) -> tuple[dict[str, str], str | None]:
    unknown = sorted(set(arguments) - set(spec.params))
    if unknown:
        return {}, (
            "I won't run that request because it included unsupported arguments: "
            + ", ".join(unknown)
            + "."
        )
    cleaned: dict[str, str] = {}
    kinds = classify_frame(frame) if frame is not None else {}
    for param_name, param in spec.params.items():
        if param_name not in arguments:
            if param.required:
                return {}, f"The {spec.name} operation needs '{param_name}'."
            cleaned[param_name] = ""
            continue
        value = arguments[param_name]
        if not isinstance(value, str):
            return {}, f"'{param_name}' must be text, so I did not run {spec.name}."
        value = value.strip()
        if param.enum:
            lookup = {item.lower(): item for item in param.enum}
            if value.lower() not in lookup:
                allowed = ", ".join(param.enum)
                return {}, f"'{param_name}' must be one of: {allowed}."
            value = lookup[value.lower()]
        if param.column and not (param.allow_blank and value == ""):
            if value not in frame.columns:
                available = ", ".join(map(str, frame.columns)) or "none"
                return {}, (
                    f"I can't find a column named '{value}'. Available columns: {available}."
                )
            actual = kinds.get(value)
            if param.kind and actual != param.kind:
                matches = [column for column, kind in kinds.items() if kind == param.kind]
                listed = ", ".join(matches) if matches else "none"
                return {}, (
                    f"{value} is {actual or 'unknown'}, and this check needs a {param.kind} column. "
                    f"{param.kind.capitalize()} columns: {listed}."
                )
        cleaned[param_name] = value
    return cleaned, None


def _overview(frame: pd.DataFrame, arguments: dict[str, str]) -> ToolResult:
    del arguments
    if frame.empty:
        return ToolResult.fail("dataset_overview", "This dataset has no rows to summarize.")
    overview = dataset_overview(frame)
    chart_note = ""
    figure = None
    evidences = [
        Evidence(
            columns=list(map(str, frame.columns)),
            formula="row count, column kinds, missing cells, and exact duplicate rows",
            records_analyzed=int(len(frame)),
            filters="No rows were filtered out of the counts.",
            method_notes="Kinds are inferred from the values in this session.",
        )
    ]
    metric = default_metric_name(frame)
    kinds = classify_frame(frame)
    categorical = [name for name, kind in kinds.items() if kind == "categorical"]
    dates = [name for name, kind in kinds.items() if kind == "datetime"]
    try:
        if dates and metric:
            chart = build_chart(frame, "line", dates[0], metric)
            figure = chart["figure"]
            chart_note = f"The line chart plots {metric} against {dates[0]}."
            evidences.append(
                Evidence(
                    columns=[dates[0], metric],
                    formula=str(chart["formula"]),
                    records_analyzed=int(chart["records"]),
                    filters="Rows missing the date or the measure are omitted from the chart.",
                )
            )
        elif categorical and metric:
            chart = build_chart(frame, "bar", categorical[0], metric)
            figure = chart["figure"]
            chart_note = f"The bar chart is the sum of {metric} by {categorical[0]}."
            evidences.append(
                Evidence(
                    columns=[categorical[0], metric],
                    formula=str(chart["formula"]),
                    records_analyzed=int(chart["records"]),
                    filters="Rows missing the measure are omitted from the chart.",
                )
            )
    except ChartError:
        figure = None
    overview["chart_note"] = chart_note
    missing = overview.pop("missing_by_column")
    table = missing[missing["missing_count"] > 0] if not missing.empty else missing
    return ToolResult(
        ok=True,
        name="dataset_overview",
        data=overview,
        table=table if table is not None and not table.empty else None,
        table_title="Columns with missing values",
        figure=figure,
        figure_title="Overview chart",
        evidences=evidences,
        limitations=["Counts describe this file only. They are not a sample of a larger population unless you intended that."],
    )


def _column(frame: pd.DataFrame, arguments: dict[str, str]) -> ToolResult:
    column = arguments["column_name"]
    summary = column_summary(frame, column)
    table = None
    if "top_values" in summary:
        table = pd.DataFrame(summary["top_values"])
    elif "stats" in summary:
        table = pd.DataFrame([{"statistic": key, "value": value} for key, value in summary["stats"].items()])
    return ToolResult(
        ok=True,
        name="get_column_summary",
        data=summary,
        table=table,
        table_title=f"Summary of {column}",
        evidences=[
            Evidence(
                columns=[column],
                formula=f"column profile for {column}",
                records_analyzed=int(len(frame)),
                filters="Missing values are counted and excluded from numeric statistics.",
            )
        ],
        context_update={"column_name": column},
        limitations=[],
    )


def _group(frame: pd.DataFrame, arguments: dict[str, str]) -> ToolResult:
    result = group_metrics(
        frame,
        arguments["group_column"],
        arguments["metric_column"],
        arguments["aggregation"],
    )
    table = result["table"]
    figure = None
    limitations = [
        "A blank group is labeled (missing) so those rows stay visible.",
        "This ranking describes the uploaded rows. It is not a forecast.",
    ]
    if arguments["aggregation"] != "count":
        limitations.append(
            f"Rows with a blank {arguments['metric_column']} were left out of the {arguments['aggregation']}."
        )
    if result["groups"] > 20 and isinstance(table, pd.DataFrame):
        display = pd.concat([table.head(15), table.tail(5)]).drop_duplicates()
        limitations.append(
            f"All {result['groups']} groups were calculated. The table shows the highest 15 and lowest 5."
        )
    else:
        display = table
    if isinstance(display, pd.DataFrame) and not display.empty and result["records"]:
        figure = bar_from_table(display, "group", "value")
    return ToolResult(
        ok=True,
        name="calculate_group_metrics",
        data={key: value for key, value in result.items() if key != "table"},
        table=display,
        table_title=f"{arguments['aggregation']} of {arguments['metric_column']} by {arguments['group_column']}",
        figure=figure,
        figure_title="Group comparison",
        evidences=[
            Evidence(
                columns=[arguments["group_column"], arguments["metric_column"]],
                formula=f"{arguments['aggregation']} of {arguments['metric_column']} grouped by {arguments['group_column']}",
                records_analyzed=int(result["records"]),
                filters="Groups are calculated from the current dataset with no extra row filter.",
            )
        ],
        limitations=limitations,
        context_update={
            "group_column": arguments["group_column"],
            "metric_column": arguments["metric_column"],
            "aggregation": arguments["aggregation"],
        },
    )


def _trend(frame: pd.DataFrame, arguments: dict[str, str]) -> ToolResult:
    result = time_trend(
        frame,
        arguments["date_column"],
        arguments["metric_column"],
        arguments["frequency"],
    )
    figure = None
    trend_table = result["table"]
    if isinstance(trend_table, pd.DataFrame) and not trend_table.empty:
        figure = line_from_table(trend_table, "period", "value")
    return ToolResult(
        ok=True,
        name="calculate_time_trend",
        data={key: value for key, value in result.items() if key != "table"},
        table=result["table"],
        table_title=f"{result['frequency_label'].title()} sum of {arguments['metric_column']}",
        figure=figure,
        figure_title="Time trend",
        evidences=[
            Evidence(
                columns=[arguments["date_column"], arguments["metric_column"]],
                formula=(
                    f"sum of {arguments['metric_column']} by {result['frequency_label']} "
                    f"using {arguments['date_column']}"
                ),
                records_analyzed=int(result["records"]),
                filters="Periods with no rows are omitted. They are not filled in as zero.",
                method_notes="The table and the line chart both use the summed period, not a forecast.",
            )
        ],
        limitations=[
            "This is the history in the file, not a forecast.",
            "Rows with a blank date or measure were excluded.",
        ],
        context_update={
            "date_column": arguments["date_column"],
            "metric_column": arguments["metric_column"],
            "frequency": result["frequency"],
        },
    )


def _missing(frame: pd.DataFrame, arguments: dict[str, str]) -> ToolResult:
    del arguments
    report = missing_report(frame)
    missing_cells = int(report["missing_count"].sum()) if not report.empty else 0
    present = report[report["missing_count"] > 0] if not report.empty else report
    return ToolResult(
        ok=True,
        name="find_missing_values",
        data={"missing_cells": missing_cells, "columns_with_gaps": int(len(present))},
        table=present if not present.empty else None,
        table_title="Missing values",
        evidences=[
            Evidence(
                columns=list(map(str, frame.columns)),
                formula="count and percent of blank cells by column",
                records_analyzed=int(len(frame)),
                filters="A cell counts as missing when pandas reads it as null.",
            )
        ],
        limitations=["A blank may be a true unknown or a value that failed type parsing."],
    )


def _duplicates(frame: pd.DataFrame, arguments: dict[str, str]) -> ToolResult:
    del arguments
    report = duplicate_report(frame)
    return ToolResult(
        ok=True,
        name="find_duplicates",
        data={
            "extra_duplicate_rows": report["extra_duplicate_rows"],
            "rows_in_duplicate_sets": report["rows_in_duplicate_sets"],
        },
        table=report["preview"] if isinstance(report["preview"], pd.DataFrame) and not report["preview"].empty else None,
        table_title="Rows that participate in an exact duplicate",
        evidences=[
            Evidence(
                columns=list(map(str, frame.columns)),
                formula="exact full-row duplicates",
                records_analyzed=int(len(frame)),
                filters="Two rows match only when every column is equal.",
                method_notes="extra_duplicate_rows is how many rows drop_duplicates would remove.",
            )
        ],
        limitations=["Near-duplicates that differ by a space or a cent are not counted."],
    )


def _outliers(frame: pd.DataFrame, arguments: dict[str, str]) -> ToolResult:
    column = arguments["column_name"]
    report = detect_outliers(frame, column)
    figure = None
    try:
        chart = build_chart(frame, "box", "", column)
        figure = chart["figure"]
    except ChartError:
        figure = None
    limitations = [IQR_METHOD, "A flagged value can be a real extreme event. The method does not decide that it is an error."]
    records = int(report["records"] or 0)
    count = int(report["outlier_count"] or 0)
    if records and count / records >= 0.1:
        limitations.append(
            f"More than 10% of {column} was flagged. On a skewed column, Tukey fences often mark ordinary high or low values, not mistakes."
        )
    lower = report["lower"]
    if lower is not None and float(lower) < 0:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not numeric.empty and float(numeric.min()) >= 0:
            limitations.append(
                "The lower fence is below zero even though this column has no negative values. That happens when the middle of the data is narrow."
            )
    if report["note"]:
        limitations.append(str(report["note"]))
    return ToolResult(
        ok=True,
        name="detect_outliers",
        data={key: value for key, value in report.items() if key != "preview"},
        table=report["preview"] if isinstance(report["preview"], pd.DataFrame) and not report["preview"].empty else None,
        table_title=f"Potential outliers in {column}",
        figure=figure,
        figure_title=f"Spread of {column}",
        evidences=[
            Evidence(
                columns=[column],
                formula="Tukey fence outlier screen (Q1 - 1.5 IQR, Q3 + 1.5 IQR)",
                records_analyzed=int(report["records"]),
                filters="Non-numeric values in the column were excluded.",
                method_notes=IQR_METHOD,
            )
        ],
        limitations=limitations,
        context_update={"column_name": column},
    )


def _correlation(frame: pd.DataFrame, arguments: dict[str, str]) -> ToolResult:
    result = correlation_pair(frame, arguments["column_a"], arguments["column_b"])
    if result["correlation"] is None:
        return ToolResult.fail(
            "calculate_correlation",
            f"I could not compute a correlation between {arguments['column_a']} and "
            f"{arguments['column_b']}. {result['reason']} Complete pairs: {result['records']}.",
        )
    figure = None
    try:
        chart = build_chart(frame, "scatter", arguments["column_a"], arguments["column_b"])
        figure = chart["figure"]
    except ChartError:
        figure = None
    return ToolResult(
        ok=True,
        name="calculate_correlation",
        data=result,
        figure=figure,
        figure_title=f"{arguments['column_a']} and {arguments['column_b']}",
        evidences=[
            Evidence(
                columns=[arguments["column_a"], arguments["column_b"]],
                formula="Pearson correlation on complete pairs",
                records_analyzed=int(result["records"]),
                filters="Rows missing either column were excluded.",
                method_notes="Pearson measures a straight-line association. It does not show cause.",
            )
        ],
        limitations=["A strong correlation is not a reason to treat one column as the cause of the other."],
        context_update={"column_a": arguments["column_a"], "column_b": arguments["column_b"]},
    )


def _chart(frame: pd.DataFrame, arguments: dict[str, str]) -> ToolResult:
    try:
        chart = build_chart(
            frame,
            arguments["chart_type"],
            arguments.get("x_column", ""),
            arguments.get("y_column", ""),
        )
    except ChartError as exc:
        return ToolResult.fail("create_visualization", exc.user_message)
    columns = [name for name in (arguments.get("x_column", ""), arguments.get("y_column", "")) if name]
    return ToolResult(
        ok=True,
        name="create_visualization",
        data={
            "chart_type": chart["chart_type"],
            "records": chart["records"],
            "formula": chart["formula"],
        },
        figure=chart["figure"],
        figure_title=str(chart["formula"]),
        evidences=[
            Evidence(
                columns=columns or list(map(str, frame.columns)),
                formula=str(chart["formula"]),
                records_analyzed=int(chart["records"]),
                filters="See the chart notes for sampling or top-group limits.",
            )
        ],
        limitations=list(chart["limitations"]),
    )


def _suggest(frame: pd.DataFrame, arguments: dict[str, str]) -> ToolResult:
    del arguments
    from agent.intent import suggested_questions

    questions = suggested_questions(frame)
    return ToolResult(
        ok=True,
        name="suggest_next_questions",
        data={"questions": questions},
        evidences=[
            Evidence(
                columns=list(map(str, frame.columns)),
                formula="no aggregation — suggestions are based on column kinds",
                records_analyzed=int(len(frame)),
                filters="No metric was calculated for this reply.",
                method_notes="Suggestions are questions the local router can run on these columns.",
            )
        ],
        limitations=["These are next questions, not findings."],
    )


TOOLS: dict[str, ToolSpec] = {
    "dataset_overview": ToolSpec(
        name="dataset_overview",
        description="Summarize rows, columns, kinds, missing values, and duplicate rows.",
        params={},
        required=(),
        handler=_overview,
    ),
    "get_column_summary": ToolSpec(
        name="get_column_summary",
        description="Profile one column: kind, missing values, and either statistics or common values.",
        params={"column_name": Param("Exact column name.", column=True)},
        required=("column_name",),
        handler=_column,
    ),
    "calculate_group_metrics": ToolSpec(
        name="calculate_group_metrics",
        description="Aggregate a measure by a category. Aggregation is sum, mean, median, min, max, or count.",
        params={
            "group_column": Param("Column to group by.", column=True),
            "metric_column": Param("Column to aggregate.", column=True),
            "aggregation": Param("Aggregation name.", enum=tuple(AGGREGATIONS)),
        },
        required=("group_column", "metric_column", "aggregation"),
        handler=_group,
    ),
    "calculate_time_trend": ToolSpec(
        name="calculate_time_trend",
        description="Sum a numeric column by day, week, month, quarter, or year.",
        params={
            "date_column": Param("Date column.", column=True, kind="datetime"),
            "metric_column": Param("Numeric column to sum.", column=True, kind="numeric"),
            "frequency": Param("D, W, M, Q, or Y.", enum=tuple(FREQUENCIES)),
        },
        required=("date_column", "metric_column", "frequency"),
        handler=_trend,
    ),
    "find_missing_values": ToolSpec(
        name="find_missing_values",
        description="Count blank cells by column.",
        params={},
        required=(),
        handler=_missing,
    ),
    "find_duplicates": ToolSpec(
        name="find_duplicates",
        description="Count exact duplicate rows.",
        params={},
        required=(),
        handler=_duplicates,
    ),
    "detect_outliers": ToolSpec(
        name="detect_outliers",
        description="Flag potential outliers in one numeric column with Tukey fences.",
        params={"column_name": Param("Numeric column.", column=True, kind="numeric")},
        required=("column_name",),
        handler=_outliers,
    ),
    "calculate_correlation": ToolSpec(
        name="calculate_correlation",
        description="Pearson correlation between two numeric columns.",
        params={
            "column_a": Param("First numeric column.", column=True, kind="numeric"),
            "column_b": Param("Second numeric column.", column=True, kind="numeric"),
        },
        required=("column_a", "column_b"),
        handler=_correlation,
    ),
    "create_visualization": ToolSpec(
        name="create_visualization",
        description="Draw a bar, line, histogram, scatter, box, or heatmap from named columns.",
        params={
            "chart_type": Param("Chart type.", enum=("bar", "line", "histogram", "scatter", "box", "heatmap")),
            "x_column": Param("X column. Blank for heatmap or a single box plot.", column=True, allow_blank=True, required=False),
            "y_column": Param("Y column. Blank for histogram or heatmap.", column=True, allow_blank=True, required=False),
        },
        required=("chart_type",),
        handler=_chart,
    ),
    "suggest_next_questions": ToolSpec(
        name="suggest_next_questions",
        description="Suggest follow-up questions that the available columns can answer.",
        params={},
        required=(),
        handler=_suggest,
    ),
}
