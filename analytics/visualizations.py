"""Plotly charts built only from the columns the caller names."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from analytics.dataset_loader import classify_frame
from analytics.metrics import correlation_matrix

CHART_TYPES = ("bar", "line", "histogram", "scatter", "box", "heatmap")
COLORWAY = ["#3451D1", "#6D5BD0", "#1F8A70", "#C4841D", "#C2516B"]


class ChartError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


def bar_from_table(table: pd.DataFrame, x: str, y: str) -> object:
    plot = table.head(20)
    figure = px.bar(plot, x=x, y=y, color_discrete_sequence=COLORWAY)
    _polish(figure)
    return figure


def line_from_table(table: pd.DataFrame, x: str, y: str) -> object:
    figure = px.line(table, x=x, y=y, markers=True, color_discrete_sequence=COLORWAY)
    _polish(figure)
    return figure


def build_chart(
    frame: pd.DataFrame,
    chart_type: str,
    x_column: str = "",
    y_column: str = "",
) -> dict[str, object]:
    if chart_type not in CHART_TYPES:
        raise ChartError(
            "Choose a chart type of bar, line, histogram, scatter, box, or heatmap."
        )
    if frame.empty:
        raise ChartError("This dataset has no rows to chart.")
    kinds = classify_frame(frame)
    builder = {
        "bar": _bar,
        "line": _line,
        "histogram": _histogram,
        "scatter": _scatter,
        "box": _box,
        "heatmap": _heatmap,
    }[chart_type]
    figure, records, formula, limitations = builder(frame, kinds, x_column, y_column)
    _polish(figure)
    return {
        "figure": figure,
        "records": records,
        "formula": formula,
        "limitations": limitations,
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
    }


def _bar(frame, kinds, x_column, y_column):
    _require_column(x_column, frame, "a category for the bars")
    _require_column(y_column, frame, "a numeric measure for the bar heights")
    _require_kind(y_column, kinds, "numeric")
    work = frame[[x_column, y_column]].copy()
    work[y_column] = pd.to_numeric(work[y_column], errors="coerce")
    work[x_column] = work[x_column].astype("object").where(work[x_column].notna(), "(missing)")
    work = work.dropna(subset=[y_column])
    if work.empty:
        raise ChartError(f"{y_column} has no numeric values to chart.")
    grouped = work.groupby(x_column, dropna=False)[y_column].sum().sort_values(ascending=False)
    limitations = []
    if len(grouped) > 20:
        grouped = grouped.head(20)
        limitations.append("The bar chart shows the 20 largest groups. The table for a ranking can list more.")
    plot_frame = grouped.rename("value").reset_index()
    figure = px.bar(plot_frame, x=x_column, y="value", color_discrete_sequence=COLORWAY)
    figure.update_layout(xaxis_title=str(x_column), yaxis_title=f"Sum of {y_column}")
    return figure, int(len(work)), f"sum of {y_column} by {x_column}", limitations


def _line(frame, kinds, x_column, y_column):
    _require_column(x_column, frame, "an x-axis column")
    _require_column(y_column, frame, "a numeric y-axis column")
    _require_kind(y_column, kinds, "numeric")
    work = frame[[x_column, y_column]].copy()
    work[y_column] = pd.to_numeric(work[y_column], errors="coerce")
    if kinds.get(x_column) == "datetime":
        work[x_column] = pd.to_datetime(work[x_column], errors="coerce")
    work = work.dropna()
    if work.empty:
        raise ChartError("Those columns do not have overlapping values to draw a line.")
    work = work.sort_values(x_column)
    if len(work) > 5000:
        work = work.tail(5000)
        limitations = ["The line chart uses the latest 5,000 complete rows."]
    else:
        limitations = []
    figure = px.line(work, x=x_column, y=y_column, color_discrete_sequence=COLORWAY)
    return figure, int(len(work)), f"line of {y_column} against {x_column}", limitations


def _histogram(frame, kinds, x_column, y_column):
    del y_column
    _require_column(x_column, frame, "a numeric column")
    _require_kind(x_column, kinds, "numeric")
    series = pd.to_numeric(frame[x_column], errors="coerce").dropna()
    if series.empty:
        raise ChartError(f"{x_column} has no numeric values to chart.")
    unique = int(series.nunique())
    bins = min(30, max(5, unique))
    figure = px.histogram(series.to_frame(x_column), x=x_column, nbins=bins, color_discrete_sequence=COLORWAY)
    return figure, int(len(series)), f"histogram of {x_column}", []


def _scatter(frame, kinds, x_column, y_column):
    _require_column(x_column, frame, "a numeric x column")
    _require_column(y_column, frame, "a numeric y column")
    _require_kind(x_column, kinds, "numeric")
    _require_kind(y_column, kinds, "numeric")
    work = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 2:
        raise ChartError("A scatter plot needs at least two complete pairs.")
    limitations = []
    plotted = work
    if len(work) > 5000:
        plotted = work.sample(5000, random_state=0)
        limitations.append(
            "The dots are a 5,000-row sample. Correlation, when requested, still uses every complete pair."
        )
    figure = px.scatter(plotted, x=x_column, y=y_column, color_discrete_sequence=COLORWAY, opacity=0.75)
    return figure, int(len(work)), f"scatter of {y_column} against {x_column}", limitations


def _box(frame, kinds, x_column, y_column):
    _require_column(y_column, frame, "a numeric column")
    _require_kind(y_column, kinds, "numeric")
    columns = [y_column] if not x_column else [x_column, y_column]
    if x_column:
        _require_column(x_column, frame, "a grouping column")
    work = frame[columns].copy()
    work[y_column] = pd.to_numeric(work[y_column], errors="coerce")
    work = work.dropna(subset=[y_column])
    if work.empty:
        raise ChartError(f"{y_column} has no numeric values to chart.")
    if x_column:
        work[x_column] = work[x_column].astype("object").where(work[x_column].notna(), "(missing)")
        figure = px.box(work, x=x_column, y=y_column, color_discrete_sequence=COLORWAY)
        formula = f"box plot of {y_column} by {x_column}"
    else:
        figure = px.box(work, y=y_column, color_discrete_sequence=COLORWAY)
        formula = f"box plot of {y_column}"
    return figure, int(len(work)), formula, []


def _heatmap(frame, kinds, x_column, y_column):
    del x_column, y_column, kinds
    matrix = correlation_matrix(frame)
    if matrix.empty:
        numeric = [name for name, kind in classify_frame(frame).items() if kind == "numeric"]
        if len(numeric) < 2:
            raise ChartError(
                "A correlation heatmap needs at least two numeric columns. "
                f"Numeric columns here: {', '.join(numeric) if numeric else 'none'}."
            )
        raise ChartError("There are not enough complete numeric rows to draw a correlation heatmap.")
    figure = px.imshow(
        matrix,
        text_auto=".2f",
        color_continuous_scale=["#F4F0FF", "#3451D1"],
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    figure.update_layout(xaxis_title="", yaxis_title="")
    records = int(frame.shape[0])
    return (
        figure,
        records,
        "Pearson correlation matrix of the numeric columns",
        ["Correlation is not causation, and blank cells were left out of each pair."],
    )


def _polish(figure) -> None:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,252,1)",
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "color": "#1C2230", "size": 13},
        margin={"l": 48, "r": 24, "t": 36, "b": 48},
        colorway=COLORWAY,
    )


def _require_column(name: str, frame: pd.DataFrame, role: str) -> None:
    if not name or name not in frame.columns:
        available = ", ".join(map(str, frame.columns))
        raise ChartError(
            f"I need {role}. "
            + (f"I can't find a column named '{name}'. " if name else "No column was selected. ")
            + f"Available columns: {available}."
        )


def _require_kind(name: str, kinds: dict[str, str], expected: str) -> None:
    if kinds.get(name) != expected:
        matches = [column for column, kind in kinds.items() if kind == expected]
        listed = ", ".join(matches) if matches else "none"
        raise ChartError(
            f"{name} is {kinds.get(name, 'unknown')}, and this chart needs a {expected} column. "
            f"{expected.capitalize()} columns: {listed}."
        )
