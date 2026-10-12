"""Pandas calculations used by the analyst tools.

Every function reads the dataframe it is given and returns computed values.
None of them call a language model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analytics.data_quality import duplicate_report, missing_report
from analytics.dataset_loader import classify_frame, memory_label

AGGREGATIONS = {
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "min": "min",
    "max": "max",
    "count": "count",
}

FREQUENCIES = {
    "D": "D",
    "W": "W",
    "M": "ME",
    "ME": "ME",
    "Q": "QE",
    "QE": "QE",
    "Y": "YE",
    "YE": "YE",
    "day": "D",
    "daily": "D",
    "week": "W",
    "weekly": "W",
    "month": "ME",
    "monthly": "ME",
    "quarter": "QE",
    "quarterly": "QE",
    "year": "YE",
    "yearly": "YE",
    "annual": "YE",
}

FREQUENCY_LABELS = {
    "D": "day",
    "W": "week",
    "ME": "month",
    "QE": "quarter",
    "YE": "year",
}


def dataset_overview(frame: pd.DataFrame) -> dict[str, Any]:
    kinds = classify_frame(frame)
    grouped: dict[str, list[str]] = {"numeric": [], "categorical": [], "datetime": [], "boolean": []}
    for name, kind in kinds.items():
        grouped.setdefault(kind, []).append(name)
    duplicates = duplicate_report(frame)
    missing_cells = int(frame.isna().sum().sum()) if not frame.empty else 0
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "memory_label": memory_label(frame),
        "duplicate_extra": int(duplicates["extra_duplicate_rows"]),
        "rows_in_duplicate_sets": int(duplicates["rows_in_duplicate_sets"]),
        "missing_cells": missing_cells,
        "kinds": {key: values for key, values in grouped.items()},
        "kind_counts": {key: len(values) for key, values in grouped.items()},
        "missing_by_column": missing_report(frame),
    }


def column_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    kinds = classify_frame(frame)
    kind = kinds.get(column, "categorical")
    series = frame[column]
    summary: dict[str, Any] = {
        "column": column,
        "kind": kind,
        "dtype": str(series.dtype),
        "records": int(len(frame)),
        "missing": int(series.isna().sum()),
        "unique": int(series.nunique(dropna=True)),
    }
    if kind == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        described = numeric.describe()
        summary["stats"] = {str(index): _plain_number(described[index]) for index in described.index}
    elif kind == "datetime":
        dates = pd.to_datetime(series, errors="coerce")
        summary["minimum"] = _plain_date(dates.min()) if dates.notna().any() else None
        summary["maximum"] = _plain_date(dates.max()) if dates.notna().any() else None
    else:
        counts = series.astype("string").fillna("(missing)").value_counts().head(8)
        summary["top_values"] = [
            {"value": str(index), "count": int(value)} for index, value in counts.items()
        ]
    return summary


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    kinds = classify_frame(frame)
    columns = [name for name, kind in kinds.items() if kind == "numeric"]
    if not columns:
        return pd.DataFrame()
    described = frame[columns].apply(pd.to_numeric, errors="coerce").describe().transpose()
    described.index.name = "column"
    return described.reset_index()


def categorical_summary(frame: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    kinds = classify_frame(frame)
    rows = []
    for name, kind in kinds.items():
        if kind not in {"categorical", "boolean"}:
            continue
        counts = frame[name].astype("string").fillna("(missing)").value_counts()
        top = ", ".join(f"{index} ({int(value)})" for index, value in counts.head(limit).items())
        rows.append({"column": name, "kind": kind, "unique_values": int(counts.shape[0]), "most_common": top})
    return pd.DataFrame(rows)


def date_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    kinds = classify_frame(frame)
    rows = []
    for name, kind in kinds.items():
        if kind != "datetime":
            continue
        dates = pd.to_datetime(frame[name], errors="coerce")
        rows.append(
            {
                "column": name,
                "earliest": _plain_date(dates.min()) if dates.notna().any() else None,
                "latest": _plain_date(dates.max()) if dates.notna().any() else None,
                "missing": int(dates.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def group_metrics(
    frame: pd.DataFrame,
    group_column: str,
    metric_column: str,
    aggregation: str,
) -> dict[str, Any]:
    agg = AGGREGATIONS[aggregation]
    work = frame[[group_column, metric_column]].copy()
    work[group_column] = work[group_column].astype("object")
    work.loc[work[group_column].isna(), group_column] = "(missing)"
    if agg == "count":
        grouped = work.groupby(group_column, dropna=False)[metric_column].count()
        records = int(len(work))
    else:
        work[metric_column] = pd.to_numeric(work[metric_column], errors="coerce")
        work = work.dropna(subset=[metric_column])
        records = int(len(work))
        grouped = getattr(work.groupby(group_column, dropna=False)[metric_column], agg)()
    table = grouped.rename("value").reset_index().rename(columns={group_column: "group"})
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    table = table.sort_values(["value", "group"], ascending=[False, True]).reset_index(drop=True)
    highest = None
    lowest = None
    if not table.empty and table["value"].notna().any():
        ordered = table.dropna(subset=["value"])
        high_row = ordered.iloc[0]
        low_row = ordered.iloc[-1]
        highest = {"group": str(high_row["group"]), "value": _plain_number(high_row["value"])}
        lowest = {"group": str(low_row["group"]), "value": _plain_number(low_row["value"])}
    return {
        "group_column": group_column,
        "metric_column": metric_column,
        "aggregation": aggregation,
        "records": records,
        "groups": int(len(table)),
        "highest": highest,
        "lowest": lowest,
        "table": table,
    }


def time_trend(
    frame: pd.DataFrame,
    date_column: str,
    metric_column: str,
    frequency: str,
) -> dict[str, Any]:
    freq = FREQUENCIES[frequency]
    work = frame[[date_column, metric_column]].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work[metric_column] = pd.to_numeric(work[metric_column], errors="coerce")
    work = work.dropna()
    records = int(len(work))
    if records == 0:
        return {
            "date_column": date_column,
            "metric_column": metric_column,
            "frequency": freq,
            "frequency_label": FREQUENCY_LABELS[freq],
            "records": 0,
            "periods": 0,
            "table": pd.DataFrame(columns=["period", "value", "rows"]),
            "peak": None,
            "trough": None,
            "start": None,
            "end": None,
        }
    indexed = work.set_index(date_column).sort_index()
    grouped = indexed.resample(freq).agg(value=(metric_column, "sum"), rows=(metric_column, "count"))
    grouped = grouped[grouped["rows"] > 0].reset_index()
    grouped = grouped.rename(columns={date_column: "period"})
    grouped["period_label"] = grouped["period"].map(_plain_date)
    peak = trough = None
    if not grouped.empty:
        peak_row = grouped.loc[grouped["value"].idxmax()]
        trough_row = grouped.loc[grouped["value"].idxmin()]
        peak = {"period": str(peak_row["period_label"]), "value": _plain_number(peak_row["value"])}
        trough = {"period": str(trough_row["period_label"]), "value": _plain_number(trough_row["value"])}
    return {
        "date_column": date_column,
        "metric_column": metric_column,
        "frequency": freq,
        "frequency_label": FREQUENCY_LABELS[freq],
        "records": records,
        "periods": int(len(grouped)),
        "table": grouped[["period_label", "value", "rows"]].rename(columns={"period_label": "period"}),
        "peak": peak,
        "trough": trough,
        "start": _plain_date(work[date_column].min()),
        "end": _plain_date(work[date_column].max()),
    }


def correlation_pair(frame: pd.DataFrame, column_a: str, column_b: str) -> dict[str, Any]:
    work = frame[[column_a, column_b]].apply(pd.to_numeric, errors="coerce").dropna()
    records = int(len(work))
    if records < 3:
        return {
            "column_a": column_a,
            "column_b": column_b,
            "records": records,
            "correlation": None,
            "reason": "At least 3 complete pairs are required.",
        }
    if work[column_a].nunique() < 2 or work[column_b].nunique() < 2:
        return {
            "column_a": column_a,
            "column_b": column_b,
            "records": records,
            "correlation": None,
            "reason": "One of the columns does not vary, so correlation is undefined.",
        }
    value = float(work[column_a].corr(work[column_b], method="pearson"))
    if np.isnan(value):
        return {
            "column_a": column_a,
            "column_b": column_b,
            "records": records,
            "correlation": None,
            "reason": "Pearson correlation was undefined for these values.",
        }
    return {
        "column_a": column_a,
        "column_b": column_b,
        "records": records,
        "correlation": value,
        "reason": "",
    }


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    kinds = classify_frame(frame)
    columns = [name for name, kind in kinds.items() if kind == "numeric"]
    if len(columns) < 2:
        return pd.DataFrame()
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    if numeric.dropna(how="all").shape[0] < 3:
        return pd.DataFrame()
    return numeric.corr(method="pearson")


def default_metric_name(frame: pd.DataFrame) -> str | None:
    kinds = classify_frame(frame)
    preferred = ("revenue", "sales", "profit", "amount", "quantity", "unit_price", "price")
    numeric = [name for name, kind in kinds.items() if kind == "numeric"]
    for key in preferred:
        for name in numeric:
            token = str(name).lower().replace(" ", "_")
            if token == key or key in token:
                return name
    return numeric[0] if numeric else None


def columns_of_kind(frame: pd.DataFrame, kind: str) -> list[str]:
    return [name for name, value in classify_frame(frame).items() if value == kind]


def strongest_pair(frame: pd.DataFrame) -> tuple[str, str] | None:
    matrix = correlation_matrix(frame)
    if matrix.empty:
        return None
    best: tuple[str, str] | None = None
    best_score = -1.0
    for left in matrix.columns:
        for right in matrix.columns:
            if left >= right:
                continue
            value = matrix.loc[left, right]
            if pd.isna(value):
                continue
            score = abs(float(value))
            if score > best_score:
                best_score = score
                best = (str(left), str(right))
    return best


def _plain_number(value: object) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if number.is_integer() and abs(number) < 1e15:
        return int(number)
    return round(number, 6)


def _plain_date(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    if stamp.hour == 0 and stamp.minute == 0 and stamp.second == 0:
        return stamp.strftime("%Y-%m-%d")
    return stamp.strftime("%Y-%m-%d %H:%M")
