"""Data-quality checks.

Potential outliers use Tukey's fences: a value is flagged when it is below
Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR. This is a screening rule, not proof
that a value is wrong.
"""

from __future__ import annotations

import pandas as pd

from analytics.dataset_loader import classify_frame

IQR_METHOD = (
    "Tukey fences: values below Q1 - 1.5 × IQR or above Q3 + 1.5 × IQR "
    "are marked as potential outliers."
)


def missing_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows = max(len(frame), 1)
    records = []
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        records.append(
            {
                "column": str(column),
                "missing_count": missing,
                "missing_percent": round(missing / rows * 100, 2) if len(frame) else 0.0,
            }
        )
    report = pd.DataFrame(records)
    if report.empty:
        return report
    return report.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def duplicate_report(frame: pd.DataFrame) -> dict[str, int | pd.DataFrame]:
    if frame.empty:
        return {
            "extra_duplicate_rows": 0,
            "rows_in_duplicate_sets": 0,
            "preview": frame.copy(),
        }
    extra = int(len(frame) - len(frame.drop_duplicates()))
    in_sets = int(frame.duplicated(keep=False).sum())
    preview = frame.loc[frame.duplicated(keep=False)].head(20).copy()
    return {
        "extra_duplicate_rows": extra,
        "rows_in_duplicate_sets": in_sets,
        "preview": preview,
    }


def detect_outliers(frame: pd.DataFrame, column: str) -> dict[str, object]:
    if column not in frame.columns:
        raise KeyError(column)
    series = pd.to_numeric(frame[column], errors="coerce")
    present = series.dropna()
    result: dict[str, object] = {
        "column": column,
        "records": int(present.shape[0]),
        "method": IQR_METHOD,
        "outlier_count": 0,
        "lower": None,
        "upper": None,
        "q1": None,
        "q3": None,
        "preview": pd.DataFrame(),
        "note": "",
    }
    if present.shape[0] < 4:
        result["note"] = (
            f"{column} has {present.shape[0]} numeric value(s). "
            "At least 4 are needed before Tukey fences are meaningful."
        )
        return result
    q1 = float(present.quantile(0.25))
    q3 = float(present.quantile(0.75))
    iqr = q3 - q1
    result["q1"] = q1
    result["q3"] = q3
    if iqr == 0:
        result["note"] = (
            f"{column} has no spread between the 25th and 75th percentiles, "
            "so this method does not flag outliers."
        )
        result["lower"] = q1
        result["upper"] = q3
        return result
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = series.notna() & ((series < lower) | (series > upper))
    preview = frame.loc[mask, [column]].copy()
    preview.insert(0, "row_number", preview.index + 1)
    flagged = [float(value) for value in series[mask].head(5).tolist()]
    result.update(
        {
            "outlier_count": int(mask.sum()),
            "lower": lower,
            "upper": upper,
            "preview": preview.head(20),
            "flagged_values": flagged,
        }
    )
    return result


def quality_table(frame: pd.DataFrame) -> pd.DataFrame:
    kinds = classify_frame(frame)
    missing = missing_report(frame).set_index("column")
    rows = []
    for column in frame.columns:
        name = str(column)
        kind = kinds[name]
        outlier_count: int | None = None
        notes: list[str] = []
        if kind == "numeric":
            report = detect_outliers(frame, name)
            outlier_count = int(report["outlier_count"])
            if report["note"]:
                notes.append(str(report["note"]))
        missing_count = int(missing.loc[name, "missing_count"]) if name in missing.index else 0
        missing_percent = float(missing.loc[name, "missing_percent"]) if name in missing.index else 0.0
        if missing_percent >= 20:
            notes.append("At least 20% of the values are missing.")
        unique = int(frame[column].nunique(dropna=True))
        if len(frame) and unique <= 1 and missing_count < len(frame):
            notes.append("This column has only one distinct value.")
        rows.append(
            {
                "column": name,
                "kind": kind,
                "dtype": str(frame[column].dtype),
                "missing_count": missing_count,
                "missing_percent": round(missing_percent, 2),
                "unique_values": unique,
                "outlier_count": outlier_count if outlier_count is not None else "",
                "notes": " ".join(notes),
            }
        )
    return pd.DataFrame(rows)
