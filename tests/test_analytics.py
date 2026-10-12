"""Metrics, quality checks, charts, and exports."""

from __future__ import annotations

import pandas as pd
import pytest

from analytics.data_quality import detect_outliers, duplicate_report, missing_report, quality_table
from analytics.export import analysis_markdown, cleaned_frame, csv_bytes, quality_csv_bytes
from analytics.metrics import correlation_pair, group_metrics, numeric_summary, time_trend
from analytics.visualizations import ChartError, build_chart
from agent.tool_registry import execute


def frame():
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-02", "2024-02-02"]),
            "product": ["A", "A", "B", "B"],
            "region": ["East", "West", "East", None],
            "revenue": [10, 5, 3, 4],
            "quantity": [1, 2, 3, 100],
        }
    )


def test_missing_values_and_duplicate_rows():
    data = frame()
    missing = missing_report(data)
    region = missing.set_index("column").loc["region"]
    assert int(region["missing_count"]) == 1
    assert float(region["missing_percent"]) == pytest.approx(25.0)
    duplicates = data.copy()
    duplicates = pd.concat([duplicates, duplicates.iloc[[0]]], ignore_index=True)
    report = duplicate_report(duplicates)
    assert report["extra_duplicate_rows"] == 1


def test_numeric_summary_and_grouped_metrics():
    summary = numeric_summary(frame())
    assert "revenue" in summary["column"].tolist()
    result = group_metrics(frame(), "product", "revenue", "sum")
    assert result["highest"]["group"] == "A"
    assert result["highest"]["value"] == 15
    assert result["lowest"]["group"] == "B"
    assert result["records"] == 4


def test_grouped_metrics_reject_a_missing_column_at_the_tool_boundary():
    result = execute(
        "calculate_group_metrics",
        {"group_column": "customer", "metric_column": "revenue", "aggregation": "sum"},
        frame(),
    )
    assert not result.ok
    assert "customer" in result.message
    assert "Traceback" not in result.message


def test_date_trend_sums_months_and_skips_empty_periods():
    result = time_trend(frame(), "order_date", "revenue", "monthly")
    assert result["records"] == 4
    assert result["periods"] == 2
    values = result["table"]["value"].tolist()
    assert values == [15, 7]


def test_correlation_and_undefined_correlation():
    data = pd.DataFrame({"quantity": [1, 2, 3, 4, 5], "revenue": [2, 4, 6, 8, 10]})
    result = correlation_pair(data, "quantity", "revenue")
    assert result["correlation"] == pytest.approx(1.0)
    assert result["records"] == 5
    flat = pd.DataFrame({"quantity": [1, 1, 1], "revenue": [2, 3, 4]})
    undefined = correlation_pair(flat, "quantity", "revenue")
    assert undefined["correlation"] is None


def test_outliers_use_tukey_fences():
    data = pd.DataFrame({"quantity": [1, 2, 3, 4, 5, 6, 7, 8, 100]})
    report = detect_outliers(data, "quantity")
    assert report["outlier_count"] == 1
    assert report["method"].startswith("Tukey")
    table = quality_table(data)
    quantity = table.set_index("column").loc["quantity"]
    assert int(quantity["outlier_count"]) == 1


def test_chart_and_missing_column_error():
    chart = build_chart(frame(), "bar", "product", "revenue")
    assert chart["figure"].data
    assert chart["records"] == 4
    with pytest.raises(ChartError) as caught:
        build_chart(frame(), "scatter", "product", "not_a_column")
    assert "not_a_column" in caught.value.user_message
    assert "Traceback" not in caught.value.user_message


def test_heatmap_needs_two_numeric_columns():
    only = pd.DataFrame({"product": ["A", "B"], "revenue": [1, 2]})
    with pytest.raises(ChartError) as caught:
        build_chart(only, "heatmap", "", "")
    assert "numeric" in caught.value.user_message.lower()


def test_cleaned_export_drops_duplicate_rows_and_keeps_blanks():
    data = pd.DataFrame({"product": ["A", "A", "B"], "revenue": [1, 1, None]})
    cleaned, removed = cleaned_frame(data)
    assert removed == 1
    assert len(cleaned) == 2
    assert cleaned["revenue"].isna().any()
    assert b"product" in csv_bytes(cleaned)
    quality = quality_csv_bytes(data)
    assert b"missing_count" in quality


def test_analysis_markdown_contains_the_recorded_formula():
    text = analysis_markdown(
        "orders.csv",
        "abc",
        [
            {
                "question": "Which product leads?",
                "answer": "A leads with 15.",
                "mode_label": "Demo Mode · local calculations only",
                "findings": ["Highest: A at 15."],
                "limitations": ["This is only the uploaded file."],
                "evidences": [
                    {
                        "columns": ["product", "revenue"],
                        "formula": "sum of revenue grouped by product",
                        "records_analyzed": 4,
                        "filters": "No extra row filter.",
                        "method_notes": "",
                    }
                ],
            }
        ],
    )
    assert "A leads with 15." in text
    assert "sum of revenue grouped by product" in text
    assert "abc" in text
