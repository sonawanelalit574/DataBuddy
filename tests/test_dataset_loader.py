"""CSV loading, validation, and the bundled sample."""

from __future__ import annotations

import pandas as pd
import pytest

from analytics.dataset_loader import DatasetError, classify_series, load_csv_bytes
from scripts.build_sample_sales import MARGINS, build


def test_valid_csv_reports_shape_and_kinds():
    bundle = load_csv_bytes(b"order_date,product,revenue,returned\n2024-01-02,A,10,Yes\n", "orders.csv")
    assert len(bundle.frame) == 1
    assert bundle.kinds["revenue"] == "numeric"
    assert bundle.kinds["order_date"] == "datetime"
    assert bundle.kinds["returned"] == "boolean"
    assert bundle.fingerprint


def test_empty_file_is_a_friendly_error():
    with pytest.raises(DatasetError) as caught:
        load_csv_bytes(b"   \n", "empty.csv")
    assert "empty" in caught.value.user_message.lower()
    assert "Traceback" not in caught.value.user_message


def test_header_only_file_loads_with_a_warning():
    bundle = load_csv_bytes(b"product,revenue\n", "header.csv")
    assert len(bundle.frame) == 0
    assert any("no data rows" in warning.lower() for warning in bundle.warnings)


def test_malformed_csv_does_not_leak_a_traceback():
    with pytest.raises(DatasetError) as caught:
        load_csv_bytes(b'product,revenue\n"unterminated,10\n', "broken.csv")
    assert "Traceback" not in caught.value.user_message
    assert "csv" in caught.value.user_message.lower()


def test_binary_payload_is_rejected():
    with pytest.raises(DatasetError) as caught:
        load_csv_bytes(b"product,revenue\nA\x00,10\n", "binary.csv")
    assert "binary" in caught.value.user_message.lower()


def test_wrong_extension_is_rejected():
    with pytest.raises(DatasetError) as caught:
        load_csv_bytes(b"product,revenue\nA,10\n", "notes.xlsx")
    assert ".csv" in caught.value.user_message


def test_file_size_limit():
    payload = b"product,revenue\n" + b"A,10\n" * 20
    with pytest.raises(DatasetError) as caught:
        load_csv_bytes(payload, "big.csv", max_bytes=30)
    assert "limit" in caught.value.user_message.lower()


def test_semicolon_separator():
    bundle = load_csv_bytes("product;revenue\nA;12.5\nB;3\n".encode(), "eu.csv")
    assert list(bundle.frame.columns) == ["product", "revenue"]
    assert bundle.kinds["revenue"] == "numeric"
    assert bundle.frame["revenue"].sum() == pytest.approx(15.5)


def test_latin1_text_is_readable():
    bundle = load_csv_bytes("product,revenue\ncafé,10\n".encode("latin-1"), "cafe.csv")
    assert bundle.frame.iloc[0]["product"] == "café"


def test_duplicate_column_names_are_renamed():
    bundle = load_csv_bytes(b"amount,amount\n1,2\n", "dupes.csv")
    assert len(set(bundle.frame.columns)) == 2
    assert any("duplicate" in warning.lower() for warning in bundle.warnings)


def test_duplicate_rows_and_missing_values_survive_loading():
    bundle = load_csv_bytes(b"product,revenue\nA,10\nA,\nA,10\n", "quality.csv")
    assert int(bundle.frame["revenue"].isna().sum()) == 1
    assert len(bundle.frame) - len(bundle.frame.drop_duplicates()) == 1


def test_mixed_numeric_text_is_called_out():
    bundle = load_csv_bytes(b"revenue\n1\n2\nx\n4\n", "mixed.csv")
    assert bundle.kinds["revenue"] == "numeric"
    assert any("mixes numbers" in warning.lower() for warning in bundle.warnings)
    assert int(bundle.frame["revenue"].isna().sum()) == 1


def test_sample_sales_formulas_and_quality_issues():
    frame = build()
    assert list(frame.columns)[:8] == [
        "order_date",
        "product",
        "category",
        "region",
        "quantity",
        "unit_price",
        "revenue",
        "profit",
    ]
    expected_revenue = (frame["quantity"] * frame["unit_price"]).round(2)
    assert frame["revenue"].tolist() == expected_revenue.tolist()
    known = frame[frame["profit"].notna()]
    rebuilt = [
        round(float(row.revenue) * MARGINS[row.product], 2)
        for row in known.itertuples(index=False)
    ]
    assert known["profit"].round(2).tolist() == rebuilt
    assert frame["profit"].isna().any()
    assert frame["region"].isna().any()
    assert (frame["quantity"] >= 80).any()
    assert len(frame) - len(frame.drop_duplicates()) >= 1


def test_committed_sample_file_loads():
    from config import ROOT

    raw = (ROOT / "data" / "sample_sales.csv").read_bytes()
    bundle = load_csv_bytes(raw, "sample_sales.csv")
    fresh = build()
    assert len(bundle.frame) == len(fresh)
    assert bundle.kinds["order_date"] == "datetime"
    assert bundle.kinds["revenue"] == "numeric"
    assert bundle.kinds["returned"] == "boolean"
    assert bundle.frame["revenue"].sum() == pytest.approx(fresh["revenue"].sum())


def test_classify_boolean_and_year_columns():
    assert classify_series(pd.Series(["Yes", "No", "Yes"])) == "boolean"
    assert classify_series(pd.Series(["2020", "2021", "2022"])) == "numeric"
    assert classify_series(pd.Series(["2024-03-01", "2024-04-01"])) == "datetime"
