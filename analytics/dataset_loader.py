"""Safe CSV loading and column classification.

The loader accepts text CSV files only. It does not evaluate formulas,
open archives, or send bytes anywhere.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
from dataclasses import dataclass, field

import pandas as pd

from config import get_settings

logger = logging.getLogger("databuddy.loader")

BOOL_TOKENS = {"true", "false", "yes", "no", "y", "n", "0", "1", "t", "f"}


class DatasetError(Exception):
    """A file problem the user can act on. Never wrap a raw traceback."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    source_name: str
    fingerprint: str
    warnings: list[str] = field(default_factory=list)
    kinds: dict[str, str] = field(default_factory=dict)


def load_csv_bytes(data: bytes, filename: str, max_bytes: int | None = None) -> DatasetBundle:
    settings = get_settings()
    limit = settings.max_upload_bytes if max_bytes is None else max_bytes
    _reject_name(filename)
    _reject_payload(data, limit)

    text = _decode(data)
    frame, separator = _parse_table(text)
    frame = _apply_header_names(frame, text, separator)
    if frame.shape[1] > settings.max_columns:
        raise DatasetError(
            f"This file has {frame.shape[1]} columns. DataBuddy stops above "
            f"{settings.max_columns} columns so the session stays responsive."
        )
    if len(frame) > settings.max_rows:
        raise DatasetError(
            f"This file has {len(frame):,} rows. DataBuddy stops above "
            f"{settings.max_rows:,} rows. Filter the export and try a smaller CSV."
        )

    prepared, warnings, kinds = prepare_frame(frame)
    if prepared.empty:
        warnings.append(
            "The file has column names but no data rows, so there is nothing to calculate yet."
        )
    fingerprint = fingerprint_frame(prepared, filename)
    logger.info(
        "loaded csv rows=%s columns=%s warnings=%s",
        len(prepared),
        prepared.shape[1],
        len(warnings),
    )
    return DatasetBundle(
        frame=prepared,
        source_name=filename,
        fingerprint=fingerprint,
        warnings=warnings,
        kinds=kinds,
    )


def prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    warnings: list[str] = []
    working = frame.copy()
    renamed, duplicates = _uniquify_columns(list(working.columns))
    working.columns = renamed
    if duplicates:
        unique_names = sorted(set(duplicates))
        warnings.append(
            "Duplicate column names were renamed so each column stays addressable: "
            + ", ".join(unique_names)
            + "."
        )

    kinds: dict[str, str] = {}
    for column in list(working.columns):
        kind, converted, notes = _prepare_series(working[column])
        working[column] = converted
        kinds[column] = kind
        warnings.extend(notes)
    return working, warnings, kinds


def classify_frame(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column): classify_series(frame[column]) for column in frame.columns}


def classify_series(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    values = series.dropna()
    if values.empty:
        return "categorical"
    if _is_boolean_like(values):
        return "boolean"
    sample = values.astype(str).str.strip().head(200)
    if _is_year_like(sample):
        return "numeric"
    if sample.str.fullmatch(r"\d+").mean() > 0.9:
        numeric = pd.to_numeric(sample, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            return "numeric"
        return "categorical"
    if _date_rate(sample) >= 0.8:
        return "datetime"
    numeric = pd.to_numeric(values.astype(str).str.replace(",", "", regex=False), errors="coerce")
    # 60% is enough to treat a column as numeric and blank the rest. A single
    # stray word in a short column should not hide the numbers.
    if numeric.notna().mean() >= 0.6:
        return "numeric"
    return "categorical"


def fingerprint_frame(frame: pd.DataFrame, source_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(source_name.encode("utf-8", errors="replace"))
    digest.update(str(frame.shape).encode())
    digest.update("|".join(map(str, frame.columns)).encode("utf-8", errors="replace"))
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    digest.update(hashed)
    return digest.hexdigest()[:20]


def memory_label(frame: pd.DataFrame) -> str:
    size = int(frame.memory_usage(deep=True).sum())
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _reject_name(filename: str) -> None:
    name = (filename or "").strip()
    if not name.lower().endswith(".csv"):
        raise DatasetError(
            "Please upload a file that ends in .csv. DataBuddy reads comma-separated "
            "text, not Excel workbooks, images, or scripts."
        )


def _reject_payload(data: bytes, limit: int) -> None:
    if data is None or len(data) == 0 or not data.strip():
        raise DatasetError(
            "This file is empty. Choose a CSV with a header row and at least one data "
            "row, or load the bundled sample dataset."
        )
    if b"\x00" in data:
        raise DatasetError(
            "This file looks binary rather than a text CSV. Export the sheet as CSV text and try again."
        )
    if len(data) > limit:
        size_mb = len(data) / (1024 * 1024)
        cap_mb = limit / (1024 * 1024)
        raise DatasetError(
            f"This file is {size_mb:.1f} MB, which is above the {cap_mb:.0f} MB limit. "
            "Export a smaller extract and try again."
        )


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DatasetError(
        "I could not read the file's text encoding. Save the CSV as UTF-8 and try again."
    )


def _parse_table(text: str) -> tuple[pd.DataFrame, str]:
    seps: list[str] = []
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
        seps.append(dialect.delimiter)
    except csv.Error:
        pass
    for fallback in (",", ";", "\t", "|"):
        if fallback not in seps:
            seps.append(fallback)

    best: pd.DataFrame | None = None
    best_sep = ","
    last_error: Exception | None = None
    for sep in seps:
        try:
            frame = _read_sep(text, sep)
        except DatasetError:
            raise
        except Exception as exc:  # parser differences across engines
            last_error = exc
            logger.info("csv parse attempt failed error_type=%s", type(exc).__name__)
            continue
        if best is None or (frame.shape[1], len(frame)) > (best.shape[1], len(best)):
            best = frame
            best_sep = sep
    if best is None:
        logger.info(
            "csv rejected error_type=%s",
            type(last_error).__name__ if last_error else "none",
        )
        raise DatasetError(
            "I could not read this file as a CSV. Check that values are separated by "
            "commas, semicolons, or tabs, and that quotation marks are closed. "
            "Excel workbooks need to be saved as CSV first."
        )
    if best.shape[1] == 0:
        raise DatasetError("I could not find any columns in this CSV.")
    return best, best_sep


def _read_sep(text: str, sep: str) -> pd.DataFrame:
    kwargs = {
        "sep": sep,
        "skipinitialspace": True,
        "low_memory": False,
    }
    try:
        return pd.read_csv(io.StringIO(text), **kwargs)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(
            "This CSV has no table to read. Add a header row and at least one data row."
        ) from exc
    except pd.errors.ParserError as exc:
        try:
            return pd.read_csv(io.StringIO(text), engine="python", **kwargs)
        except pd.errors.ParserError:
            raise DatasetError(
                "I could not read this file as a CSV. Check that values are separated by "
                "commas, semicolons, or tabs, and that quotation marks are closed."
            ) from exc


def _apply_header_names(frame: pd.DataFrame, text: str, sep: str) -> pd.DataFrame:
    header_line = ""
    for line in text.splitlines():
        if line.strip():
            header_line = line
            break
    try:
        header = next(csv.reader([header_line], delimiter=sep, skipinitialspace=True))
    except csv.Error:
        return frame
    header = [cell.strip() for cell in header]
    if len(header) == len(frame.columns):
        frame = frame.copy()
        frame.columns = header
    return frame


def _uniquify_columns(columns: list[object]) -> tuple[list[str], list[str]]:
    seen: dict[str, int] = {}
    names: list[str] = []
    duplicates: list[str] = []
    for raw in columns:
        name = str(raw).strip() or "unnamed"
        if name in seen:
            seen[name] += 1
            duplicates.append(name)
            names.append(f"{name} ({seen[name]})")
        else:
            seen[name] = 1
            names.append(name)
    return names, duplicates


def _prepare_series(series: pd.Series) -> tuple[str, pd.Series, list[str]]:
    kind = classify_series(series)
    notes: list[str] = []
    label = str(series.name)
    if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(series):
        converted = pd.to_datetime(series, errors="coerce", format="mixed")
        failed = int(series.notna().sum() - converted.notna().sum())
        if failed:
            notes.append(
                f"Column {label} looks like dates, but {failed} value(s) could not be read and were left blank."
            )
        return kind, converted, notes
    if kind == "numeric" and not pd.api.types.is_numeric_dtype(series):
        cleaned = series.astype(str).str.replace(",", "", regex=False)
        cleaned = cleaned.where(~cleaned.str.strip().str.lower().isin({"", "nan", "none", "null"}))
        converted = pd.to_numeric(cleaned, errors="coerce")
        original_present = series.notna() & ~series.astype(str).str.strip().isin({"", "nan", "None"})
        failed = int(original_present.sum() - converted.notna().sum())
        if failed:
            notes.append(
                f"Column {label} mixes numbers and text. {failed} non-numeric value(s) were left blank for calculations."
            )
        return kind, converted, notes
    return kind, series, notes


def _is_boolean_like(values: pd.Series) -> bool:
    tokens = {str(value).strip().lower() for value in values.unique()[:12]}
    if not tokens or not tokens <= BOOL_TOKENS:
        return False
    return 1 <= len(tokens) <= 3


def _is_year_like(sample: pd.Series) -> bool:
    return bool(sample.str.fullmatch(r"(19|20)\d{2}").mean() > 0.9)


def _date_rate(sample: pd.Series) -> float:
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean())
