"""Optional Lambda entry point for the same local analytical tools.

This handler always uses Demo Mode. It does not call Amazon Bedrock, read
S3, or create AWS resources. Wiring it to API Gateway is a manual step
described in infrastructure/README.md.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.orchestrator import Orchestrator
from analytics.dataset_loader import DatasetError, load_csv_bytes

logger = logging.getLogger("databuddy.lambda")

MAX_CSV_BYTES = 1_000_000
MAX_QUESTION_CHARS = 2_000


def lambda_handler(event: dict[str, Any] | None, context: object) -> dict[str, Any]:
    del context
    payload, proxy = _payload(event or {})
    status, body = _handle(payload)
    if proxy:
        return {
            "statusCode": status,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body),
        }
    return body


def _handle(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if payload.get("__error"):
        return 400, {"error": str(payload["__error"])}
    question = str(payload.get("question") or "").strip()
    csv_text = payload.get("csv_text")
    if not isinstance(csv_text, str) or not csv_text.strip():
        return 400, {
            "error": "Add csv_text with the CSV contents. This handler does not read Amazon S3."
        }
    if len(question) > MAX_QUESTION_CHARS:
        return 400, {"error": f"Questions are limited to {MAX_QUESTION_CHARS} characters."}
    encoded = csv_text.encode("utf-8")
    if len(encoded) > MAX_CSV_BYTES:
        return 400, {"error": "This endpoint accepts CSV text up to 1 MB."}
    if not question:
        return 400, {"error": "Add a question about the CSV."}
    try:
        bundle = load_csv_bytes(encoded, "upload.csv", max_bytes=MAX_CSV_BYTES)
        response = Orchestrator(use_bedrock=False).answer(
            question,
            bundle.frame,
            bundle.fingerprint,
            dataset_warnings=bundle.warnings,
        )
    except DatasetError as exc:
        return 400, {"error": exc.user_message}
    except Exception:
        logger.error("lambda analysis failed")
        return 500, {"error": "The analysis could not be completed."}
    body = response.to_public_dict()
    body["dataset"] = {
        "rows": int(len(bundle.frame)),
        "columns": list(map(str, bundle.frame.columns)),
    }
    return 200, body


def _payload(event: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if "csv_text" in event or "question" in event:
        return event, False
    if "body" not in event:
        return event, False
    if event.get("isBase64Encoded"):
        return {"__error": "Send plain JSON, not a base64 body."}, True
    raw = event.get("body") or ""
    if raw == "":
        return {}, True
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__error": "The request body must be JSON."}, True
    if not isinstance(parsed, dict):
        return {"__error": "The JSON body must be an object."}, True
    return parsed, True
