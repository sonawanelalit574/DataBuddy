"""Runtime configuration.

Secrets and model IDs come from the environment or a local .env file.
Nothing in this module grants AWS access on its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    region: str
    model_id: str
    max_upload_mb: float
    max_rows: int = 200_000
    max_columns: int = 300
    tool_timeout_seconds: float = 30.0

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def bedrock_configured(self) -> bool:
        return bool(self.region.strip() and self.model_id.strip())


def get_settings() -> Settings:
    raw_mb = (os.getenv("MAX_UPLOAD_MB") or "10").strip() or "10"
    try:
        max_mb = float(raw_mb)
    except ValueError:
        max_mb = 10.0
    if max_mb <= 0 or max_mb > 100:
        max_mb = 10.0
    region = (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "").strip()
    model_id = (os.getenv("BEDROCK_MODEL_ID") or "").strip()
    return Settings(region=region, model_id=model_id, max_upload_mb=max_mb)
