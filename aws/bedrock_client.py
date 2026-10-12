"""Amazon Bedrock client.

This module does not choose a model for you. Set BEDROCK_MODEL_ID to a model
that is enabled in your account and region. A call is made only when the
user turns Bedrock on or presses Test connection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.intent import ToolCall
from agent.prompts import (
    SYSTEM_EXPLAIN,
    SYSTEM_SELECT,
    explanation_user_prompt,
    selection_user_prompt,
)
from agent.tool_registry import bedrock_tool_config

logger = logging.getLogger("databuddy.bedrock")


class BedrockError(Exception):
    def __init__(self, code: str, user_message: str):
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message


class BedrockClient:
    def __init__(self, region: str, model_id: str, client: object | None = None):
        if not (region or "").strip() or not (model_id or "").strip():
            raise BedrockError(
                "config",
                "Set an AWS region and a Bedrock model ID before using Amazon Bedrock.",
            )
        self.region = region.strip()
        self.model_id = model_id.strip()
        self._client = client

    def select_tools(self, question: str, schema: str, prior_note: str) -> list[ToolCall]:
        response = self._converse(
            SYSTEM_SELECT,
            selection_user_prompt(question, schema, prior_note),
            tools=bedrock_tool_config(),
            max_tokens=800,
            temperature=0.0,
        )
        return parse_converse_tools(response)

    def explain(self, question: str, draft: str) -> str:
        response = self._converse(
            SYSTEM_EXPLAIN,
            explanation_user_prompt(question, draft),
            tools=None,
            max_tokens=500,
            temperature=0.2,
        )
        text = extract_text(response).strip()
        if not text:
            raise BedrockError("malformed", "Amazon Bedrock returned an empty explanation.")
        return text

    def test_connection(self) -> dict[str, str]:
        response = self._converse(
            "Reply with the single word ready.",
            "Reply with the single word ready.",
            tools=None,
            max_tokens=20,
            temperature=0.0,
        )
        text = extract_text(response).strip()
        if not text:
            raise BedrockError("malformed", "Amazon Bedrock returned an empty test response.")
        return {"ok": "true", "reply": text[:80]}

    def _converse(
        self,
        system: str,
        user: str,
        tools: dict[str, Any] | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        client = self._runtime()
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if tools:
            request["toolConfig"] = tools
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = client.converse(**request)
                logger.info("bedrock converse ok model=%s region=%s", self.model_id, self.region)
                return response
            except BedrockError:
                raise
            except Exception as exc:
                mapped = map_client_error(exc)
                last_error = mapped
                if mapped.code == "throttling" and attempt == 0:
                    logger.info("bedrock throttled; retrying once")
                    continue
                logger.info("bedrock converse failed code=%s", mapped.code)
                raise mapped from exc
        raise last_error or BedrockError("unknown", "Amazon Bedrock did not respond.")

    def _runtime(self):
        if self._client is not None:
            return self._client
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise BedrockError(
                "dependency",
                "boto3 is not installed, so Amazon Bedrock is unavailable. Demo Mode still works.",
            ) from exc
        return boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            config=Config(
                connect_timeout=5,
                read_timeout=40,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )


def credentials_available() -> bool:
    """Look for local credentials. This does not call AWS."""

    try:
        import boto3
    except ImportError:
        return False
    try:
        credentials = boto3.Session().get_credentials()
        if credentials is None:
            return False
        frozen = credentials.get_frozen_credentials()
    except Exception:
        return False
    return bool(getattr(frozen, "access_key", None) and getattr(frozen, "secret_key", None))


def map_client_error(exc: Exception) -> BedrockError:
    name = exc.__class__.__name__
    if name in {"NoCredentialsError", "PartialCredentialsError"}:
        return BedrockError(
            "credentials",
            "AWS credentials were not found. Demo Mode can still analyze the file on this machine. "
            "To use Bedrock, configure the standard AWS credential provider chain.",
        )
    if name in {"ReadTimeoutError", "ConnectTimeoutError", "ModelTimeoutException"}:
        return BedrockError(
            "timeout",
            "Amazon Bedrock did not answer before the timeout. Demo Mode can answer locally.",
        )
    if name in {"EndpointConnectionError", "ConnectionClosedError"}:
        return BedrockError(
            "network",
            "I could not reach Amazon Bedrock. Check the region and your network, or stay in Demo Mode.",
        )
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
    if code in {"AccessDeniedException", "UnauthorizedException"}:
        return BedrockError(
            "access_denied",
            "Amazon Bedrock denied this request. The IAM principal needs bedrock:InvokeModel "
            "for the model you selected, and that model must be enabled in the region.",
        )
    if code in {"ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"}:
        return BedrockError(
            "throttling",
            "Amazon Bedrock is throttling requests. Wait a moment and try again, or use Demo Mode.",
        )
    if code in {"ResourceNotFoundException"}:
        return BedrockError(
            "model_unavailable",
            "That Bedrock model ID was not found in this region. Enable a model in the Bedrock console "
            "and put its ID in BEDROCK_MODEL_ID. DataBuddy does not assume a default model.",
        )
    if code in {"ValidationException"}:
        return BedrockError(
            "validation",
            "Amazon Bedrock rejected the request. Check the model ID, the region, and that the model "
            "supports the Converse API.",
        )
    if name == "ClientError" and code:
        return BedrockError(
            "aws",
            f"Amazon Bedrock returned {code}. The dataset was not sent again. You can continue in Demo Mode.",
        )
    return BedrockError(
        "unknown",
        "Amazon Bedrock could not complete the request. You can continue in Demo Mode.",
    )


def parse_converse_tools(response: dict[str, Any]) -> list[ToolCall]:
    content = _content_blocks(response)
    calls: list[ToolCall] = []
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict):
            calls.append(_call_from_tool_use(tool_use))
        elif isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    if calls:
        return calls
    text = "\n".join(text_parts).strip()
    if not text:
        raise BedrockError("malformed", "Amazon Bedrock returned an empty response.")
    if "{" not in text:
        return []
    return _calls_from_text(text)


def extract_text(response: dict[str, Any]) -> str:
    parts = []
    for block in _content_blocks(response):
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _content_blocks(response: dict[str, Any]) -> list[Any]:
    try:
        content = response["output"]["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise BedrockError("malformed", "Amazon Bedrock returned a response I could not read.") from exc
    if not isinstance(content, list):
        raise BedrockError("malformed", "Amazon Bedrock returned a response I could not read.")
    return content


def _call_from_tool_use(tool_use: dict[str, Any]) -> ToolCall:
    name = tool_use.get("name")
    raw_input = tool_use.get("input", {})
    if not isinstance(name, str) or not name.strip():
        raise BedrockError("malformed", "Amazon Bedrock returned a tool request without a name.")
    if not isinstance(raw_input, dict):
        raise BedrockError("malformed", "Amazon Bedrock returned tool arguments that were not an object.")
    return ToolCall(name.strip(), _stringify_arguments(raw_input))


def _calls_from_text(text: str) -> list[ToolCall]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise BedrockError("malformed", "Amazon Bedrock returned a tool request that was not valid JSON.")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise BedrockError(
            "malformed",
            "Amazon Bedrock returned a tool request that was not valid JSON.",
        ) from exc
    tools = payload.get("tools") if isinstance(payload, dict) else None
    if not isinstance(tools, list):
        raise BedrockError("malformed", "Amazon Bedrock returned JSON that did not list tools.")
    calls = []
    for item in tools:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise BedrockError("malformed", "Amazon Bedrock returned a tool request I could not use.")
        arguments = item.get("arguments", item.get("input", {}))
        if not isinstance(arguments, dict):
            raise BedrockError("malformed", "Amazon Bedrock returned tool arguments that were not an object.")
        calls.append(ToolCall(item["name"], _stringify_arguments(arguments)))
    return calls


def _stringify_arguments(arguments: dict[str, Any]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, value in arguments.items():
        if isinstance(value, (dict, list)):
            raise BedrockError(
                "malformed",
                "Amazon Bedrock returned nested tool arguments, so I did not run them.",
            )
        clean[str(key)] = "" if value is None else str(value)
    return clean
