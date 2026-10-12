"""Agent routing, tool validation, demo mode, and Bedrock failure behavior."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from agent.intent import ToolCall, route, suggested_questions
from agent.orchestrator import Orchestrator, dataset_schema_text, numbers_supported
from agent.session_cache import charts_for_dataset, replace_dataset
from agent.tool_registry import execute
from aws.bedrock_client import BedrockError, map_client_error, parse_converse_tools
from config import get_settings
from infrastructure.lambda_handler import lambda_handler


def sales_frame():
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-02", "2024-02-18"]),
            "product": ["A", "A", "B", "B"],
            "region": ["East", "West", "East", "East"],
            "quantity": [1, 2, 3, 4],
            "revenue": [10.0, 5.0, 3.0, 4.0],
            "profit": [4.0, 1.0, 1.0, 2.0],
        }
    )


class FakeLLM:
    def __init__(self, mode: str):
        self.mode = mode
        self.schemas: list[str] = []

    def select_tools(self, question, schema, prior_note):
        del question, prior_note
        self.schemas.append(schema)
        if self.mode == "throttle":
            raise BedrockError(
                "throttling",
                "Amazon Bedrock is throttling requests. Wait a moment and try again, or use Demo Mode.",
            )
        if self.mode == "malformed":
            raise BedrockError(
                "malformed",
                "Amazon Bedrock returned a tool request that was not valid JSON.",
            )
        if self.mode == "bad_tool":
            return [ToolCall("os.system", {"command": "ls"})]
        if self.mode == "empty":
            return []
        return [ToolCall("dataset_overview", {})]

    def explain(self, question, draft):
        del question
        if self.mode == "numbers":
            return "The secret total is 999999."
        if self.mode == "explain_fail":
            raise BedrockError("malformed", "Amazon Bedrock returned an empty explanation.")
        return "Here is the checked result. " + draft


def test_unknown_tool_and_bad_arguments_are_rejected():
    data = sales_frame()
    unknown = execute("os.system", {"command": "ls"}, data)
    assert not unknown.ok
    assert "os.system" in unknown.message
    bad = execute(
        "calculate_group_metrics",
        {
            "group_column": "product",
            "metric_column": "revenue",
            "aggregation": "__import__",
            "code": "print(1)",
        },
        data,
    )
    assert not bad.ok
    assert "Traceback" not in bad.message


def test_overview_ranking_trend_and_chart():
    data = sales_frame()
    agent = Orchestrator(use_bedrock=False)
    overview = agent.answer("Give me a quick overview of this dataset.", data, "fp-sales")
    assert overview.mode == "demo"
    assert overview.evidences[0].records_analyzed == 4
    assert "4" in overview.answer
    assert overview.fingerprint == "fp-sales"

    ranking = agent.answer("Find the top-performing products.", data, "fp-sales")
    assert "calculate_group_metrics" in ranking.tools_used
    assert "A" in ranking.answer
    assert "15" in ranking.answer
    assert ranking.figures

    trend = agent.answer("Show monthly sales trends.", data, "fp-sales")
    assert "calculate_time_trend" in trend.tools_used
    assert "15" in trend.answer
    assert "7" in trend.answer
    assert trend.figures
    joined = " ".join(evidence.formula for evidence in trend.evidences)
    assert "revenue" in joined

    quality = agent.answer("Check for missing values and duplicates.", data, "fp-sales")
    assert "find_missing_values" in quality.tools_used
    assert "find_duplicates" in quality.tools_used
    assert quality.evidences


def test_outliers_suggestions_and_unsupported_questions():
    data = sales_frame()
    agent = Orchestrator(use_bedrock=False)
    outliers = agent.answer("Find unusual values and potential outliers.", data, "fp")
    assert "detect_outliers" in outliers.tools_used
    assert outliers.explanation
    suggestions = agent.answer("What should I investigate next?", data, "fp")
    assert suggestions.tools_used == ["suggest_next_questions"]
    assert suggestions.findings == ["No metric was calculated for this reply."]
    assert suggestions.next_questions
    unknown = agent.answer("What is the meaning of life?", data, "fp")
    assert unknown.tools_used == []
    assert unknown.findings == []
    assert "fixed set of checks" in unknown.answer
    forecast = agent.answer("Predict revenue for next year", data, "fp")
    assert forecast.tools_used == []
    assert "forecast" in forecast.answer.lower()


def test_missing_column_lists_real_columns():
    result = Orchestrator().answer("total revenue by customer", sales_frame(), "fp")
    assert result.tools_used == []
    assert "customer" in result.answer
    assert "product" in result.answer
    assert "999" not in result.answer


def test_follow_up_reuses_the_previous_measure():
    agent = Orchestrator()
    data = sales_frame()
    first = agent.answer("Find the top-performing products.", data, "fp")
    second = agent.answer("Break that down by region.", data, "fp", prior=first.context)
    assert second.context.get("group_column") == "region"
    assert second.context.get("metric_column") == "revenue"
    assert "East" in second.answer


def test_dataset_switch_clears_stale_charts():
    state = {"dataset_fingerprint": "one", "messages": [{"role": "assistant", "fingerprint": "one", "figures": ["old"]}]}
    replace_dataset(state, "two")
    assert state["messages"] == []
    assert state["dataset_fingerprint"] == "two"
    stale = [{"role": "assistant", "fingerprint": "one", "figures": ["old-chart"]}]
    current = [{"role": "assistant", "fingerprint": "two", "figures": ["new-chart"]}]
    assert charts_for_dataset(stale + current, "two") == ["new-chart"]


def test_demo_mode_without_bedrock_configuration(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "")
    monkeypatch.setenv("AWS_REGION", "")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "")
    settings = get_settings()
    assert settings.bedrock_configured is False
    result = Orchestrator(use_bedrock=False).answer(
        "Give me a quick overview of this dataset.",
        sales_frame(),
        "fp",
    )
    assert result.mode == "demo"
    assert "Bedrock" not in result.mode_label


def test_schema_omits_cell_values_unless_preview_is_requested():
    data = pd.DataFrame({"note": ["SECRET_VALUE"], "revenue": [3]})
    hidden = dataset_schema_text(data, include_preview=False)
    shown = dataset_schema_text(data, include_preview=True)
    assert "SECRET_VALUE" not in hidden
    assert "Cell values were not included." in hidden
    assert "SECRET_VALUE" in shown


def test_bedrock_failures_fall_back_to_local_calculations():
    data = sales_frame()
    for mode in ("throttle", "malformed", "bad_tool", "empty"):
        result = Orchestrator(use_bedrock=True, llm=FakeLLM(mode)).answer(
            "Give me a quick overview of this dataset.",
            data,
            "fp",
        )
        assert result.mode == "local_fallback"
        assert result.evidences
        assert "os.system" not in result.tools_used
        assert "Demo Mode" in result.warnings[0] or "local" in result.warnings[0].lower()


def test_bedrock_discards_numbers_the_model_invents():
    result = Orchestrator(use_bedrock=True, llm=FakeLLM("numbers")).answer(
        "Give me a quick overview of this dataset.",
        sales_frame(),
        "fp",
    )
    assert result.mode == "bedrock"
    assert "999999" not in result.answer
    assert any("verified wording" in warning for warning in result.warnings)


def test_bedrock_can_rewrite_without_new_numbers():
    result = Orchestrator(use_bedrock=True, llm=FakeLLM("overview")).answer(
        "Give me a quick overview of this dataset.",
        sales_frame(),
        "fp",
    )
    assert result.mode == "bedrock"
    assert result.answer.startswith("Here is the checked result.")
    assert "4" in result.answer


def test_number_guard_and_converse_parser():
    assert numbers_supported("The total is 15.", "Highest: A at 15.")
    assert not numbers_supported("The total is 999999.", "Highest: A at 15.")
    parsed = parse_converse_tools(
        {
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"toolUseId": "1", "name": "find_duplicates", "input": {}}}
                    ]
                }
            }
        }
    )
    assert parsed[0].name == "find_duplicates"
    from_text = parse_converse_tools(
        {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": json.dumps(
                                {"tools": [{"name": "dataset_overview", "arguments": {}}]}
                            )
                        }
                    ]
                }
            }
        }
    )
    assert from_text[0].name == "dataset_overview"
    with pytest.raises(BedrockError):
        parse_converse_tools(
            {"output": {"message": {"content": [{"text": "plan {not json"}]}}}
        )


def test_access_denied_is_a_named_user_message():
    error = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "hidden account detail"}},
        "Converse",
    )
    mapped = map_client_error(error)
    assert mapped.code == "access_denied"
    assert "hidden account" not in mapped.user_message
    assert "Traceback" not in mapped.user_message


def test_suggested_questions_are_routable():
    data = sales_frame()
    for question in suggested_questions(data):
        plan = route(question, data)
        assert plan.calls, question
        assert plan.clarification is None


def test_lambda_handler_answers_locally_and_rejects_bad_input():
    event = {
        "question": "Give me a quick overview of this dataset.",
        "csv_text": "product,revenue\nA,10\nB,5\n",
    }
    body = lambda_handler(event, None)
    assert body["mode"] == "demo"
    assert body["evidences"][0]["records_analyzed"] == 2
    assert "Traceback" not in json.dumps(body)
    bad = lambda_handler({"question": "overview", "csv_text": ""}, None)
    assert "error" in bad
    assert "Traceback" not in json.dumps(bad)
    proxied = lambda_handler(
        {"body": json.dumps({"question": "", "csv_text": "a,b\n1,2\n"}), "isBase64Encoded": False},
        None,
    )
    assert proxied["statusCode"] == 400


def test_tool_timeout(monkeypatch):
    from agent.guard import CalculationTimeout, run_with_timeout

    monkeypatch.setattr("agent.guard.threading.current_thread", lambda: __import__("threading").main_thread())
    with pytest.raises(CalculationTimeout):
        run_with_timeout(lambda: __import__("time").sleep(0.4), 0.05)
