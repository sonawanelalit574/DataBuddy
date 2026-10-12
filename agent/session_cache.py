"""Dataset fingerprint checks so an old chart cannot be shown as current."""

from __future__ import annotations


def replace_dataset(state: dict, fingerprint: str) -> dict:
    if state.get("dataset_fingerprint") != fingerprint:
        state["messages"] = []
        state["last_context"] = {}
        state["explained_ids"] = []
    state["dataset_fingerprint"] = fingerprint
    return state


def charts_for_dataset(messages: list[dict], fingerprint: str) -> list[object]:
    charts = []
    for message in messages:
        if message.get("role") != "assistant" or message.get("fingerprint") != fingerprint:
            continue
        charts.extend(message.get("figures") or [])
    return charts


def is_current(message: dict, fingerprint: str) -> bool:
    if message.get("role") == "system":
        return True
    return message.get("fingerprint") == fingerprint
