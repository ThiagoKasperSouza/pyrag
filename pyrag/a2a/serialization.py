"""Helpers de serializacao A2A (a spec usa camelCase nas mensagens)."""

from __future__ import annotations

from typing import Any


def dump_a2a(model: Any) -> dict:
    """Serializa um modelo A2A em dict camelCase."""
    return model.model_dump(mode="json", by_alias=True)