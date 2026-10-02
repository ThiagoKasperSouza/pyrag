"""Memoria de conversa por sessao, persistida em JSONL sob data_dir/sessions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .config import get_settings

log = logging.getLogger(__name__)


def sessions_dir() -> Path:
    d = get_settings().data_dir / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_path(session: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session)[:64] or "default"
    return sessions_dir() / f"{safe}.jsonl"


def load_session(session: str = "default", max_turns: int | None = None) -> list[BaseMessage]:
    """Le o historico e devolve mensagens LangChain (respeitando o limite de turnos)."""
    p = session_path(session)
    if not p.exists():
        return []
    max_turns = max_turns or get_settings().history_turns
    turns: list[BaseMessage] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        turns.append(HumanMessage(content=rec.get("question", "")))
        turns.append(AIMessage(content=rec.get("answer", "")))
    return turns[-max_turns * 2:]


def append_turn(session: str, question: str, answer: str, max_turns: int | None = None) -> None:
    p = session_path(session)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"question": question, "answer": answer}, ensure_ascii=False) + "\n")
    trim(session, max_turns)


def trim(session: str, max_turns: int | None = None) -> int:
    """Mantem apenas os N ultimos turnos no arquivo."""
    max_turns = max_turns or get_settings().history_turns
    p = session_path(session)
    if not p.exists():
        return 0
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = lines[-max_turns:]
    if len(kept) != len(lines):
        p.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return len(kept)


def clear_session(session: str = "default") -> bool:
    p = session_path(session)
    if p.exists():
        p.unlink()
        return True
    return False


def list_sessions() -> list[str]:
    return sorted(p.stem for p in sessions_dir().glob("*.jsonl"))