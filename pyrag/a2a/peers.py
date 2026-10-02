"""Registro local de agentes remotos (peers) e descoberta via Agent Card."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import get_settings
from ..models import Peer

log = logging.getLogger(__name__)


def _path() -> Path:
    p = get_settings().a2a_peers_file
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def list_peers() -> list[Peer]:
    p = _path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("peers.json invalido; iniciando vazio.")
        return []
    return [Peer(**d) for d in data if isinstance(d, dict)]


def save_peers(peers: list[Peer]) -> None:
    _path().write_text(json.dumps([p.model_dump() for p in peers], indent=2, ensure_ascii=False),
                       encoding="utf-8")


def _skill_ids(card: dict) -> list[str]:
    """Aceita skills como dicts (JSON) ou objetos."""
    out = []
    for s in card.get("skills") or []:
        if isinstance(s, dict):
            sid = s.get("id") or s.get("name")
        else:
            sid = getattr(s, "id", None)
        if sid:
            out.append(sid)
    return out


def add_peer(name: str, url: str, skills: list[str] | None = None, description: str = "") -> Peer:
    """Adiciona/substitui um peer e tenta descobrir as skills pelo Agent Card."""
    url = url.rstrip("/")
    if not skills:
        try:
            card = discover_card(url)
            skills = _skill_ids(card)
            description = description or card.get("description", "")
            log.info("Agent Card de %s: %s skills", name, skills)
        except Exception as exc:
            log.info("Nao foi possivel ler o Agent Card de %s (%s); registrando sem skills.", url, exc)
    peer = Peer(name=name, url=url, skills=skills or [], description=description)
    peers = [p for p in list_peers() if p.name != name and p.url != url]
    peers.append(peer)
    save_peers(peers)
    return peer


def remove_peer(name_or_url: str) -> bool:
    peers = list_peers()
    kept = [p for p in peers if p.name != name_or_url and p.url != name_or_url.rstrip("/")]
    save_peers(kept)
    return len(kept) != len(peers)


def resolve_peer(name_or_url: str) -> Peer | None:
    """Resolve por nome exato, URL exata ou prefixo (case-insensitive)."""
    target = (name_or_url or "").strip().rstrip("/").lower()
    peers = list_peers()
    for p in peers:
        if p.name.lower() == target:
            return p
    for p in peers:
        if p.url.lower().rstrip("/") == target:
            return p
    for p in peers:
        if target and (target in p.name.lower() or target in p.url.lower()):
            return p
    return None


def discover_card(url: str) -> dict:
    """Le o Agent Card publico do agente remoto."""
    import httpx

    base = url.rstrip("/")
    candidates = [f"{base}/.well-known/agent.json", f"{base}/.well-known/agent-card.json",
                  f"{base}/agent/card"]
    last: Exception | None = None
    for c in candidates:
        try:
            r = httpx.get(c, timeout=10.0)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code} em {c}")
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Agent Card nao encontrado em {url}: {last}")