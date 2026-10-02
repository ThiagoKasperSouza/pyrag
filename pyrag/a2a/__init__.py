"""A2A (Agent2Agent): peers, cliente/servidor JSON-RPC e Agent Card.

Implementa o subconjunto essencial da spec A2A:
- Descoberta via Agent Card em `/.well-known/agent.json`
- `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`
- Ciclo de vida de task: submitted -> working -> completed/failed
"""

from .card import build_agent_card, skill_definitions
from .client import A2AClient
from .peers import add_peer, list_peers, remove_peer, resolve_peer
from .serialization import dump_a2a
from .server import A2AServer, build_a2a_router, run_a2a_server

__all__ = [
    "A2AClient", "A2AServer", "add_peer", "list_peers", "remove_peer", "resolve_peer",
    "build_agent_card", "skill_definitions", "build_a2a_router", "run_a2a_server",
    "dump_a2a",
]