"""Modelos de dados do RAG e do protocolo A2A."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def to_camel(s: str) -> str:
    head, *rest = s.split("_")
    return head + "".join(w.capitalize() for w in rest)


class A2AModel(BaseModel):
    """Base dos modelos A2A: serializa em camelCase, como pede a spec."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    def model_dump(self, *args, by_alias: bool = True, **kwargs) -> dict:
        """Serializa sempre com alias camelCase, a menos que seja desativado explicitamente."""
        return super().model_dump(*args, by_alias=by_alias, **kwargs)


# --------------------------------------------------------------------------- RAG
class Source(BaseModel):
    path: str
    title: str = ""
    chunk: int = 0
    score: float = 0.0
    preview: str = ""


class Citation(BaseModel):
    index: int
    source: Source


class RetrievedChunk(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    dense_score: float | None = None
    lexical_score: float | None = None
    source: Source | None = None

    def citation(self, index: int = 1) -> Citation:
        self.source.index = index
        return Citation(index=index, source=self.source or Source(path="desconhecido"))


class RAGAnswer(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    provider: str = ""
    model: str = ""
    elapsed_ms: int = 0
    used_fallback: bool = False


# -------------------------------------------------------------------------- A2A
Role = Literal["user", "agent", "system"]
TaskState = Literal[
    "submitted", "working", "input-required", "completed", "failed", "canceled", "unknown"
]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


class A2APart(A2AModel):
    kind: Literal["text", "file", "data"] = "text"
    text: str | None = None
    mime_type: str | None = None
    data: dict[str, Any] | None = None


class A2AMessage(A2AModel):
    role: Role = "user"
    parts: list[A2APart] = Field(default_factory=list)
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    task_id: str | None = None
    context_id: str | None = None
    created_at: float = Field(default_factory=time.time)

    @classmethod
    def text_message(cls, text: str, role: Role = "user", **kw) -> "A2AMessage":
        return cls(role=role, parts=[A2APart(kind="text", text=text)], **kw)

    def text(self) -> str:
        return "\n".join(p.text for p in self.parts if p.kind == "text" and p.text)


class Artifact(A2AModel):
    artifact_id: str = Field(default_factory=lambda: new_id("art"))
    name: str = "resultado"
    description: str = ""
    parts: list[A2APart] = Field(default_factory=list)

    def text(self) -> str:
        return "\n".join(p.text for p in self.parts if p.text)


class TaskStatus(A2AModel):
    state: TaskState = "submitted"
    message: A2AMessage | None = None
    timestamp: float = Field(default_factory=time.time)


class A2ATask(A2AModel):
    id: str = Field(default_factory=lambda: new_id("task"))
    context_id: str = Field(default_factory=lambda: new_id("ctx"))
    status: TaskStatus = Field(default_factory=TaskStatus)
    history: list[A2AMessage] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSkill(A2AModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=lambda: ["text"])
    output_modes: list[str] = Field(default_factory=lambda: ["text"])


class AgentCapabilities(A2AModel):
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = True


class AgentCard(A2AModel):
    """Agent Card do protocolo A2A (agent discovery)."""

    protocol_version: str = "0.3.0"
    name: str
    description: str
    version: str = "0.1.0"
    url: str
    preferred_transport: str = "JSONRPC"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    default_input_modes: list[str] = Field(default_factory=lambda: ["text"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text"])
    skills: list[AgentSkill] = Field(default_factory=list)
    provider: dict[str, str] = Field(default_factory=dict)


class Peer(A2AModel):
    name: str
    url: str
    skills: list[str] = Field(default_factory=list)
    description: str = ""


class JSONRPCError(A2AModel):
    code: int
    message: str
    data: Any | None = None


class JSONRPCResponse(A2AModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Any = None
    result: Any | None = None
    error: JSONRPCError | None = None


class A2AMethod(str, Enum):
    SEND_MESSAGE = "message/send"
    STREAM_MESSAGE = "message/stream"
    GET_TASK = "tasks/get"
    CANCEL_TASK = "tasks/cancel"
    LIST_TASKS = "tasks/list"
    AGENT_CARD = "agent/getAuthenticatedExtendedCard"