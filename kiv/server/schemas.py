"""Pydantic request/response models for the ollama-compatible API.

Only the subset of fields the common ollama clients (Open WebUI,
Continue, Cline, LangChain) rely on is modelled. Extra fields in
incoming requests are accepted and ignored so forward-compatible
clients still work.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str


class GenerateOptions(BaseModel):
    """Subset of ollama's generation options we honor."""

    model_config = ConfigDict(extra="allow")

    num_predict: int | None = Field(default=None, description="max tokens")
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | None = None
    seed: int | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    stream: bool = True
    options: GenerateOptions | None = None
    keep_alive: Any | None = None
    format: Any | None = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    stream: bool = True
    raw: bool = False
    options: GenerateOptions | None = None
    keep_alive: Any | None = None
    format: Any | None = None


class ShowRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None
    model: str | None = None
