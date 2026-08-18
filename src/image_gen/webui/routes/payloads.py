from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NamedPayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    values: dict[str, Any] = Field(default_factory=dict)
    plugin_id: str | None = None
    overwrite: bool = True


class PromptPresetPayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    positive_prompt: str = ""
    negative_prompt: str = ""
    notes: str = ""
    tags: list[str] = Field(default_factory=list)


class ModelActivationPayload(BaseModel):
    model_path: str = Field(min_length=1)


class OutputFolderPayload(BaseModel):
    path: str = ""


class CivitaiCredentialPayload(BaseModel):
    api_key: str = Field(min_length=1, max_length=4096)
