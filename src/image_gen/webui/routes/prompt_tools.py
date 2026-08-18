from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from modules.prompt_parsers import (
    PROMPT_MERGE_CONTRACT_VERSION,
    PROMPT_ROUTE_CONTRACT_VERSION,
    PROMPT_SHADOW_CONTRACT_VERSION,
)


def build_prompt_tools_router(*, prompt_parsers, prompt_configuration) -> APIRouter:
    router = APIRouter()

    @router.get("/api/prompt-parsers")
    async def prompt_parser_catalog() -> dict[str, Any]:
        return {
            "default": "legacy",
            "contract_version": "image-gen-prompt-parser-v1",
            "canonical_contract_version": "image-gen-canonical-prompt-v1",
            "route_contract_version": PROMPT_ROUTE_CONTRACT_VERSION,
            "shadow_contract_version": PROMPT_SHADOW_CONTRACT_VERSION,
            "merge_contract_version": PROMPT_MERGE_CONTRACT_VERSION,
            "parsers": prompt_parsers.descriptors(),
        }


    @router.get("/api/prompt-shortcut-profiles")
    async def list_prompt_shortcut_profiles() -> list[dict[str, Any]]:
        return prompt_configuration.list_profiles()


    @router.post("/api/prompt-shortcut-profiles/validate")
    async def validate_prompt_shortcut_profile(payload: dict[str, Any]) -> dict[str, Any]:
        return prompt_configuration.validate_profile(payload)


    @router.post("/api/prompt-shortcut-profiles")
    async def save_prompt_shortcut_profile(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            saved = prompt_configuration.save_profile(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"profile": saved, "profiles": prompt_configuration.list_profiles()}


    @router.delete("/api/prompt-shortcut-profiles/{profile_id}")
    async def delete_prompt_shortcut_profile(profile_id: str) -> dict[str, Any]:
        try:
            deleted = prompt_configuration.delete_profile(profile_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"deleted": deleted, "profiles": prompt_configuration.list_profiles()}


    @router.get("/api/prompt-parser-presets")
    async def list_prompt_parser_presets() -> list[dict[str, Any]]:
        return prompt_configuration.parser_presets()


    @router.post("/api/prompt-parser-presets")
    async def save_prompt_parser_preset(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            saved = prompt_configuration.save_preset(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"preset": saved, "presets": prompt_configuration.parser_presets()}


    @router.delete("/api/prompt-parser-presets/{preset_id}")
    async def delete_prompt_parser_preset(preset_id: str) -> dict[str, Any]:
        try:
            deleted = prompt_configuration.delete_preset(preset_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"deleted": deleted, "presets": prompt_configuration.parser_presets()}


    @router.post("/api/prompts/translate")
    async def translate_prompts(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return prompt_configuration.translate_preview(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/prompts/preflight")
    async def preflight_prompts(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return prompt_configuration.preflight_report(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    return router
