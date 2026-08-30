from __future__ import annotations

"""Shared A1111 attention + long-CLIP chunk encoder for ImageGen runtimes."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch

from modules.prompt_parsers.a1111_semantics import (
    A1111_CLIP_CONTENT_TOKENS,
    A1111_CLIP_ENCODED_TOKENS,
    parse_a1111_attention,
)


@dataclass(frozen=True)
class A1111PromptCapabilities:
    architecture: str
    attention: bool
    composable_and: bool
    schedules: bool
    alternation: bool
    chunk_break: bool
    long_clip_chunking: bool
    clip_streams: tuple[str, ...] = ()
    non_clip_policy: str = "none"
    pooled_policy: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "algorithms": {
                "attention": "a1111_attention_v1" if self.attention else "unsupported",
                "composable_and": "a1111_composable_guidance_v1" if self.composable_and else "unsupported",
                "schedule": "a1111_schedule_v1" if self.schedules else "unsupported",
                "alternate": "a1111_alternate_v1" if self.alternation else "unsupported",
                "break": "encoder_chunk_break_v1" if self.chunk_break else "unsupported",
                "clip_chunking": "a1111_clip_chunk_v1" if self.long_clip_chunking else "unsupported",
            },
            "clip_streams": list(self.clip_streams),
            "non_clip_policy": self.non_clip_policy,
            "pooled_policy": self.pooled_policy,
        }


@dataclass(frozen=True)
class A1111ClipChunk:
    input_ids: tuple[int, ...]
    multipliers: tuple[float, ...]
    content_tokens: int
    forced_break_before: bool = False

    def __post_init__(self) -> None:
        if len(self.input_ids) != A1111_CLIP_ENCODED_TOKENS:
            raise ValueError("A1111 CLIP chunks must contain exactly 77 encoded token positions.")
        if len(self.multipliers) != A1111_CLIP_ENCODED_TOKENS:
            raise ValueError("A1111 CLIP multiplier rows must contain exactly 77 positions.")


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    encoded = tokenizer(
        str(text),
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", encoded)
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(value) for value in (ids or [])]


def _special_token_ids(tokenizer: Any) -> tuple[int, int, int]:
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    if bos is None:
        bos = getattr(tokenizer, "cls_token_id", None)
    if eos is None:
        eos = getattr(tokenizer, "sep_token_id", None)
    if bos is None or eos is None:
        raise ValueError("A1111 CLIP chunking requires tokenizer BOS/CLS and EOS/SEP token IDs.")
    if pad is None:
        pad = eos
    return int(bos), int(eos), int(pad)


def build_a1111_clip_chunks(
    tokenizer: Any,
    prompt: str,
    *,
    forced_segments: Sequence[str] | None = None,
) -> tuple[A1111ClipChunk, ...]:
    """Tokenize weighted spans into 75-content/77-encoded A1111 CLIP chunks."""

    if forced_segments is not None:
        combined: list[A1111ClipChunk] = []
        for segment_index, segment in enumerate(forced_segments):
            segment_chunks = list(build_a1111_clip_chunks(tokenizer, str(segment or "")))
            if segment_index and segment_chunks:
                first = segment_chunks[0]
                segment_chunks[0] = A1111ClipChunk(
                    input_ids=first.input_ids,
                    multipliers=first.multipliers,
                    content_tokens=first.content_tokens,
                    forced_break_before=True,
                )
            combined.extend(segment_chunks)
        return tuple(combined or build_a1111_clip_chunks(tokenizer, ""))

    bos, eos, pad = _special_token_ids(tokenizer)
    chunks: list[A1111ClipChunk] = []
    content_ids: list[int] = []
    content_weights: list[float] = []
    next_forced_break = False

    def flush(*, forced: bool = False) -> None:
        nonlocal content_ids, content_weights, next_forced_break
        if not content_ids and chunks and not forced:
            return
        count = len(content_ids)
        padded_ids = content_ids + [pad] * (A1111_CLIP_CONTENT_TOKENS - count)
        padded_weights = content_weights + [1.0] * (A1111_CLIP_CONTENT_TOKENS - count)
        chunks.append(
            A1111ClipChunk(
                input_ids=tuple([bos, *padded_ids, eos]),
                multipliers=tuple([1.0, *padded_weights, 1.0]),
                content_tokens=count,
                forced_break_before=bool(next_forced_break),
            )
        )
        content_ids = []
        content_weights = []
        next_forced_break = bool(forced)

    def consume_text(value: str) -> None:
        nonlocal content_ids, content_weights, next_forced_break
        for span in parse_a1111_attention(value):
            if span.weight < 0.0 and span.text == "BREAK":
                flush(forced=True)
                continue
            ids = _token_ids(tokenizer, span.text)
            cursor = 0
            while cursor < len(ids):
                room = A1111_CLIP_CONTENT_TOKENS - len(content_ids)
                if room <= 0:
                    flush()
                    room = A1111_CLIP_CONTENT_TOKENS
                take = min(room, len(ids) - cursor)
                content_ids.extend(ids[cursor : cursor + take])
                content_weights.extend([float(span.weight)] * take)
                cursor += take
                if len(content_ids) == A1111_CLIP_CONTENT_TOKENS:
                    flush()

    consume_text(str(prompt or ""))

    if content_ids or not chunks:
        flush()
    return tuple(chunks)


def _component_device(component: Any) -> torch.device:
    try:
        return next(component.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _select_hidden(outputs: Any, hidden_state_index: int | None) -> torch.Tensor:
    if hidden_state_index is None:
        hidden = getattr(outputs, "last_hidden_state", None)
    else:
        states = getattr(outputs, "hidden_states", None)
        hidden = states[hidden_state_index] if states is not None else None
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise RuntimeError("A1111 CLIP encoder did not return a [batch,tokens,width] hidden-state tensor.")
    return hidden


def _encode_chunk_rows(
    *,
    tokenizer: Any,
    text_encoder: Any,
    chunks: Sequence[A1111ClipChunk],
    hidden_state_index: int | None,
) -> torch.Tensor:
    device = _component_device(text_encoder)
    ids = torch.tensor([chunk.input_ids for chunk in chunks], dtype=torch.long, device=device)
    multipliers = torch.tensor(
        [chunk.multipliers for chunk in chunks],
        dtype=torch.float32,
        device=device,
    )
    kwargs: dict[str, Any] = {"input_ids": ids}
    if hidden_state_index is not None:
        kwargs["output_hidden_states"] = True
    config = getattr(text_encoder, "config", None)
    if bool(getattr(config, "use_attention_mask", False)):
        pad = int(getattr(tokenizer, "pad_token_id", -1))
        if pad >= 0:
            kwargs["attention_mask"] = (ids != pad).to(dtype=torch.long)
    with torch.inference_mode():
        outputs = text_encoder(**kwargs)
    hidden = _select_hidden(outputs, hidden_state_index)
    weights = multipliers.to(dtype=hidden.dtype).unsqueeze(-1)
    original_mean = hidden.mean(dim=(1, 2), keepdim=True)
    weighted = hidden * weights
    new_mean = weighted.mean(dim=(1, 2), keepdim=True)
    safe = torch.where(new_mean.abs() > 1.0e-12, original_mean / new_mean, torch.ones_like(new_mean))
    return weighted * safe


def encode_a1111_clip_batch(
    *,
    tokenizer: Any,
    text_encoder: Any,
    prompts: Iterable[str],
    hidden_state_index: int | None = None,
    forced_segments_by_prompt: Sequence[Sequence[str] | None] | None = None,
) -> torch.Tensor:
    """Encode a prompt batch with A1111 attention and unlimited CLIP chunks.

    Prompts are padded to the batch's maximum chunk count using the tokenizer's
    empty-prompt chunk so the returned tensor remains batchable.
    """

    texts = [str(value or "") for value in prompts]
    if not texts:
        raise ValueError("A1111 CLIP conditioning requires at least one prompt.")
    forced = list(forced_segments_by_prompt or [None] * len(texts))
    if len(forced) != len(texts):
        raise ValueError("forced_segments_by_prompt must match prompt batch length.")
    planned = [
        list(build_a1111_clip_chunks(tokenizer, text, forced_segments=forced[index]))
        for index, text in enumerate(texts)
    ]
    max_chunks = max(len(items) for items in planned)
    empty_chunk = build_a1111_clip_chunks(tokenizer, "")[0]
    flattened: list[A1111ClipChunk] = []
    for items in planned:
        items.extend([empty_chunk] * (max_chunks - len(items)))
        flattened.extend(items)
    encoded = _encode_chunk_rows(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        chunks=flattened,
        hidden_state_index=hidden_state_index,
    )
    width = int(encoded.shape[-1])
    encoded = encoded.reshape(len(texts), max_chunks, A1111_CLIP_ENCODED_TOKENS, width)
    return encoded.reshape(len(texts), max_chunks * A1111_CLIP_ENCODED_TOKENS, width)
