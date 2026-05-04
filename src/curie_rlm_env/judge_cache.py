"""Stage 5b — LLMSim judge call cache.

Cache lifetime = one training step. The trainer is responsible for calling
clear_cache() at step boundaries; Stage 5b ships the cache module + a hook
function but does not wire the cache lifetime (that's prime-rl's job).

Key: (sha256(gt_item)[:16], sha256(prediction)[:16]).

Within a training step, identical (gt_item, prediction) pairs return the
cached judge response — saves repeated Gemini calls when GRPO produces
the same output across rollouts of the same example, or when LLMSim
loops over the same gt list with stable prediction list.

Two entry points:
- cached_llmsim_sync — used by the sync llm_sim() in scorers.py
- cached_llmsim — async-aware version for prime-rl orchestrator hooks
Both share the module-level _CACHE.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Callable

# Module-level cache. Cleared by trainer hook at each step boundary.
_CACHE: dict[tuple[str, str], Any] = {}


def _hash_obj(obj: Any) -> str:
    """sha256-prefix-16 of JSON-serialized object (deterministic key ordering)."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def cached_llmsim_sync(
    judge_client: Callable[[str], str],
    gt_item: Any,
    prediction: Any,
    prompt: str,
) -> str:
    """Sync cache wrapper. Used by sync llm_sim() in scorers.py."""
    key = (_hash_obj(gt_item), _hash_obj(prediction))
    if key in _CACHE:
        return _CACHE[key]
    result = judge_client(prompt)
    _CACHE[key] = result
    return result


async def cached_llmsim(
    judge_client: Callable[[str], Any],
    gt_item: Any,
    prediction: Any,
    prompt: str,
) -> Any:
    """Async-aware cache wrapper. judge_client may be sync or async; both supported."""
    key = (_hash_obj(gt_item), _hash_obj(prediction))
    if key in _CACHE:
        return _CACHE[key]
    result = judge_client(prompt)
    if asyncio.iscoroutine(result):
        result = await result
    _CACHE[key] = result
    return result


def clear_cache() -> None:
    """Reset the cache. Trainer hooks this at the end of each training step."""
    _CACHE.clear()


def cache_size() -> int:
    """Public for telemetry + tests."""
    return len(_CACHE)
