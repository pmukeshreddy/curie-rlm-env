"""Patch the cached Qwen/Qwen3-8B config.json to enable YaRN long context.

Why this script exists: vLLM 0.19 derives its `max_model_len` ceiling from the
model's `config.json` (`max_position_embeddings=40960` for Qwen3-8B), and rejects
any user-specified value above that ceiling unless rope scaling is configured
inside the model config itself. Two passthrough mechanisms we tried failed:

  * `inference.vllm_extra.rope_scaling = {...}` — prime-rl's vllm_extra
    forwarder strips nested dict values; the kwarg never reaches vLLM.
  * `inference.vllm_extra.hf_overrides = '{"...":...}'` — vLLM's pydantic
    schema for ModelConfig.hf_overrides expects `dict[str,any]` or a callable,
    not a JSON string; pydantic rejects it.

So the cleanest fix is to patch the model config on disk once. After this runs,
vLLM reads `max_position_embeddings=65536` + YaRN scaling directly from
config.json and accepts our `inference.vllm_extra.max_model_len = 65536` cap.

Run once on each pod that has the Qwen3-8B HF cache. Idempotent.

Usage:
    uv run python scripts/patch_qwen3_8b_long_context.py
    uv run python scripts/patch_qwen3_8b_long_context.py --revert
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NEW_MAX_POSITION_EMBEDDINGS = 65536
ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 40960,
}
ORIGINAL_MAX_POSITION_EMBEDDINGS = 40960


def find_qwen3_8b_config() -> Path:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = list(cache.glob("models--Qwen--Qwen3-8B/snapshots/*/config.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Qwen3-8B config.json not found under {cache}. "
            "Has the model been downloaded yet?"
        )
    if len(candidates) > 1:
        # Multiple snapshots — patch them all so any cached revision works.
        return candidates[0]  # report the first; we'll patch all in main
    return candidates[0]


def patch_one(path: Path, revert: bool) -> bool:
    """Patch (or revert) a single config.json. Return True if file changed."""
    cfg = json.loads(path.read_text())
    if revert:
        changed = (
            cfg.get("max_position_embeddings") != ORIGINAL_MAX_POSITION_EMBEDDINGS
            or "rope_scaling" in cfg
        )
        cfg["max_position_embeddings"] = ORIGINAL_MAX_POSITION_EMBEDDINGS
        cfg.pop("rope_scaling", None)
    else:
        changed = (
            cfg.get("max_position_embeddings") != NEW_MAX_POSITION_EMBEDDINGS
            or cfg.get("rope_scaling") != ROPE_SCALING
        )
        cfg["max_position_embeddings"] = NEW_MAX_POSITION_EMBEDDINGS
        cfg["rope_scaling"] = ROPE_SCALING
    if changed:
        path.write_text(json.dumps(cfg, indent=2))
    return changed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--revert", action="store_true",
        help="Restore original max_position_embeddings=40960 and remove rope_scaling.",
    )
    args = p.parse_args()

    cache = Path.home() / ".cache" / "huggingface" / "hub"
    paths = sorted(cache.glob("models--Qwen--Qwen3-8B/snapshots/*/config.json"))
    if not paths:
        print(
            f"ERROR: no Qwen3-8B config.json found under {cache}. "
            "Run training once to download the model, then re-run this script.",
            file=sys.stderr,
        )
        return 1

    action = "Reverting" if args.revert else "Patching"
    print(f"{action} {len(paths)} config.json file(s):")
    any_changed = False
    for path in paths:
        changed = patch_one(path, args.revert)
        marker = "CHANGED" if changed else "no-op"
        print(f"  [{marker}] {path}")
        any_changed = any_changed or changed

    if any_changed:
        target = "40960 (original)" if args.revert else f"{NEW_MAX_POSITION_EMBEDDINGS} + YaRN"
        print(f"\nDone. max_position_embeddings is now {target}.")
        print("Restart vLLM (kill and re-run training) for the change to take effect.")
    else:
        print("\nNothing to do — config.json was already in the requested state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
