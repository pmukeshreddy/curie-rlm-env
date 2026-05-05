"""Live judge client factory for CURIE LLMSim.

Repository quotes anchoring this module:
- config/judge.yaml: "judge_model_id: \"gemini-2.5-pro\""
- config/judge.yaml: "NEVER swap, downgrade, or fall back — identity is the freeze."
- src/curie_rlm_env/rubric.py: "LLMSim for {task_id} requires judge_client; got None"
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import yaml

_JUDGE_CFG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "judge.yaml"


def make_gemini_judge_from_env() -> Callable[[str], str]:
    """Build the locked Gemini judge callable from GEMINI_API_KEY."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None or not api_key.strip():
        raise RuntimeError(
            "GEMINI_API_KEY env var required for continual replay phases with retrieval LLMSim."
        )

    from google import genai

    cfg = yaml.safe_load(_JUDGE_CFG_PATH.read_text())
    judge_model_id = cfg["judge_model_id"]
    client = genai.Client(api_key=api_key)

    def judge_call(prompt: str) -> str:
        response = client.models.generate_content(
            model=judge_model_id,
            contents=prompt,
        )
        text = response.text
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"{judge_model_id} returned empty LLMSim judge text.")
        return text

    return judge_call
