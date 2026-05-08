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
    """Build the locked Gemini judge callable using Vertex AI.

    Required: GOOGLE_CLOUD_PROJECT (GCP project id).
    Optional: GOOGLE_CLOUD_LOCATION (default: us-central1).
    Auth: handled by the SDK via Application Default Credentials
    (`gcloud auth application-default login`), GOOGLE_APPLICATION_CREDENTIALS
    service-account JSON, or workload identity on GCP-hosted runtimes.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project is None or not project.strip():
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT env var required for the Vertex AI judge "
            "used by continual replay phases with retrieval LLMSim."
        )
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    from google import genai

    cfg = yaml.safe_load(_JUDGE_CFG_PATH.read_text())
    judge_model_id = cfg["judge_model_id"]
    client = genai.Client(vertexai=True, project=project, location=location)

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
