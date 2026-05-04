"""Stage 5.5 — RFT contingency tests.

Test fixtures use small inline synthetic JSONL records. These exercise the
FILTER LOGIC of extract_high_reward_rollouts.py — they are unit-test inputs,
NOT training data, and are CLAUDE.md L15 compliant (filter logic ≠ synthetic
training data; Stage 3 schema-validator tests follow the same pattern).
"""
from __future__ import annotations

import json
import py_compile
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

# Make the script importable as a module
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
_CONFIGS = _PROJECT_ROOT / "configs"
_DOCS = _PROJECT_ROOT / "docs"

sys.path.insert(0, str(_SCRIPTS))
import extract_high_reward_rollouts as ex  # noqa: E402


def _mk_record(task_id: str, reward: float, content: str = "demo") -> dict:
    return {
        "prompt": [{"role": "user", "content": f"Q for {task_id}"}],
        "completion": [{"role": "assistant", "content": f"A: {content}"}],
        "reward": reward,
        "info": {"task_id": task_id, "record_id": f"rec_{task_id}_{reward}"},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


# ---------------------------------------------------------------------------

def test_extract_script_imports():
    py_compile.compile(str(_SCRIPTS / "extract_high_reward_rollouts.py"), doraise=True)


def test_extract_threshold_filter(tmp_path):
    rollouts_dir = tmp_path / "rollouts"
    rollouts_dir.mkdir()
    records = [
        _mk_record("DFT-C", 0.7),
        _mk_record("DFT-C", 0.3),  # below threshold
        _mk_record("HFE", 0.8),
        _mk_record("HFE", 0.49),   # below threshold
    ]
    _write_jsonl(rollouts_dir / "r1.jsonl", records)
    out = tmp_path / "out.jsonl"

    result = ex.extract_records(
        rollouts_dir=rollouts_dir, output=out,
        threshold=0.5, max_per_task=10, verbose=False,
    )

    assert len(result) == 2
    rewards = sorted(r["reward"] for r in result)
    assert rewards == [0.7, 0.8]


def test_extract_per_task_cap(tmp_path):
    rollouts_dir = tmp_path / "rollouts"
    rollouts_dir.mkdir()
    # 5 above-threshold for DFT-C, 2 above-threshold for HFE
    records = (
        [_mk_record("DFT-C", 0.5 + i * 0.05) for i in range(5)]
        + [_mk_record("HFE", 0.6), _mk_record("HFE", 0.7)]
    )
    _write_jsonl(rollouts_dir / "r1.jsonl", records)
    out = tmp_path / "out.jsonl"

    result = ex.extract_records(
        rollouts_dir=rollouts_dir, output=out,
        threshold=0.5, max_per_task=2, verbose=False,
    )

    by_task: dict[str, int] = {}
    for r in result:
        by_task[r["task_id"]] = by_task.get(r["task_id"], 0) + 1
    assert by_task["DFT-C"] == 2  # capped from 5 → 2
    assert by_task["HFE"] == 2    # 2 ≤ cap → kept


def test_extract_skips_malformed(tmp_path):
    rollouts_dir = tmp_path / "rollouts"
    rollouts_dir.mkdir()
    records = [
        _mk_record("DFT-C", 0.7),
        {"reward": 0.9, "info": {"task_id": "DFT-C"}},  # missing prompt+completion
        {"prompt": [{"role": "user", "content": "x"}], "completion": [], "reward": 0.8},  # missing task_id
        {"messages": [{"role": "user", "content": "x"}], "task_id": "HFE"},  # missing reward
        _mk_record("HFE", 0.6),
    ]
    _write_jsonl(rollouts_dir / "r1.jsonl", records)
    out = tmp_path / "out.jsonl"

    result = ex.extract_records(
        rollouts_dir=rollouts_dir, output=out,
        threshold=0.5, max_per_task=10, verbose=False,
    )

    # Only 2 records pass: the valid DFT-C(0.7) and HFE(0.6)
    assert len(result) == 2
    task_ids = sorted(r["task_id"] for r in result)
    assert task_ids == ["DFT-C", "HFE"]


def test_extract_empty_set_hard_fails(tmp_path):
    rollouts_dir = tmp_path / "rollouts"
    rollouts_dir.mkdir()
    records = [_mk_record("DFT-C", 0.1), _mk_record("HFE", 0.2)]  # all below 0.5
    _write_jsonl(rollouts_dir / "r1.jsonl", records)
    out = tmp_path / "out.jsonl"

    with pytest.raises(SystemExit):
        ex.extract_records(
            rollouts_dir=rollouts_dir, output=out,
            threshold=0.5, max_per_task=10, verbose=False,
        )


def test_rft_config_exists_and_parses():
    path = _CONFIGS / "curie_rft_phase1.toml"
    assert path.is_file()
    cfg = tomllib.loads(path.read_text())
    assert cfg["model"]["ac"] is True
    assert cfg["wandb"]["project"] == "curie-rlm"


def test_rft_config_lr_lower_than_phase1():
    rft = tomllib.loads((_CONFIGS / "curie_rft_phase1.toml").read_text())
    phase1 = tomllib.loads((_CONFIGS / "curie_grpo_freeform.toml").read_text())
    assert rft["trainer"]["optim"]["lr"] < phase1["trainer"]["optim"]["lr"]
    assert rft["trainer"]["optim"]["lr"] == 5e-7
    assert phase1["trainer"]["optim"]["lr"] == 1e-6


def test_rft_config_loss_mask_assistant_only():
    cfg = tomllib.loads((_CONFIGS / "curie_rft_phase1.toml").read_text())
    assert cfg["trainer"]["loss_mask"]["strategy"] == "assistant_only"


def test_run_rft_script_executable():
    path = _SCRIPTS / "run_rft.sh"
    assert path.is_file()
    mode = path.stat().st_mode
    assert mode & stat.S_IXUSR


def test_runbook_doc_exists():
    path = _DOCS / "STAGE_5_5_CONTINGENCY.md"
    assert path.is_file()
    text = path.read_text()
    # Spot-check key sections present
    assert "## When to run Stage 5.5" in text
    assert "## What it does" in text
    assert "## How to run" in text
    assert "## How to verify it worked" in text
    assert "## Failure modes" in text
