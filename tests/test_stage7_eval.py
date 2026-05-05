"""Stage 7 — multi-checkpoint eval + report tests.

Static-structure tests only. Live multi-checkpoint eval requires a GPU pod
with multiple served checkpoints and is run separately on the cluster.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"

# Make scripts importable as modules
sys.path.insert(0, str(_SCRIPTS))


def _mk_per_task(reward: float, n: int = 10) -> dict:
    return {"mean_reward": reward, "mean_headline": reward, "std": 0.05, "n": n}


def _mk_per_checkpoint_dict(name: str, base_reward: float = 0.5, n: int = 10) -> dict:
    """Build a synthetic per-checkpoint result dict (test fixture, not training data)."""
    per_task = {
        t: _mk_per_task(base_reward, n)
        for t in ("DFT-S", "DFT-P", "DFT-C", "MPVE", "BIOGR", "PDB",
                  "HFE", "HFD", "QECC_65", "GEO")
    }
    return {
        "model": name,
        "split": "test",
        "n_problems": n * 10,
        "per_task": per_task,
        "overall": {"mean_reward": base_reward, "mean_headline": base_reward},
        "rollouts": [],
    }


# ---------------------------------------------------------------------------
# run_full_eval
# ---------------------------------------------------------------------------

def test_run_full_eval_imports():
    import run_full_eval as rfe
    assert callable(rfe.aggregate_full_eval)
    assert callable(rfe.write_full_eval)


def test_run_full_eval_writes_expected_schema(tmp_path):
    import run_full_eval as rfe
    per_ckpt = {"baseline": _mk_per_checkpoint_dict("baseline", 0.4, 10)}
    metadata = {"model": "Qwen3.5-7B", "test_split_n": 100,
                "judge_model": "gemini-2.5-pro", "splits_seed": 42,
                "split": "test", "timestamp": "2026-05-04T00:00:00Z"}
    out = tmp_path / "full_eval.json"
    aggregated = rfe.aggregate_full_eval(per_ckpt, ablation=None, metadata=metadata)
    rfe.write_full_eval(aggregated, out)
    parsed = json.loads(out.read_text())
    assert parsed["checkpoints"] == ["baseline"]
    assert "per_checkpoint" in parsed
    assert "ablations" in parsed
    assert "metadata" in parsed
    assert parsed["metadata"]["splits_seed"] == 42


def test_run_full_eval_handles_zero_reward_rollouts():
    import run_full_eval as rfe
    per_ckpt = {"baseline": _mk_per_checkpoint_dict("baseline", 0.0, 5)}
    metadata = {"model": "X", "test_split_n": 50, "judge_model": "Y",
                "splits_seed": 42, "split": "test", "timestamp": "now"}
    aggregated = rfe.aggregate_full_eval(per_ckpt, ablation=None, metadata=metadata)
    assert aggregated["per_checkpoint"]["baseline"]["overall"]["mean_reward"] == 0.0
    # All 10 tasks present
    assert len(aggregated["per_checkpoint"]["baseline"]["per_task"]) == 10


# ---------------------------------------------------------------------------
# run_rlm_ablation
# ---------------------------------------------------------------------------

def test_run_rlm_ablation_imports():
    import run_rlm_ablation as rra
    assert callable(rra.compute_delta)
    assert callable(rra.aggregate_ablation)
    assert callable(rra.write_ablation)


def test_run_rlm_ablation_two_modes(tmp_path):
    import run_rlm_ablation as rra
    on = _mk_per_checkpoint_dict("phase3", 0.7, 10)
    off = _mk_per_checkpoint_dict("phase3", 0.4, 10)
    aggregated = rra.aggregate_ablation("phase3", on, off)
    out = tmp_path / "ablation.json"
    rra.write_ablation(aggregated, out)
    parsed = json.loads(out.read_text())
    assert parsed["checkpoint"] == "phase3"
    assert "mode_A_rlm_on" in parsed
    assert "mode_B_rlm_off" in parsed
    assert "delta_per_task" in parsed
    # Delta should be ~0.3 for every task (0.7 - 0.4)
    for task, d in parsed["delta_per_task"].items():
        assert abs(d - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

def _mk_eval_json(checkpoints: list[str], rewards: list[float], n: int = 10) -> dict:
    per_ckpt = {
        name: _mk_per_checkpoint_dict(name, r, n)
        for name, r in zip(checkpoints, rewards)
    }
    return {
        "checkpoints": checkpoints,
        "per_checkpoint": per_ckpt,
        "ablations": {},
        "metadata": {
            "model": "Qwen3.5-7B",
            "test_split_n": n * 10,
            "judge_model": "gemini-2.5-pro",
            "splits_seed": 42,
            "split": "test",
            "timestamp": "2026-05-04T00:00:00Z",
        },
    }


def test_generate_report_imports():
    import generate_report as gr
    assert callable(gr.render_report)


def test_generate_report_renders_table_correctly():
    import generate_report as gr
    eval_data = _mk_eval_json(["baseline", "phase3"], [0.4, 0.7], n=10)
    md = gr.render_report(eval_data, ablation_data=None)
    # Every task name appears in the markdown
    for task in ("DFT-S", "DFT-P", "DFT-C", "MPVE", "BIOGR",
                 "PDB", "HFE", "HFD", "QECC_65", "GEO"):
        assert task in md, f"task {task} missing from rendered report"
    # Both checkpoint columns present
    assert "baseline" in md
    assert "phase3" in md


def test_generate_report_flags_small_n():
    import generate_report as gr
    eval_data = _mk_eval_json(["baseline", "phase3"], [0.4, 0.7], n=10)
    # Override one task to small-N to check flag
    eval_data["per_checkpoint"]["phase3"]["per_task"]["HFD"]["n"] = 3
    md = gr.render_report(eval_data, ablation_data=None)
    assert "small-N" in md
    # The HFD row should be flagged (small-N flag near HFD)
    hfd_lines = [line for line in md.splitlines() if "HFD" in line]
    assert any("small-N" in line for line in hfd_lines), "HFD row missing small-N flag"


def test_generate_report_no_hardcoded_numbers():
    """Template body must not embed result-shaped decimal literals (e.g. 0.45, 0.7)."""
    src = (_SCRIPTS / "generate_report.py").read_text()
    # Strip comments
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    decimals = re.findall(r"\b\d+\.\d+\b", code)
    # Allowlist (post Stage 7 hotfix — 0.0 default fallbacks REMOVED):
    #   0.05 — DELTA_HIGHLIGHT_THRESHOLD constant (design knob, not result data)
    #   1.5, 2.5 — judge model version refs (gemini-1.5-pro-latest, gemini-2.5-pro)
    #   6.5 — "Stage 6.5 memo" ref in caveats text
    allowed = {"0.05", "1.5", "2.5", "6.5"}
    leaked = [d for d in decimals if d not in allowed]
    assert not leaked, (
        f"Possibly hardcoded result-decimals in generate_report.py: {leaked}. "
        f"All result numbers must come from JSON via interpolation."
    )


def test_generate_report_hard_fails_on_missing_required_key():
    """Stage 7 hotfix: missing mandatory eval-JSON key → KeyError with full path."""
    import generate_report as gr
    eval_data = _mk_eval_json(["baseline"], [0.5], n=10)
    # Sabotage: drop mean_reward from DFT-S
    del eval_data["per_checkpoint"]["baseline"]["per_task"]["DFT-S"]["mean_reward"]
    with pytest.raises(KeyError) as exc_info:
        gr.render_report(eval_data, ablation_data=None)
    msg = str(exc_info.value)
    # Full nested path must appear in the error message
    assert "per_checkpoint" in msg
    assert "baseline" in msg
    assert "DFT-S" in msg
    assert "mean_reward" in msg


def test_generate_report_renders_na_for_missing_std():
    """Stage 7 hotfix: std is optional (e.g. n=1). Missing → 'n/a' in table cell."""
    import generate_report as gr
    eval_data = _mk_eval_json(["baseline"], [0.5], n=10)
    # Sabotage: remove std from one task (legitimate for n=1)
    del eval_data["per_checkpoint"]["baseline"]["per_task"]["HFD"]["std"]
    md = gr.render_report(eval_data, ablation_data=None)
    # The HFD row should render std as "n/a"
    hfd_lines = [line for line in md.splitlines() if line.startswith("| HFD ")]
    assert hfd_lines, "HFD row missing from rendered table"
    assert any("n/a" in line for line in hfd_lines), (
        f"HFD row should render std as 'n/a' when std field missing; got: {hfd_lines}"
    )


def test_generate_report_caveats_section_present():
    import generate_report as gr
    eval_data = _mk_eval_json(["baseline"], [0.5], n=10)
    md = gr.render_report(eval_data, ablation_data=None)
    assert "## Honest caveats" in md
    # 5 caveat bullets per spec
    caveats = [
        "Test-split sizes per task",
        "Judge model deviation",
        "PDB scorer: code-exec branch dropped",
        "Free-form: BERTScore",
        "Sub-LM tokens not gradient-trained",
    ]
    for c in caveats:
        assert c in md, f"caveat missing: {c}"


def test_generate_report_config_snapshot_includes_all_yamls():
    import generate_report as gr
    eval_data = _mk_eval_json(["baseline"], [0.5], n=10)
    md = gr.render_report(eval_data, ablation_data=None)
    # All 4 yaml configs
    for yaml_name in ("safeguards.yaml", "judge.yaml",
                      "rubric_dispatcher.yaml", "curie_tasks.yaml"):
        assert f"config/{yaml_name}" in md, f"{yaml_name} not embedded in config snapshot"
    # All 3 continual replay RL toml configs
    for toml_name in ("curie_grpo_continual_phase1.toml", "curie_grpo_continual_phase2.toml",
                      "curie_grpo_continual_phase3.toml"):
        assert f"configs/{toml_name}" in md, f"{toml_name} not embedded in config snapshot"


def test_generate_report_per_task_delta_sorted():
    import generate_report as gr
    eval_data = _mk_eval_json(["baseline", "phase3"], [0.5, 0.5], n=10)
    # Make GEO have biggest improvement, MPVE the next, etc.
    eval_data["per_checkpoint"]["phase3"]["per_task"]["GEO"]["mean_reward"] = 0.95
    eval_data["per_checkpoint"]["phase3"]["per_task"]["MPVE"]["mean_reward"] = 0.85
    md = gr.render_report(eval_data, ablation_data=None)
    # Find the delta section, parse task order
    delta_section = md.split("Per-task delta:")[1].split("##")[0]
    rows = [
        line for line in delta_section.splitlines()
        if line.startswith("| ") and "Task" not in line and "---" not in line
    ]
    # First non-zero delta should be GEO (largest), then MPVE
    assert "GEO" in rows[0], f"largest-delta row should be GEO; got: {rows[0]}"
    assert "MPVE" in rows[1], f"second-largest-delta row should be MPVE; got: {rows[1]}"
