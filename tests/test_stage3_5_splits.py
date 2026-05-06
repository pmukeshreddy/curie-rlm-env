"""Stage 3.5 — Stratified split tests.

Tests do NOT touch real data/curie/splits/ files; they use tmp_path or
synthetic record dicts as inputs to the pure functions. The actual splits
files are built by scripts/build_splits.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from curie_rlm_env.splits import (
    DEFAULT_RATIOS,
    LOCKED_SEED,
    build_records_per_task,
    stratified_split,
    write_splits,
)
from curie_rlm_env.baseline_eval import TASK_IDS


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data" / "curie" / "data" / "data"


def _mk_records(task_id: str, ids: list[str]) -> list[dict]:
    """Build synthetic per-task record list (test-fixture inputs to pure split fn)."""
    return [
        {"task_id": task_id, "record_id": rid, "input": f"x_{rid}", "ground_truth": {"r": rid}}
        for rid in ids
    ]


def test_stratified_no_global_split():
    # Each task gets its own train/val/test independent of others' sizes.
    records_per_task = {
        "BIGTASK": _mk_records("BIGTASK", [f"b{i}" for i in range(100)]),
        "TINYTASK": _mk_records("TINYTASK", [f"t{i}" for i in range(15)]),
    }
    splits = stratified_split(records_per_task)
    # Per-task sizes follow the ratio independently
    assert len(splits["BIGTASK"]["train"]) == 70
    assert len(splits["BIGTASK"]["val"]) == 15
    assert len(splits["BIGTASK"]["test"]) == 15
    assert len(splits["TINYTASK"]["train"]) == 10  # int(15*0.7)
    assert len(splits["TINYTASK"]["val"]) == 2     # int(15*0.15)
    assert len(splits["TINYTASK"]["test"]) == 3    # 15 - 10 - 2


def test_seed_42_deterministic(tmp_path):
    records_per_task = {
        "A": _mk_records("A", [f"a{i}" for i in range(20)]),
        "B": _mk_records("B", [f"b{i}" for i in range(30)]),
    }
    splits1 = stratified_split(records_per_task, seed=42)
    splits2 = stratified_split(records_per_task, seed=42)
    assert splits1 == splits2

    # File-level determinism
    write_splits(splits1, tmp_path / "first")
    write_splits(splits2, tmp_path / "second")
    for split_name in ("train", "val", "test"):
        b1 = (tmp_path / "first" / f"{split_name}.jsonl").read_bytes()
        b2 = (tmp_path / "second" / f"{split_name}.jsonl").read_bytes()
        assert b1 == b2, f"{split_name}.jsonl differs across two seed=42 runs"


def test_all_10_tasks_have_train_val_test():
    # Real Curie data: every task must have nonzero train+val+test
    records_per_task = build_records_per_task(_DATA_ROOT)
    splits = stratified_split(records_per_task)
    for task_id in TASK_IDS:
        assert len(splits[task_id]["train"]) > 0, f"{task_id} train empty"
        assert len(splits[task_id]["val"]) > 0, f"{task_id} val empty"
        assert len(splits[task_id]["test"]) > 0, f"{task_id} test empty"


def test_no_record_in_two_splits():
    # Within each task: train, val, test record_id sets are pairwise disjoint
    records_per_task = build_records_per_task(_DATA_ROOT)
    splits = stratified_split(records_per_task)
    for task_id in TASK_IDS:
        train_ids = {r["record_id"] for r in splits[task_id]["train"]}
        val_ids = {r["record_id"] for r in splits[task_id]["val"]}
        test_ids = {r["record_id"] for r in splits[task_id]["test"]}
        assert train_ids.isdisjoint(val_ids), f"{task_id} train∩val nonempty"
        assert train_ids.isdisjoint(test_ids), f"{task_id} train∩test nonempty"
        assert val_ids.isdisjoint(test_ids), f"{task_id} val∩test nonempty"


def test_dft_family_unification():
    # DFT-S/P/C share record_ids → must share partition (no leakage)
    records_per_task = build_records_per_task(_DATA_ROOT)
    splits = stratified_split(records_per_task)
    for split_name in ("train", "val", "test"):
        ids_s = {r["record_id"] for r in splits["DFT-S"][split_name]}
        ids_p = {r["record_id"] for r in splits["DFT-P"][split_name]}
        ids_c = {r["record_id"] for r in splits["DFT-C"][split_name]}
        assert ids_s == ids_p == ids_c, (
            f"DFT-S/P/C must share {split_name} partition; got "
            f"S={len(ids_s)} P={len(ids_p)} C={len(ids_c)}"
        )


def test_splits_files_have_expected_schema(tmp_path):
    records_per_task = {
        "A": _mk_records("A", [f"a{i}" for i in range(10)]),
    }
    splits = stratified_split(records_per_task)
    write_splits(splits, tmp_path)
    required = {"task_id", "record_id", "input", "ground_truth"}
    found_lines = 0
    for split_name in ("train", "val", "test"):
        path = tmp_path / f"{split_name}.jsonl"
        assert path.is_file()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            assert required.issubset(entry.keys()), (
                f"{split_name}.jsonl entry missing keys: "
                f"required={required}, got={set(entry.keys())}"
            )
            found_lines += 1
    assert found_lines == 10  # all 10 records appear exactly once across the 3 files


def test_locked_seed_is_42():
    # Document the locked seed value in the public constant
    assert LOCKED_SEED == 42


def test_default_ratios_sum_to_one():
    assert abs(sum(DEFAULT_RATIOS) - 1.0) < 1e-9
    assert DEFAULT_RATIOS == (0.7, 0.15, 0.15)


# ---------------------------------------------------------------------------
# Strict failure semantics for load_curie_task — no all-records-as-test fallback.
# ---------------------------------------------------------------------------


def test_load_curie_task_test_split_hard_fails_when_splits_file_missing(monkeypatch, tmp_path):
    """Strict: missing data/curie/splits/test.jsonl raises FileNotFoundError, even for split='test'."""
    from curie_rlm_env import datasets as ds

    monkeypatch.setattr(ds, "_SPLITS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        ds.load_curie_task("DFT-C", "test")
    assert "test.jsonl" in str(exc_info.value)
    assert "build_splits.py" in str(exc_info.value)


def test_load_curie_task_train_split_hard_fails_when_missing(monkeypatch, tmp_path):
    from curie_rlm_env import datasets as ds

    monkeypatch.setattr(ds, "_SPLITS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        ds.load_curie_task("DFT-C", "train")


def test_load_curie_task_val_split_hard_fails_when_missing(monkeypatch, tmp_path):
    from curie_rlm_env import datasets as ds

    monkeypatch.setattr(ds, "_SPLITS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        ds.load_curie_task("DFT-C", "val")


def test_datasets_module_does_not_export_legacy_loader():
    """Strict: the all-records-as-test loader was deleted, not just unused."""
    from curie_rlm_env import datasets as ds

    assert not hasattr(ds, "_load_all_records"), (
        "datasets._load_all_records must be removed — no all-records-as-test fallback path"
    )


def test_datasets_module_does_not_warn_about_legacy_fallback():
    """Strict: no warnings.warn() in datasets.py — fallback path was removed entirely."""
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "curie_rlm_env" / "datasets.py"
    ).read_text()
    assert "warnings.warn" not in src, (
        "datasets.py must not call warnings.warn — strict failure replaces the legacy warning"
    )
    for phrase in ("legacy fallback", "Backward-compat fallback", "Backward compat fallback"):
        assert phrase not in src, f"datasets.py still mentions {phrase!r}"


# ---------------------------------------------------------------------------
# RLM-shaped dataset rows: long input → info["context"], short prompt
# ---------------------------------------------------------------------------


def _mk_minimal_split_entry(task_id: str = "DFT-C") -> dict:
    return {
        "task_id": task_id,
        "record_id": "rec_dummy",
        "input": "x" * 200_000,  # ~50k tokens of long input
        "ground_truth": {"answer": "stub"},
    }


def test_row_puts_long_input_in_context_not_prompt():
    """RLM contract: long input lives in info['context'], NOT in prompt[0].content.

    Stuffing the input into the user prompt would defeat RLM (the root model
    would see all 50k tokens at once). RLMEnv writes info['context'] to
    <rlm_fs_root>/context.txt inside the sandbox so the model reads it via the
    REPL.
    """
    from curie_rlm_env.datasets import _row_from_split_entry

    entry = _mk_minimal_split_entry("DFT-C")
    folder_difficulty = {entry["record_id"]: "medium"}
    row = _row_from_split_entry(entry, "DFT-C", "code", folder_difficulty)

    user_text = row["prompt"][0]["content"]
    assert len(user_text) < 4_000, (
        f"task prompt should stay short for RLM; got {len(user_text)} chars"
    )
    assert "x" * 1000 not in user_text, "long input must NOT be inlined in the prompt"

    info = row["info"]
    assert info["context"] == entry["input"], (
        "info['context'] must carry the full long input verbatim "
        "(RLMEnv writes it to <rlm_fs_root>/context.txt)"
    )


def test_row_prompt_mentions_context_file_and_repl_workflow():
    """The short prompt must tell the model where the input is and how to read it."""
    from curie_rlm_env.datasets import _row_from_split_entry

    entry = _mk_minimal_split_entry("HFE")
    folder_difficulty = {entry["record_id"]: "easy"}
    row = _row_from_split_entry(entry, "HFE", None, folder_difficulty)

    user_text = row["prompt"][0]["content"]
    assert "context.txt" in user_text
    assert "submit_answer" in user_text
    assert "answer" in user_text  # tells model how to set the final answer
    assert "ready" in user_text   # tells model how to signal completion


def test_row_prompt_has_per_task_answer_format_hint():
    """Each task family gets a format-specific hint matching the rubric."""
    from curie_rlm_env.datasets import _row_from_split_entry

    expectations = {
        "DFT-S": "JSON list",
        "BIOGR": '"W"',
        "PDB": "FASTA",
        "HFE": "Free-form",
    }
    for task_id, must_contain in expectations.items():
        entry = _mk_minimal_split_entry(task_id)
        folder_difficulty = {entry["record_id"]: "medium"}
        row = _row_from_split_entry(entry, task_id, None, folder_difficulty)
        text = row["prompt"][0]["content"]
        assert must_contain in text, (
            f"prompt for {task_id} should mention {must_contain!r}; got:\n{text}"
        )
