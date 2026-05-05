"""Continual replay dataset tests.

Quote from src/curie_rlm_env/continual.py:
"Phase 3: 60% geometric/structural current tasks + 20% Phase 1 replay"
"+ 20% Phase 2 replay."

The records below are unit-test fixtures for replay mechanics, not training
data.
"""
from __future__ import annotations

from collections import Counter

import datasets
import pytest

from curie_rlm_env import continual as cont


def _mk_dataset(task_id: str, n: int = 1) -> datasets.Dataset:
    rows = [
        {
            "prompt": [{"role": "user", "content": f"prompt {task_id} {i}"}],
            "answer": f"answer {task_id} {i}",
            "info": {
                "task_id": task_id,
                "record_id": f"{task_id}-{i}",
                "difficulty": "unit",
                "dft_field": None,
            },
        }
        for i in range(n)
    ]
    return datasets.Dataset.from_list(rows)


def _all_task_datasets(n: int = 1) -> dict[str, datasets.Dataset]:
    task_ids = set(cont.FREEFORM_TASKS + cont.RETRIEVAL_TASKS + cont.GEOMETRIC_TASKS)
    return {task_id: _mk_dataset(task_id, n=n) for task_id in task_ids}


def _signature(ds: datasets.Dataset) -> list[tuple[str, str, str]]:
    return [
        (
            row["info"]["continual_component"],
            row["info"]["task_id"],
            row["info"]["record_id"],
        )
        for row in ds
    ]


def _metadata_by_component(ds: datasets.Dataset) -> dict[str, dict]:
    return {
        row["info"]["continual_component"]: row["info"]
        for row in ds
    }


def test_continual_phase_definitions_are_locked():
    assert cont.FREEFORM_TASKS == ["DFT-C", "HFE", "HFD", "QECC_65", "GEO"]
    assert cont.RETRIEVAL_TASKS == ["DFT-S", "DFT-P", "MPVE"]
    assert cont.GEOMETRIC_TASKS == ["BIOGR", "PDB"]
    assert cont.CONTINUAL_PHASES[2]["mixture"] == {"current": 0.70, "phase1": 0.30}
    assert cont.CONTINUAL_PHASES[3]["mixture"] == {"current": 0.60, "phase1": 0.20, "phase2": 0.20}


def test_phase2_mixture_counts_exact_70_30():
    mixed = cont.mix_task_datasets(2, _all_task_datasets(), seed=42)
    counts = Counter(row["info"]["continual_component"] for row in mixed)
    assert counts == {"current": 7, "phase1": 3}


def test_phase3_mixture_counts_exact_60_20_20():
    mixed = cont.mix_task_datasets(3, _all_task_datasets(), seed=42)
    counts = Counter(row["info"]["continual_component"] for row in mixed)
    assert counts == {"current": 3, "phase1": 1, "phase2": 1}


def test_replay_sampling_is_deterministic_for_same_seed():
    task_datasets = _all_task_datasets(n=2)
    first = cont.mix_task_datasets(3, task_datasets, seed=42)
    second = cont.mix_task_datasets(3, task_datasets, seed=42)
    assert _signature(first) == _signature(second)


def test_replay_sampling_changes_with_seed_but_keeps_counts():
    task_datasets = _all_task_datasets(n=2)
    first = cont.mix_task_datasets(2, task_datasets, seed=42)
    second = cont.mix_task_datasets(2, task_datasets, seed=43)
    assert _signature(first) != _signature(second)
    assert Counter(row["info"]["continual_component"] for row in first) == Counter(
        row["info"]["continual_component"] for row in second
    )


def test_replay_rows_are_annotated_with_phase_and_weight():
    mixed = cont.mix_task_datasets(2, _all_task_datasets(), seed=42)
    for row in mixed:
        info = row["info"]
        assert info["continual_phase"] == 2
        assert info["continual_component"] in {"current", "phase1"}
        assert info["continual_component_weight"] in {0.70, 0.30}


def test_replay_rows_include_stream_role_and_replay_source():
    mixed = cont.mix_task_datasets(2, _all_task_datasets(), seed=42)
    metadata = _metadata_by_component(mixed)
    assert metadata["current"]["stream_role"] == "current"
    assert metadata["current"]["replay_source"] == "current"
    assert metadata["phase1"]["stream_role"] == "replay"
    assert metadata["phase1"]["replay_source"] == "phase1"


def test_load_continual_phase_dataset_uses_requested_split(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_loader(task_id: str, split: str) -> datasets.Dataset:
        calls.append((task_id, split))
        return _mk_dataset(task_id)

    monkeypatch.setattr(cont, "load_curie_task", fake_loader)
    mixed = cont.load_continual_phase_dataset(2, split="train", seed=42)
    loaded_tasks = {task_id for task_id, split in calls}
    expected_tasks = set(cont.FREEFORM_TASKS + cont.RETRIEVAL_TASKS)
    assert loaded_tasks == expected_tasks
    assert {split for task_id, split in calls} == {"train"}
    assert len(mixed) == 10


def test_invalid_phase_raises_value_error():
    with pytest.raises(ValueError):
        cont.mix_task_datasets(4, _all_task_datasets(), seed=42)


def test_missing_task_dataset_raises_value_error():
    task_datasets = _all_task_datasets()
    del task_datasets["DFT-S"]
    with pytest.raises(ValueError):
        cont.mix_task_datasets(2, task_datasets, seed=42)
