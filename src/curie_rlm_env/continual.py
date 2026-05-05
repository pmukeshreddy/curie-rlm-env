"""Continual CURIE replay dataset utilities.

Repository quotes anchoring this module:
- src/curie_rlm_env/datasets.py: "def load_curie_task(task_id: str, split: str) -> datasets.Dataset:"
- config/rubric_dispatcher.yaml: "ids: [\"DFT-C\", \"HFE\", \"HFD\", \"QECC_65\", \"GEO\"]"
- config/rubric_dispatcher.yaml: "ids: [\"DFT-S\", \"DFT-P\", \"MPVE\"]"
- config/rubric_dispatcher.yaml: "ids: [\"BIOGR\"]" and "ids: [\"PDB\"]"

Stage 5 now trains through continual replay phases:
- Phase 1: 100% free-form current tasks.
- Phase 2: 70% retrieval current tasks + 30% Phase 1 replay.
- Phase 3: 60% geometric/structural current tasks + 20% Phase 1 replay
  + 20% Phase 2 replay.
"""
from __future__ import annotations

import hashlib
import random
from fractions import Fraction
from math import ceil, lcm
from typing import Any, Mapping

import datasets

from .datasets import load_curie_task

FREEFORM_TASKS = ["DFT-C", "HFE", "HFD", "QECC_65", "GEO"]
RETRIEVAL_TASKS = ["DFT-S", "DFT-P", "MPVE"]
GEOMETRIC_TASKS = ["BIOGR", "PDB"]

CONTINUAL_SEED = 42

CONTINUAL_PHASES: dict[int, dict[str, Any]] = {
    1: {
        "current": FREEFORM_TASKS,
        "mixture": {"current": 1.0},
    },
    2: {
        "current": RETRIEVAL_TASKS,
        "replay": {"phase1": FREEFORM_TASKS},
        "mixture": {"current": 0.70, "phase1": 0.30},
    },
    3: {
        "current": GEOMETRIC_TASKS,
        "replay": {
            "phase1": FREEFORM_TASKS,
            "phase2": RETRIEVAL_TASKS,
        },
        "mixture": {"current": 0.60, "phase1": 0.20, "phase2": 0.20},
    },
}


def validate_continual_phase(continual_phase: int) -> None:
    """Validate a continual phase id."""
    if continual_phase not in CONTINUAL_PHASES:
        raise ValueError(
            f"Unknown continual phase: {continual_phase!r}. Valid phases: {sorted(CONTINUAL_PHASES)}."
        )


def component_tasks_for_continual_phase(continual_phase: int) -> dict[str, list[str]]:
    """Return current/replay component task lists for a continual phase."""
    validate_continual_phase(continual_phase)
    definition = CONTINUAL_PHASES[continual_phase]
    components: dict[str, list[str]] = {"current": list(definition["current"])}
    for name, task_ids in definition.get("replay", {}).items():
        components[name] = list(task_ids)
    return components


def mixture_for_continual_phase(continual_phase: int) -> dict[str, float]:
    """Return replay mixture weights for a continual phase."""
    validate_continual_phase(continual_phase)
    return dict(CONTINUAL_PHASES[continual_phase]["mixture"])


def _stable_seed(seed: int, continual_phase: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{continual_phase}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _ratio_fractions(mixture: Mapping[str, float]) -> dict[str, Fraction]:
    fractions = {
        name: Fraction(str(weight)).limit_denominator(100)
        for name, weight in mixture.items()
    }
    total = sum(fractions.values(), Fraction(0, 1))
    if total != Fraction(1, 1):
        raise ValueError(f"Continual replay mixture weights must sum to 1.0; got {total}.")
    return fractions


def _quota_by_component(mixture: Mapping[str, float], current_size: int) -> dict[str, int]:
    if current_size <= 0:
        raise ValueError("Continual replay current component must contain at least one record.")

    fractions = _ratio_fractions(mixture)
    denominator = lcm(*(fraction.denominator for fraction in fractions.values()))
    units = {
        name: int(fraction * denominator)
        for name, fraction in fractions.items()
    }
    current_units = units["current"]
    cycles = ceil(current_size / current_units)
    return {
        name: count * cycles
        for name, count in units.items()
    }


def _row_from_dataset(
    dataset: datasets.Dataset,
    index: int,
    task_id: str,
    component: str,
    continual_phase: int,
    component_weight: float,
) -> dict[str, Any]:
    row = dict(dataset[index])
    info = row.get("info")
    if not isinstance(info, dict):
        raise ValueError(f"CURIE row for task {task_id} is missing dict info metadata.")
    if info.get("task_id") != task_id:
        raise ValueError(
            f"CURIE row task mismatch: expected {task_id}, got {info.get('task_id')!r}."
        )
    if "record_id" not in info:
        raise ValueError(f"CURIE row for task {task_id} is missing info.record_id.")

    annotated_info = dict(info)
    annotated_info["continual_phase"] = continual_phase
    annotated_info["continual_component"] = component
    annotated_info["continual_component_weight"] = component_weight
    annotated_info["stream_role"] = "current" if component == "current" else "replay"
    annotated_info["replay_source"] = component
    row["info"] = annotated_info
    return row


def _sample_rows(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Continual replay component cannot be sampled because it is empty.")
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    while len(sampled) < count:
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for index in indices:
            sampled.append(dict(rows[index]))
            if len(sampled) == count:
                break
    return sampled


def mix_task_datasets(
    continual_phase: int,
    task_datasets: Mapping[str, datasets.Dataset],
    seed: int = CONTINUAL_SEED,
) -> datasets.Dataset:
    """Build the deterministic replay dataset for one continual training phase.

    The epoch length is derived from the current-task component and exact replay
    ratios. Replay samples are repeated real CURIE rows when the requested
    quota exceeds the replay source size.
    """
    validate_continual_phase(continual_phase)
    components = component_tasks_for_continual_phase(continual_phase)
    mixture = mixture_for_continual_phase(continual_phase)

    component_rows: dict[str, list[dict[str, Any]]] = {}
    for component, task_ids in components.items():
        if component not in mixture:
            raise ValueError(
                f"Continual component {component!r} is missing from continual phase {continual_phase} mixture."
            )
        rows: list[dict[str, Any]] = []
        for task_id in task_ids:
            if task_id not in task_datasets:
                raise ValueError(f"Task dataset missing for continual task {task_id}.")
            task_dataset = task_datasets[task_id]
            for index in range(len(task_dataset)):
                rows.append(
                    _row_from_dataset(
                        task_dataset,
                        index,
                        task_id,
                        component,
                        continual_phase,
                        mixture[component],
                    )
                )
        component_rows[component] = rows

    quotas = _quota_by_component(mixture, len(component_rows["current"]))
    mixed_rows: list[dict[str, Any]] = []
    for component in components:
        mixed_rows.extend(
            _sample_rows(
                component_rows[component],
                quotas[component],
                _stable_seed(seed, continual_phase, component),
            )
        )

    rng = random.Random(_stable_seed(seed, continual_phase, "interleave"))
    rng.shuffle(mixed_rows)
    return datasets.Dataset.from_list(mixed_rows)


def load_continual_phase_dataset(
    continual_phase: int,
    split: str = "train",
    seed: int = CONTINUAL_SEED,
) -> datasets.Dataset:
    """Load and mix the CURIE dataset for one continual training phase."""
    validate_continual_phase(continual_phase)
    task_ids = {
        task_id
        for task_list in component_tasks_for_continual_phase(continual_phase).values()
        for task_id in task_list
    }
    task_datasets = {
        task_id: load_curie_task(task_id, split)
        for task_id in sorted(task_ids)
    }
    return mix_task_datasets(continual_phase, task_datasets, seed=seed)
