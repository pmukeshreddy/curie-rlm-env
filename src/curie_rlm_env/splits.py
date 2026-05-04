"""Stage 3.5 — Stratified per-task train/val/test split for CURIE.

Locked seed = 42. Why: any single integer is fine for reproducibility; 42 is
the project-wide convention (Hitchhiker's reference, no statistical meaning).
Document so future readers don't think it carries a secret optimization.

Per-task stratification (HFD has only 15 records — global random split would
randomly leave some tasks with zero val/test coverage).

DFT family unification: DFT-S, DFT-P, DFT-C all evaluate the SAME 74 underlying
record_ids in `data/curie/data/data/dft/` (only the model output field they
score against differs — `structure_metadata` / `dft_metadata` / `code` per
Curie cell 16). If we partitioned each independently, the same record_id could
land in train for DFT-S but test for DFT-P → train/test leakage. We detect
families by shared record_id sets and partition once per family, then assign
the same partition to every task in the family.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import json5

from .datasets import TASK_MAP

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data" / "curie" / "data" / "data"
SPLITS_DIR = _PROJECT_ROOT / "data" / "curie" / "splits"

LOCKED_SEED = 42
DEFAULT_RATIOS = (0.7, 0.15, 0.15)


def build_records_per_task(data_root: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Load all 10 tasks' records from data/curie/data/data/.

    Returns: {task_id: [{task_id, record_id, input, ground_truth}, ...]}
    """
    root = Path(data_root) if data_root is not None else _DATA_ROOT
    records_per_task: dict[str, list[dict[str, Any]]] = {}
    for task_id, (folder, _dft_field) in TASK_MAP.items():
        folder_path = root / folder
        inputs_dir = folder_path / "inputs"
        gt_dir = folder_path / "ground_truth"
        if not inputs_dir.is_dir() or not gt_dir.is_dir():
            raise FileNotFoundError(
                f"Expected inputs/ and ground_truth/ under {folder_path}"
            )
        records: list[dict[str, Any]] = []
        for input_file in sorted(inputs_dir.glob("*.json")):
            record_id = input_file.stem
            gt_file = gt_dir / f"{record_id}.json"
            if not gt_file.is_file():
                raise FileNotFoundError(
                    f"Missing ground_truth for {record_id} at {gt_file}"
                )
            # json5 for Curie GT files: some contain non-strict escapes / single
            # quotes (Curie cell 16 uses json5 for the same reason).
            input_data = json5.loads(input_file.read_text())
            gt_data = json5.loads(gt_file.read_text())
            records.append({
                "task_id": task_id,
                "record_id": record_id,
                "input": input_data["text"],
                "ground_truth": gt_data,
            })
        records_per_task[task_id] = records
    return records_per_task


def _detect_families(
    records_per_task: dict[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    """Group tasks by identical record_id sets (DFT-S/P/C share → one family)."""
    id_sets: dict[str, frozenset[str]] = {
        tid: frozenset(r["record_id"] for r in recs)
        for tid, recs in records_per_task.items()
    }
    seen: dict[frozenset[str], str] = {}
    families: dict[str, list[str]] = {}
    for tid in sorted(records_per_task.keys()):
        ids = id_sets[tid]
        if ids in seen:
            family_key = seen[ids]
            families[family_key].append(tid)
        else:
            family_key = tid
            seen[ids] = family_key
            families[family_key] = [tid]
    return families


def stratified_split(
    records_per_task: dict[str, list[dict[str, Any]]],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = LOCKED_SEED,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Per-task stratified split with DFT family unification.

    Returns: {task_id: {"train": [...], "val": [...], "test": [...]}}
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0; got {ratios} = {sum(ratios)}")

    families = _detect_families(records_per_task)
    rng = random.Random(seed)
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for family_key in sorted(families.keys()):
        task_ids = sorted(families[family_key])
        ref_recs = records_per_task[task_ids[0]]
        ref_ids = sorted(r["record_id"] for r in ref_recs)
        shuffled = list(ref_ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train_set = set(shuffled[:n_train])
        val_set = set(shuffled[n_train:n_train + n_val])
        test_set = set(shuffled[n_train + n_val:])
        for tid in task_ids:
            recs = records_per_task[tid]
            result[tid] = {
                "train": [r for r in recs if r["record_id"] in train_set],
                "val": [r for r in recs if r["record_id"] in val_set],
                "test": [r for r in recs if r["record_id"] in test_set],
            }
    return result


def write_splits(
    splits_per_task: dict[str, dict[str, list[dict[str, Any]]]],
    output_dir: Path | None = None,
) -> None:
    """Write {train,val,test}.jsonl files. Idempotent — overwrites existing."""
    out = Path(output_dir) if output_dir is not None else SPLITS_DIR
    out.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        path = out / f"{split_name}.jsonl"
        with path.open("w") as f:
            for tid in sorted(splits_per_task.keys()):
                for rec in splits_per_task[tid][split_name]:
                    # ensure_ascii=True escapes any surrogates as \uXXXX so the
                    # file is pure-ASCII JSON regardless of the source encoding.
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def build_and_write_splits(
    data_root: Path | None = None,
    output_dir: Path | None = None,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = LOCKED_SEED,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """End-to-end: load records → stratify → write JSONL files. Returns splits dict."""
    records_per_task = build_records_per_task(data_root=data_root)
    splits = stratified_split(records_per_task, ratios=ratios, seed=seed)
    write_splits(splits, output_dir=output_dir)
    return splits
