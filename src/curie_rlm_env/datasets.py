"""CURIE data loader — reads JSON files from data/curie/data/data/{folder}/.

Folder mapping per data/curie/README.md verbatim:
  "Our data is organized into eight domain-specific subfolders: 'biogr', 'dft',
  'pdb', 'geo', 'mpve', 'qecc_65', 'hfd', and 'hfe'."

DFT-S/P/C distinction per data/curie/colabs/curie_run_eval.ipynb cell 16:
  field_name should be one of "structure_metadata", "dft_metadata", or "code".

Stage 3.5: load_curie_task now reads from data/curie/splits/{split}.jsonl when
present (built by scripts/build_splits.py). Backward compat: if splits files
do not exist and split == "test", falls back to loading all records as the
test set (legacy behavior with a warning).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import datasets

# task_id → (folder, dft_field_or_None)
TASK_MAP: dict[str, tuple[str, str | None]] = {
    "DFT-S": ("dft", "structure_metadata"),
    "DFT-P": ("dft", "dft_metadata"),
    "DFT-C": ("dft", "code"),
    "MPVE":  ("mpve", None),
    "BIOGR": ("biogr", None),
    "PDB":   ("pdb", None),
    "HFE":   ("hfe", None),
    "HFD":   ("hfd", None),
    "QECC_65": ("qecc_65", None),
    "GEO":   ("geo", None),
}

VALID_SPLITS = frozenset({"train", "val", "test"})

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data" / "curie" / "data" / "data"
_SPLITS_DIR = _PROJECT_ROOT / "data" / "curie" / "splits"


def _row_from_split_entry(
    entry: dict[str, Any], task_id: str, dft_field: str | None, folder_difficulty: dict[str, str]
) -> dict[str, Any]:
    """Build a Dataset row from a splits-file entry."""
    record_id = entry["record_id"]
    info: dict[str, Any] = {
        "record_id": record_id,
        "task_id": task_id,
        "difficulty": folder_difficulty[record_id],
        "dft_field": dft_field,
    }
    return {
        "prompt": [{"role": "user", "content": entry["input"]}],
        "answer": json.dumps(entry["ground_truth"]),
        "info": info,
    }


def _load_from_splits(task_id: str, split: str) -> datasets.Dataset:
    """Read records from data/curie/splits/{split}.jsonl filtered by task_id."""
    folder, dft_field = TASK_MAP[task_id]
    splits_file = _SPLITS_DIR / f"{split}.jsonl"
    difficulty_path = _DATA_ROOT / "difficulty_levels.json"
    folder_difficulty = json.loads(difficulty_path.read_text())[folder]

    rows: list[dict[str, Any]] = []
    for line in splits_file.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["task_id"] != task_id:
            continue
        rows.append(_row_from_split_entry(entry, task_id, dft_field, folder_difficulty))
    return datasets.Dataset.from_list(rows)


def _load_all_records(task_id: str) -> datasets.Dataset:
    """Legacy loader: all per-task records as one dataset (used for test fallback)."""
    folder, dft_field = TASK_MAP[task_id]
    folder_path = _DATA_ROOT / folder
    if not folder_path.is_dir():
        raise FileNotFoundError(
            f"Curie data folder missing at {folder_path}. "
            "Initialize the submodule: git submodule update --init --recursive "
            "and unzip data/curie/data/data.zip."
        )
    inputs_dir = folder_path / "inputs"
    gt_dir = folder_path / "ground_truth"
    if not inputs_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(
            f"Expected inputs/ and ground_truth/ subdirectories under {folder_path}"
        )

    difficulty_path = _DATA_ROOT / "difficulty_levels.json"
    folder_difficulty = json.loads(difficulty_path.read_text())[folder]

    rows: list[dict[str, Any]] = []
    for input_file in sorted(inputs_dir.glob("*.json")):
        record_id = input_file.stem
        gt_file = gt_dir / f"{record_id}.json"
        if not gt_file.is_file():
            raise FileNotFoundError(
                f"Missing ground truth for {record_id} at {gt_file}"
            )
        input_data = json.loads(input_file.read_text())
        gt_data = json.loads(gt_file.read_text())

        info: dict[str, Any] = {
            "record_id": record_id,
            "task_id": task_id,
            "difficulty": folder_difficulty[record_id],
            "dft_field": dft_field,
        }
        rows.append({
            "prompt": [{"role": "user", "content": input_data["text"]}],
            "answer": json.dumps(gt_data),
            "info": info,
        })
    return datasets.Dataset.from_list(rows)


def load_curie_task(task_id: str, split: str) -> datasets.Dataset:
    """Load CURIE benchmark records for the given task_id and split.

    Stage 3.5: split must be one of "train", "val", "test". When the
    corresponding splits/{split}.jsonl exists, returns the partitioned subset.
    Backward-compat fallback: if splits files don't exist and split == "test",
    returns the legacy all-records-as-test dataset (with warning).

    Raises:
        ValueError: task_id not in TASK_MAP, or split not in {"train","val","test"}.
        FileNotFoundError: submodule missing, or split is "train"/"val" without splits files.
    """
    if task_id not in TASK_MAP:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. Valid: {sorted(TASK_MAP)}"
        )
    if split not in VALID_SPLITS:
        raise ValueError(
            f'Invalid split={split!r}. Valid splits: {sorted(VALID_SPLITS)}.'
        )

    splits_file = _SPLITS_DIR / f"{split}.jsonl"
    if splits_file.exists():
        return _load_from_splits(task_id, split)

    if split == "test":
        warnings.warn(
            f"Splits file {splits_file} does not exist; falling back to "
            f"all-records-as-test (legacy behavior). "
            f"Run: uv run python scripts/build_splits.py",
            stacklevel=2,
        )
        return _load_all_records(task_id)

    raise FileNotFoundError(
        f"Splits file {splits_file} does not exist. "
        f"Run: uv run python scripts/build_splits.py before loading split={split!r}."
    )
