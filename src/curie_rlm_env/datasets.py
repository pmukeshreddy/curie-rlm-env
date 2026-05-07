"""CURIE data loader — reads JSON files from data/curie/data/data/{folder}/.

Folder mapping per data/curie/README.md verbatim:
  "Our data is organized into eight domain-specific subfolders: 'biogr', 'dft',
  'pdb', 'geo', 'mpve', 'qecc_65', 'hfd', and 'hfe'."

DFT-S/P/C distinction per data/curie/colabs/curie_run_eval.ipynb cell 16:
  field_name should be one of "structure_metadata", "dft_metadata", or "code".

Stage 3.5: load_curie_task reads exclusively from data/curie/splits/{split}.jsonl
(built by scripts/build_splits.py). Every split — including "test" — hard-fails
with FileNotFoundError when the splits file is missing. There is no
all-records-as-test path.

RLM input shape (the whole point of this project):
  CURIE inputs are 15k+ words (~20-40k+ tokens). Stuffing them into the root
  prompt defeats Recursive Language Models — the root model would never have
  the chance to recurse, and a single rollout would blow past Qwen3-8B's 40,960
  context. Instead, each row's `info["context"]` carries the full long input;
  RLMEnv writes it to ``<rlm_fs_root>/context.txt`` inside the sandbox before
  the rollout starts (see verifiers/.../rlm_env.py:_write_builtin_context). The
  REPL's cwd is rlm_fs_root, so the model reads via `open("context.txt").read()`
  and uses `call_python_repl` + `llm_batch` to navigate/summarize chunks. The
  visible `prompt` shrinks to a short task instruction (~200 tokens) — well
  within any reasonable context window.
"""
from __future__ import annotations

import json
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

# Per-task answer-format hint appended to the prompt. Matches the rubric:
#   - LLMSim retrieval (DFT-S/P, MPVE) → JSON list
#   - IoU geometric (BIOGR)             → JSON dict {W,S,E,N}
#   - ID_r structural (PDB)             → FASTA `>` line + sequence
#   - Free-form (DFT-C, HFE, HFD, QECC_65, GEO) → plain text
_ANSWER_FORMAT_HINT: dict[str, str] = {
    "DFT-S": 'A JSON list of structure-metadata objects, e.g. `[{"key": "value", ...}, ...]`.',
    "DFT-P": 'A JSON list of DFT-parameter objects, e.g. `[{"key": "value", ...}, ...]`.',
    "MPVE":  'A JSON list of materials-property/value objects, e.g. `[{"name": "...", "value": ...}, ...]`.',
    "BIOGR": 'A JSON object with the geographic bounding box: `{"W": <west>, "S": <south>, "E": <east>, "N": <north>}` (numbers).',
    "PDB":   'A FASTA-format protein sequence: a `>` header line followed by the sequence on the next line.',
    "DFT-C": "Free-form text or code as the input requests.",
    "HFE":   "Free-form scientific explanation.",
    "HFD":   "Free-form scientific description.",
    "QECC_65": "Free-form answer to the quantum-error-correction question.",
    "GEO":   "Free-form geological/geographical answer.",
}

_TASK_PROMPT_TEMPLATE = """\
CURIE task: {task_id} (difficulty={difficulty}).

You CANNOT see the input directly. The long-context input is on disk at \
`context.txt` in your working directory (Python's cwd inside the REPL). \
The ONLY way to read it is by calling `call_python_repl`. Free-text answers \
without tool calls will be scored as zero.

You have a HARD BUDGET of 12 root-model turns. Plan accordingly:
  Turn 1: read context.txt and identify the task question.
  Turns 2–4: use `llm_batch([prompt1, prompt2, ...])` in parallel to extract \
or summarize parts of the input as needed.
  Turns 5–10: combine results, do any final analysis in the REPL.
  Turn 11–12: SUBMIT (see below). If you are running out of turns, write your \
best partial answer and submit anyway — submitting a guess is strictly \
better than not submitting.

REQUIRED first call (copy-paste, then continue from there):

    text = open("context.txt").read()
    print("LENGTH:", len(text))
    print(text[:3000])

Answer format for {task_id}: {answer_format}

TO SUBMIT, make a top-level tool call to `submit_answer` with your final \
answer string:

    submit_answer(content=<your final answer as a string>)

Do not call `submit_answer` from inside Python, and do not end with plain chat \
text. If you are already inside `call_python_repl`, the equivalent answer-file \
path is also valid:

    answer["content"] = <your final answer as a string>
    answer["ready"] = True

The rollout ends the moment the answer file contains `{{"ready": true, \
"content": "..."}}`. Do not write the answer in chat — only submitted content \
is scored.
"""


def _build_task_prompt(task_id: str, difficulty: str) -> str:
    return _TASK_PROMPT_TEMPLATE.format(
        task_id=task_id,
        difficulty=difficulty,
        answer_format=_ANSWER_FORMAT_HINT[task_id],
    )


VALID_SPLITS = frozenset({"train", "val", "test"})

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data" / "curie" / "data" / "data"
_SPLITS_DIR = _PROJECT_ROOT / "data" / "curie" / "splits"


def _row_from_split_entry(
    entry: dict[str, Any], task_id: str, dft_field: str | None, folder_difficulty: dict[str, str]
) -> dict[str, Any]:
    """Build a Dataset row from a splits-file entry.

    Long input goes into `info["context"]` (RLMEnv writes it to
    ``<rlm_fs_root>/context.txt`` inside the sandbox); the visible prompt
    shrinks to a short task instruction.
    """
    record_id = entry["record_id"]
    difficulty = folder_difficulty[record_id]
    info: dict[str, Any] = {
        "record_id": record_id,
        "task_id": task_id,
        "difficulty": difficulty,
        "dft_field": dft_field,
        # RLMEnv contract: info["context"] (string) → /<rlm_fs_root>/context.txt
        # inside the sandbox. The REPL's cwd is rlm_fs_root, so the model reads
        # via `open("context.txt").read()`.
        "context": entry["input"],
    }
    return {
        "prompt": [{"role": "user", "content": _build_task_prompt(task_id, difficulty)}],
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


def load_curie_task(task_id: str, split: str) -> datasets.Dataset:
    """Load CURIE benchmark records for the given task_id and split.

    `split` must be one of "train", "val", "test" and the corresponding
    `data/curie/splits/{split}.jsonl` file must exist (built by
    `scripts/build_splits.py`).

    Raises:
        ValueError: task_id not in TASK_MAP, or split not in {"train","val","test"}.
        FileNotFoundError: split file is missing for any split, including "test".
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
    if not splits_file.exists():
        raise FileNotFoundError(
            f"Splits file {splits_file} does not exist. "
            f"Run: uv run python scripts/build_splits.py before loading split={split!r}."
        )
    return _load_from_splits(task_id, split)
