"""CLI for Stage 3.5 stratified split builder.

Usage:
    uv run python scripts/build_splits.py

Writes data/curie/splits/{train,val,test}.jsonl. Idempotent — re-running with
the same locked seed produces byte-identical files.
"""
from __future__ import annotations

import sys
from pathlib import Path

from curie_rlm_env.splits import (
    LOCKED_SEED,
    SPLITS_DIR,
    build_and_write_splits,
)


def main() -> int:
    splits = build_and_write_splits()

    print(f"Wrote splits to {SPLITS_DIR}/  (seed={LOCKED_SEED})")
    print()
    print(f"{'Task':<10s} {'Train':>6s} {'Val':>6s} {'Test':>6s} {'Total':>6s}")
    print("-" * 40)
    grand = {"train": 0, "val": 0, "test": 0}
    for tid in sorted(splits.keys()):
        n_train = len(splits[tid]["train"])
        n_val = len(splits[tid]["val"])
        n_test = len(splits[tid]["test"])
        total = n_train + n_val + n_test
        grand["train"] += n_train
        grand["val"] += n_val
        grand["test"] += n_test
        print(f"{tid:<10s} {n_train:>6d} {n_val:>6d} {n_test:>6d} {total:>6d}")
    print("-" * 40)
    overall = grand["train"] + grand["val"] + grand["test"]
    print(f"{'TOTAL':<10s} {grand['train']:>6d} {grand['val']:>6d} {grand['test']:>6d} {overall:>6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
