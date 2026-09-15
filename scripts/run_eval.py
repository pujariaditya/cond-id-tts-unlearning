#!/usr/bin/env python3
"""Reproducible cond-ID unlearning + relearn + held-out evaluation for one model.

Runs the full pipeline for a single backbone from a per-model config:

    build splits -> baseline clones (cB) -> unlearning fine-tune + edited clones (cE)
    -> relearn adversary (cRL, XTTS only) -> held-out scoring

and writes a metrics JSON. Determinism is pinned by ``--seed`` and the config;
results are reproducible on fixed hardware/CUDA.

This is a thin wrapper around ``cond_id.cli.run_evaluation`` (the same code the
``cond-id eval`` console command runs), kept for the reproduction docs and so
default output paths resolve under the repository root regardless of the working
directory.

Usage:
    python scripts/run_eval.py --config configs/xtts.yaml --out results/xtts.json
    python scripts/run_eval.py --config configs/xtts.yaml --dry-run      # smoke run
    python scripts/run_eval.py --config configs/tortoise.yaml --no-relearn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main() -> int:
    from cond_id.cli import _add_eval_args, run_evaluation

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _add_eval_args(ap)
    args = ap.parse_args()
    return run_evaluation(
        args.config,
        out=args.out,
        seed=args.seed,
        dry_run=args.dry_run,
        no_relearn=args.no_relearn,
        splits=args.splits,
        strict=args.strict,
        results_root=_REPO,
    )


if __name__ == "__main__":
    raise SystemExit(main())
