"""Command-line interface for cond-ID (the ``cond-id`` console script).

Subcommands:

* ``eval``    -- reproduce the held-out evaluation for one backbone from a config
  YAML (the same pipeline as ``scripts/run_eval.py``).
* ``unlearn`` -- unlearn a set of forget speakers from your own audio and export an
  edited-model **adapter patch** (see :mod:`cond_id.adapter`).
* ``synth``   -- synthesize text from an edited model (base + patch) cloning a
  reference clip.

The ``cond-id-eval`` alias maps directly to the ``eval`` subcommand.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

# A small, fixed sentence bank cloned once per speaker (positionally aligned with
# the cB/cE/cRL clone lists). Kept short and phonetically varied for stable WER.
_SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It is easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
]


def _load_config(path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def _env_provenance(config: dict) -> dict:
    """Capture how a run was produced, for the results JSON (git, versions, hardware)."""
    import platform
    import subprocess

    import torch

    def _pkg(name):
        try:
            from importlib.metadata import version

            return version(name)
        except Exception:
            return None

    def _git(*args):
        try:
            return (
                subprocess.check_output(
                    ["git", "-C", str(Path(__file__).resolve().parent), *args],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    cuda_ok = torch.cuda.is_available()
    return {
        "git_commit": commit,
        "git_dirty": (bool(status) if status is not None else None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: _pkg(p)
            for p in (
                "torch",
                "torchaudio",
                "transformers",
                "faster-whisper",
                "coqui-tts",
                "speechbrain",
                "numpy",
            )
        },
        "cuda": getattr(torch.version, "cuda", None),
        "cudnn": (torch.backends.cudnn.version() if cuda_ok else None),
        "gpu": (torch.cuda.get_device_name() if cuda_ok else None),
        "config": config,
    }


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def run_evaluation(
    config: str,
    out: str | None = None,
    seed: int | None = None,
    dry_run: bool = False,
    no_relearn: bool = False,
    splits: int | None = None,
    strict: bool = False,
    results_root: Path | None = None,
) -> int:
    """Run the held-out cond-ID evaluation for one backbone and write a metrics JSON.

    This is the single implementation shared by the ``eval`` subcommand and
    ``scripts/run_eval.py``. ``results_root`` is the base for default output paths
    (``<root>/results/<model>.json`` and the clone directory); it defaults to the
    current working directory.

    Args:
        config: Path to the per-model config YAML.
        out: Output metrics JSON path (default ``<results_root>/results/<model>.json``).
        seed: Override the config's master seed.
        dry_run: Tiny smoke run (2 forget / 1 retain / 1 sentence, no relearn).
        no_relearn: Skip the relearn adversary.
        splits: Number of random forget/retain splits to average.
        results_root: Base directory for default result/clone paths.

    Returns:
        Process exit code (``0`` on success).
    """
    root = Path(results_root) if results_root is not None else Path.cwd()
    cfg = _load_config(config)
    tag = cfg["model"]
    seed = seed if seed is not None else int(cfg.get("eval", {}).get("seed", 0))

    from cond_id import UnlearnConfig, build_host, set_determinism
    from cond_id.datasets import pool
    from cond_id.eval.clone import (
        CloneConfig,
        build_splits,
        synth_baseline,
        synth_edited_clones,
    )
    from cond_id.eval.score import score_split

    set_determinism(seed, strict=strict or os.environ.get("CD_STRICT") == "1")

    e = dict(cfg.get("eval", {}))
    if dry_run:
        e.update(n_forget=2, n_retain_seen=1, n_retain_unseen=0, sentences=1)
    n_f = int(e.get("n_forget", 4))
    n_rs = int(e.get("n_retain_seen", 3))
    n_ru = int(e.get("n_retain_unseen", 1))
    n_sent = int(e.get("sentences", 2))
    sentences = _SENTENCES[: max(1, n_sent)]

    n_splits = int(splits if splits is not None else e.get("n_splits", 1))
    all_speakers = pool(20)

    def _select(sub_seed):
        rng = random.Random(sub_seed)
        sp = all_speakers[:]
        rng.shuffle(sp)
        return sp[:n_f], sp[n_f : n_f + n_rs], sp[n_f + n_rs : n_f + n_rs + n_ru]

    uconf = UnlearnConfig(**cfg.get("unlearn", {}))
    clone_cfg = CloneConfig(unlearn=uconf, out_dir=str(root / "results" / f"{tag}_clones"))
    run_relearn = (tag == "xtts") and not no_relearn and not dry_run
    r = dict(cfg.get("relearn", {}))
    host = build_host(tag)

    from cond_id.methods.cond_id import applied_unlearning

    # Each split draws a random forget/retain group and is scored on held-out clips;
    # metrics are averaged over splits (the paper protocol), which keeps a single
    # hard-to-erase speaker from dominating the worst-speaker-weighted erasure.
    numeric = [
        "forget_erasure_abs",
        "retain_keep",
        "retain_wer",
        "retain_wer_base",
        "forget_wer",
        "relearn_robust",
    ]
    acc = {k: [] for k in numeric}
    per_split = []
    for s in range(n_splits):
        sub = seed * 1000 + s
        forget, retain_seen, retain_unseen = _select(sub)
        print(
            f"[run_eval] {tag} split {s + 1}/{n_splits} seed={sub} forget={forget} "
            f"relearn={run_relearn}",
            flush=True,
        )
        split = build_splits(host, forget, retain_seen, retain_unseen, sentences, sub)[0]
        synth_baseline(host, split, clone_cfg)
        with applied_unlearning(host, forget, retain_seen, seed=sub, config=uconf):
            synth_edited_clones(host, split, clone_cfg)
            if run_relearn:
                from cond_id.eval.relearn import relearn_attack

                relearn_attack(
                    host,
                    split,
                    seed=sub,
                    steps=int(r.get("steps", 40)),
                    n=int(r.get("n", 3)),
                    max_seconds=float(r.get("max_seconds", 600)),
                    out_dir=str(root / "results" / f"{tag}_clones"),
                )
        m = score_split(split, whisper_model=e.get("whisper_model", "large-v3"))
        per_split.append(m)
        for k in numeric:
            v = m.get(k)
            if v is not None:
                acc[k].append(float(v))

    metrics = {k: (sum(v) / len(v) if v else None) for k, v in acc.items()}

    out_path = Path(out) if out else (root / "results" / f"{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": tag,
        "seed": seed,
        "n_splits": n_splits,
        "metrics": metrics,
        "per_split": per_split,
        "env": _env_provenance(cfg),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    def fmt(k):
        return f"{metrics[k]:.3f}" if metrics.get(k) is not None else "None"

    _rw, _rwb = metrics.get("retain_wer"), metrics.get("retain_wer_base")
    retain_wer_delta = f"{_rw - _rwb:+.3f}" if (_rw is not None and _rwb is not None) else "None"
    print(
        f"[run_eval] {tag} ({n_splits} splits avg): "
        f"erase={fmt('forget_erasure_abs')} retain_keep={fmt('retain_keep')} "
        f"retain_wer_delta={retain_wer_delta} "
        f"forget_wer={fmt('forget_wer')} relearn_robust={fmt('relearn_robust')}",
        flush=True,
    )
    print(f"[run_eval] wrote {out_path}", flush=True)
    return 0


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="per-model config YAML (configs/<model>.yaml)")
    p.add_argument(
        "--out", default=None, help="output metrics JSON (default: results/<model>.json)"
    )
    p.add_argument("--seed", type=int, default=None, help="override the config seed")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="tiny smoke run (2 forget / 1 retain / 1 sentence, no relearn)",
    )
    p.add_argument("--no-relearn", action="store_true", help="skip the relearn adversary")
    p.add_argument(
        "--splits",
        type=int,
        default=None,
        help="number of random forget/retain splits to average (default: config n_splits or 1)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="request fully deterministic CUDA kernels (torch.use_deterministic_algorithms)",
    )


def _cmd_eval(args) -> int:
    return run_evaluation(
        args.config,
        out=args.out,
        seed=args.seed,
        dry_run=args.dry_run,
        no_relearn=args.no_relearn,
        splits=args.splits,
        strict=args.strict,
    )


# ---------------------------------------------------------------------------
# unlearn / synth  (the public package feature)
# ---------------------------------------------------------------------------
def _cmd_unlearn(args) -> int:
    from .api import unlearn

    edited = unlearn(
        args.model,
        forget=args.forget_dir,
        retain=args.retain_dir,
        config=args.config,
        seed=args.seed,
        out=args.out,
    )
    print(
        f"[unlearn] {args.model}: erased {len(edited.forget_labels)} speaker(s); "
        f"adapter patch written to {args.out}",
        flush=True,
    )
    return 0


def _cmd_synth(args) -> int:
    import soundfile as sf

    from .api import load_edited

    edited = load_edited(args.model_dir, base=args.base)
    wav = edited.synth(args.text, args.reference)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav, edited.sr)
    print(f"[synth] wrote {args.out} ({edited.sr} Hz)", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="cond-id",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("eval", help="reproduce held-out evaluation for one backbone")
    _add_eval_args(pe)
    pe.set_defaults(func=_cmd_eval)

    pu = sub.add_parser("unlearn", help="unlearn forget speakers and export an adapter patch")
    pu.add_argument(
        "--model", required=True, choices=["xtts", "tortoise", "indextts"], help="backbone to edit"
    )
    pu.add_argument(
        "--forget-dir", required=True, help="directory of forget speakers: <label>/*.wav"
    )
    pu.add_argument(
        "--retain-dir", required=True, help="directory of retain speakers: <label>/*.wav"
    )
    pu.add_argument(
        "--config",
        default=None,
        help="config YAML supplying the 'unlearn:' hyperparameters (else model defaults)",
    )
    pu.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    pu.add_argument("--out", required=True, help="output directory for the adapter patch")
    pu.set_defaults(func=_cmd_unlearn)

    ps = sub.add_parser("synth", help="synthesize text from an edited model cloning a reference")
    ps.add_argument("--model-dir", required=True, help="adapter directory written by 'unlearn'")
    ps.add_argument(
        "--base",
        default=None,
        help="base checkpoint directory (default: the patch's recorded base / env)",
    )
    ps.add_argument("--text", required=True, help="text to synthesize")
    ps.add_argument("--reference", required=True, help="reference clip to clone the voice from")
    ps.add_argument("--out", required=True, help="output WAV path")
    ps.set_defaults(func=_cmd_synth)

    return ap


def main(argv=None) -> int:
    """Entry point for the ``cond-id`` console script (subcommand dispatcher)."""
    args = _build_parser().parse_args(argv)
    return args.func(args)


def main_eval(argv=None) -> int:
    """Entry point for the ``cond-id-eval`` alias (the ``eval`` subcommand)."""
    ap = argparse.ArgumentParser(
        prog="cond-id-eval", description="Reproduce held-out evaluation for one backbone."
    )
    _add_eval_args(ap)
    return _cmd_eval(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
