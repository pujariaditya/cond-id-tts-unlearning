"""Download the pretrained TTS checkpoints for the three backbones.

* XTTS-v2   -> ``<repo>/models/xtts_v2`` (override with ``CD_XTTS_DIR``).
              Coqui Public Model License (CPML) -- review its terms before use.
* IndexTTS-1.5 -> ``<repo>/models/index_tts`` (override with ``CD_INDEXTTS_DIR``).
              Run inside the IndexTTS venv (see scripts/setup_indextts.sh).
* Tortoise-TTS downloads itself from the HuggingFace Hub on first synthesis (via
  the ``tortoise``/``coqui-tts`` stack) -- no explicit step needed.

The evaluation-only models (WavLM-Base-Plus-SV verifier, Faster-Whisper large-v3
for WER) auto-download from the HuggingFace Hub on first use.

Usage:
    python scripts/download_models.py                    # XTTS-v2 (default)
    python scripts/download_models.py --model indextts   # IndexTTS-1.5 checkpoint
    python scripts/download_models.py --model all
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Checkpoint revisions pinned to the exact commits used for the paper (reproducible).
# Override per-invocation with ``--xtts-revision`` / ``--indextts-revision``.
# Availability mirrors (unmodified): RootAccess4Life/cond-id-xtts-v2 and
# RootAccess4Life/cond-id-indextts-1.5 on the HuggingFace Hub.
XTTS_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"  # coqui/XTTS-v2
INDEXTTS_REVISION = "25851a6036dfd3095bb70fb3c8f49217104672c3"  # IndexTeam/IndexTTS-1.5


def download_xtts(dest: Path, revision: str = XTTS_REVISION):
    """Fetch the XTTS-v2 checkpoint from the HuggingFace Hub into ``dest``.

    Args:
        dest: Destination directory for the checkpoint files.
        revision: HuggingFace repo revision (commit SHA / tag / branch) to fetch;
            pin to an exact SHA for reproducibility (see the module-level constants).
    """
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    needed = [
        "config.json",
        "model.pth",
        "vocab.json",
        "dvae.pth",
        "mel_stats.pth",
        "speakers_xtts.pth",
    ]
    print(f"Downloading XTTS-v2 (coqui/XTTS-v2@{revision}) -> {dest} ...")
    snapshot_download(
        repo_id="coqui/XTTS-v2", revision=revision, local_dir=str(dest), allow_patterns=needed
    )
    missing = [
        f
        for f in ("config.json", "model.pth", "dvae.pth", "mel_stats.pth")
        if not (dest / f).exists()
    ]
    if missing:
        sys.exit(f"XTTS-v2 download incomplete; missing {missing}")
    print(f"XTTS-v2 ready at {dest}")


def download_indextts(dest: Path, revision: str = INDEXTTS_REVISION):
    """Fetch the IndexTTS-1.5 checkpoint from the HuggingFace Hub into ``dest``.

    Args:
        dest: Destination directory for the checkpoint files.
        revision: HuggingFace repo revision (commit SHA / tag / branch) to fetch;
            pin to an exact SHA for reproducibility (see the module-level constants).
    """
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    # ``bigvgan_discriminator.pth`` is training-only and never loaded at inference
    # (see cond_id/models/indextts.py), so it is intentionally not fetched.
    needed = ["config.yaml", "gpt.pth", "bigvgan_generator.pth", "dvae.pth", "bpe.model"]
    print(f"Downloading IndexTTS-1.5 (IndexTeam/IndexTTS-1.5@{revision}) -> {dest} ...")
    snapshot_download(
        repo_id="IndexTeam/IndexTTS-1.5",
        revision=revision,
        local_dir=str(dest),
        allow_patterns=needed,
    )
    missing = [
        f for f in ("config.yaml", "gpt.pth", "bigvgan_generator.pth") if not (dest / f).exists()
    ]
    if missing:
        sys.exit(f"IndexTTS-1.5 download incomplete; missing {missing}")
    print(f"IndexTTS-1.5 ready at {dest}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=["xtts", "indextts", "all"],
        default="xtts",
        help="which checkpoint to download (Tortoise auto-downloads on first use).",
    )
    p.add_argument("--xtts-dir", default=str(_REPO_ROOT / "models" / "xtts_v2"))
    p.add_argument("--indextts-dir", default=str(_REPO_ROOT / "models" / "index_tts"))
    p.add_argument(
        "--xtts-revision",
        default=XTTS_REVISION,
        help="coqui/XTTS-v2 repo revision (commit SHA / tag / branch) to fetch.",
    )
    p.add_argument(
        "--indextts-revision",
        default=INDEXTTS_REVISION,
        help="IndexTeam/IndexTTS-1.5 repo revision (commit SHA / tag / branch) to fetch.",
    )
    args = p.parse_args()

    if args.model in ("xtts", "all"):
        download_xtts(Path(args.xtts_dir), revision=args.xtts_revision)
    if args.model in ("indextts", "all"):
        download_indextts(Path(args.indextts_dir), revision=args.indextts_revision)
    print(
        "Tortoise-TTS downloads on first synthesis; WavLM + Faster-Whisper "
        "auto-download from the HuggingFace Hub on first use."
    )


if __name__ == "__main__":
    main()
