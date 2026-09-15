#!/usr/bin/env python3
"""Build self-contained VCTK subsets for cond-ID.

Reads the raw VCTK-Corpus-0.92 tree (produced by ``scripts/download_vctk.sh``)
and, for every subset declared in ``cond_id.config.SUBSETS``, selects at
least 16 utterances per listed speaker, writes them as ``{idx}.flac`` (a single
global running integer ``idx`` used as the filename stem) under the subset's
``AUDIO_DIR``, and emits the speaker-map JSON::

    { "pXXX": [idx, idx, ...], ... }

This matches the data contract the rest of the package reads (see
``cond_id.config`` and ``scripts/run_eval.py``).

The script is idempotent: re-running it reuses already-written ``{idx}.flac``
files and a matching speaker-map JSON (it only rebuilds a subset when the JSON
is missing, unreadable, or its speaker set / per-speaker counts no longer
satisfy the contract).

Dependencies: stdlib + soundfile. Clips are written verbatim at their source rate
(mono-downmixed if a file is multi-channel); VCTK is 48 kHz flac and the downstream
code resamples to 16 kHz itself, so no resampling happens here.

Usage:
    python scripts/prepare_vctk.py                  # build all subsets
    python scripts/prepare_vctk.py --eval-size 20   # build only the 20-spk subset
    python scripts/prepare_vctk.py --raw-dir /data/vctk
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the repository importable so we can read the single source of truth
# (cond_id.config.SUBSETS) regardless of where the script is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import soundfile as sf  # noqa: E402

from cond_id.config import DATA_DIR, SUBSETS  # noqa: E402

# Minimum utterances per speaker required by the data contract. Evaluation draws
# disjoint held-out windows at fractional offsets up to ~0.82*N (the relearn-attack
# partition; see cond_id.datasets), so each speaker needs enough clips to fill
# the train slice and those windows without overlap.
MIN_UTTS = 16

# Preferred microphone channel. VCTK-0.92 records each utterance on two mics;
# mic1 is the omnidirectional DPA and is the conventional choice.
PREFERRED_MIC = "mic1"


def _default_raw_dir() -> Path:
    """Cache dir consistent with download_vctk.sh (env override or data/_raw)."""
    env = os.environ.get("VCTK_RAW_DIR")
    if env:
        return Path(env)
    return _REPO_ROOT / "data" / "_raw"


def _find_audio_root(raw_dir: Path) -> Path:
    """Locate the per-speaker audio directory inside a raw VCTK tree.

    VCTK-0.92 stores audio under ``wav48_silence_trimmed/``. Some unzips nest
    the corpus one level deeper (e.g. ``VCTK-Corpus-0.92/...``); search a few
    candidates so the script is robust to either layout.
    """
    candidates = [
        raw_dir / "wav48_silence_trimmed",
        raw_dir / "VCTK-Corpus-0.92" / "wav48_silence_trimmed",
        raw_dir / "wav48",  # older releases
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    # Fall back to a shallow glob in case of an unexpected wrapper directory.
    for hit in raw_dir.glob("*/wav48_silence_trimmed"):
        if hit.is_dir():
            return hit
    raise FileNotFoundError(
        f"Could not find a VCTK audio root under {raw_dir}. "
        "Run scripts/download_vctk.sh first (or pass --raw-dir)."
    )


def _speaker_utterances(audio_root: Path, speaker: str) -> list[Path]:
    """Sorted list of preferred-mic flac files for one speaker.

    Prefers ``*_mic1.flac``; falls back to mic2 (then any flac) only for
    utterances that lack a mic1 recording, so each utterance is represented at
    most once. Sorting is by the utterance number for determinism.
    """
    spk_dir = audio_root / speaker
    if not spk_dir.is_dir():
        raise FileNotFoundError(f"Speaker directory missing: {spk_dir}")

    by_utt: dict[str, dict[str, Path]] = {}
    for f in spk_dir.glob("*.flac"):
        stem = f.stem  # e.g. p225_001_mic1
        parts = stem.split("_")
        if len(parts) >= 3:
            utt_key = parts[1]
            mic = parts[2]
        elif len(parts) == 2:
            utt_key = parts[1]
            mic = "mic1"
        else:
            continue
        by_utt.setdefault(utt_key, {})[mic] = f

    chosen: list[tuple[str, Path]] = []
    for utt_key in sorted(by_utt):
        mics = by_utt[utt_key]
        if PREFERRED_MIC in mics:
            chosen.append((utt_key, mics[PREFERRED_MIC]))
        elif "mic2" in mics:
            chosen.append((utt_key, mics["mic2"]))
        else:
            chosen.append((utt_key, next(iter(mics.values()))))
    return [p for _, p in chosen]


def _copy_flac(src: Path, dst: Path) -> None:
    """Write ``src`` to ``dst`` as flac (mono, at the source sample rate).

    Reads with soundfile, downmixes to mono if the file is multi-channel, and
    writes the samples verbatim at their original rate -- no resampling happens
    here. The downstream loaders resample to 16 kHz, so the on-disk rate only
    affects file size, not correctness.
    """
    data, sr = sf.read(str(src), dtype="float32", always_2d=False)
    # Downmix to mono if a multi-channel file ever appears.
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    sf.write(str(dst), data, sr, format="FLAC")


def _map_is_valid(map_path: Path, audio_dir: Path, speakers: list[str]) -> bool:
    """True if an existing speaker map already satisfies the contract."""
    if not map_path.is_file():
        return False
    try:
        smap = json.loads(map_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if set(smap.keys()) != set(speakers):
        return False
    for sp in speakers:
        idxs = smap.get(sp, [])
        if not isinstance(idxs, list) or len(idxs) < MIN_UTTS:
            return False
        for idx in idxs:
            if not (audio_dir / f"{idx}.flac").is_file():
                return False
    return True


def build_subset(size: str, spec: dict, audio_root: Path, force: bool) -> dict:
    """Build one subset. Returns a per-speaker count summary dict."""
    speakers: list[str] = list(spec["speakers"])
    audio_dir = DATA_DIR / spec["audio_subdir"]
    map_path = DATA_DIR / spec["speaker_map"]

    if not force and _map_is_valid(map_path, audio_dir, speakers):
        smap = json.loads(map_path.read_text())
        print(f"  [size {size}] up to date — {len(speakers)} speakers, reusing {map_path.name}")
        return {sp: len(smap[sp]) for sp in speakers}

    audio_dir.mkdir(parents=True, exist_ok=True)

    smap: dict[str, list[int]] = {}
    idx = 0  # global running integer across all speakers in THIS subset
    for sp in speakers:
        utts = _speaker_utterances(audio_root, sp)
        if len(utts) < MIN_UTTS:
            raise RuntimeError(
                f"Speaker {sp} has only {len(utts)} usable utterances "
                f"(< {MIN_UTTS} required) in {audio_root}."
            )
        smap[sp] = []
        for src in utts:
            dst = audio_dir / f"{idx}.flac"
            if not dst.is_file():
                _copy_flac(src, dst)
            smap[sp].append(idx)
            idx += 1

    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(smap, indent=2))
    print(
        f"  [size {size}] wrote {idx} files to {audio_dir} and "
        f"{map_path.name} ({len(speakers)} speakers)"
    )
    return {sp: len(smap[sp]) for sp in speakers}


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_manifest(sizes: list[str], summary: dict) -> Path:
    """Write ``data/manifest.json`` recording each subset's speaker map + checksum.

    The checksums make the prepared data verifiable: a stranger who reruns this script
    on VCTK-0.92 should get matching ``speaker_map_sha256`` values.
    """
    manifest = {
        "dataset": "CSTR VCTK-Corpus-0.92",
        "doi": "10.7488/ds/2645",
        "license": "CC-BY-4.0",
        "subsets": {},
    }
    # Record every subset currently prepared on disk (not just the ones built this
    # run), so a partial rebuild (``--eval-size N``) does not drop other subsets'
    # entries. Counts for subsets not rebuilt now are recomputed from their map file.
    for size, spec in SUBSETS.items():
        map_path = DATA_DIR / spec["speaker_map"]
        if size not in sizes and not map_path.is_file():
            continue
        counts = summary.get(size)
        if counts is None and map_path.is_file():
            try:
                counts = {sp: len(v) for sp, v in json.loads(map_path.read_text()).items()}
            except (json.JSONDecodeError, OSError):
                counts = {}
        manifest["subsets"][size] = {
            "speaker_map": spec["speaker_map"],
            "audio_subdir": spec["audio_subdir"],
            "speakers": list(spec["speakers"]),
            "utterances_per_speaker": counts or {},
            "speaker_map_sha256": _sha256(map_path) if map_path.is_file() else None,
        }
    out = DATA_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=_default_raw_dir(),
        help="Raw VCTK-Corpus-0.92 directory (default: $VCTK_RAW_DIR or <repo>/data/_raw).",
    )
    ap.add_argument(
        "--eval-size",
        choices=sorted(SUBSETS.keys()),
        default=None,
        help="Build only this subset size (default: build all).",
    )
    ap.add_argument(
        "--force", action="store_true", help="Rebuild even if an existing speaker map looks valid."
    )
    args = ap.parse_args()

    audio_root = _find_audio_root(args.raw_dir)
    print(f"VCTK audio root: {audio_root}")

    sizes = [args.eval_size] if args.eval_size else sorted(SUBSETS.keys(), key=int)

    summary: dict[str, dict] = {}
    for size in sizes:
        print(f"Building subset size {size} ...")
        summary[size] = build_subset(size, SUBSETS[size], audio_root, args.force)

    print("\n=== Summary (utterances per speaker) ===")
    for size in sizes:
        counts = summary[size]
        total = sum(counts.values())
        print(f"size {size}: {len(counts)} speakers, {total} files total")
        for sp in sorted(counts):
            print(f"    {sp}: {counts[sp]}")

    manifest_path = _write_manifest(sizes, summary)
    print(f"\nWrote data manifest: {manifest_path}")


if __name__ == "__main__":
    main()
