# cond-ID

**Host-independent speaker unlearning for autoregressive zero-shot TTS.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20demo-ffcc4d)](https://huggingface.co/spaces/RootAccess4Life/cond-id-demo)
[![Audio samples](https://img.shields.io/badge/%F0%9F%8E%A7-Audio%20samples-blue)](https://huggingface.co/datasets/RootAccess4Life/cond-id-samples)

> Reference code for the paper,
> **"cond-ID: Conditioning-Space Identity Redirection for Speaker Unlearning in Zero-Shot TTS."**
> Aditya Pujari and Ajita Rattani — University of North Texas, Denton, Texas, USA.
> *Paper: link to be added.*
>
> 🔇 **[Live audio demo](https://huggingface.co/spaces/RootAccess4Life/cond-id-demo)** — hear a
> voice cloned, then unlearned, across all three backbones &nbsp;·&nbsp;
> 🎧 [full sample set](https://huggingface.co/datasets/RootAccess4Life/cond-id-samples) &nbsp;·&nbsp;
> 📦 checkpoint mirrors: [XTTS-v2](https://huggingface.co/RootAccess4Life/cond-id-xtts-v2) ·
> [IndexTTS-1.5](https://huggingface.co/RootAccess4Life/cond-id-indextts-1.5)

## Overview

Zero-shot TTS clones a speaker from a few seconds of audio, which makes
**selective speaker removal** a deployment requirement: an operator must be able to
honor an opt-out and stop an already-deployed model from reproducing a target
speaker's identity, without retraining from scratch and without degrading everyone
else.

**cond-ID (Conditioning-Space Identity Redirection)** is a single training-based method
that does this by fine-tuning only the model's *speaker-conditioning path* for a
few steps: it decomposes the conditioning latent into a pooled identity and a
per-token structure residual, redirects each forget speaker's identity onto its own
dispersed orthonormal anchor, preserves the residual (intelligibility) and pins the
retain speakers to the frozen model. In reproducible evaluation the edit is
snapshotted and restored on exit; the library API (`unlearn`) instead persists it
and exports a small adapter patch. One method, written against a small host interface
(`cond_latents` / `cond_params` / `token_axis`), runs on three architecturally
distinct backbones:

- **XTTS-v2** — autoregressive codec language model
- **Tortoise-TTS** — autoregressive
- **IndexTTS-1.5** — autoregressive, perceiver-conditioned (BigVGAN vocoder)

Erasure, retention, intelligibility and **relearn-robustness** (does the erasure
survive a white-box re-finetuning attack?) are all measured on **held-out** clips
the method never sees.

## Results

cond-ID erases held-out forget identity while preserving retain identity and
intelligibility on all three backbones, each reproduced end-to-end by this repo
(VCTK, 20-speaker pool, averaged over random forget/retain splits). The columns are
held-out forget **erasure** (higher = more erased), **retain-keep** (edited/baseline
retain identity; 1.0 = fully preserved), retain-WER **Δ** over the un-edited baseline,
**forget-WER**, and **relearn-robustness**:

| Backbone     | erasure | retain-keep | retain-WER Δ | forget-WER | relearn-robust |
|--------------|:-------:|:-----------:|:------------:|:----------:|:--------------:|
| IndexTTS-1.5 | 0.62    | 0.91        | +0.00        | 0.00       | —              |
| Tortoise-TTS | 0.71    | 0.92        | +0.01        | 0.01       | —              |
| XTTS-v2      | 0.44    | 0.89        | +0.03        | 0.07       | 0.48           |

These pool 5 splits × 3 seeds (n=15), matching Table 1 of the paper. Erasure is
worst-speaker-weighted and averaged over random forget groups, so single splits vary
(95% CI ±0.12 / 0.14 / 0.15); use `--splits 5` (the default) and multiple seeds for a
stable estimate. Relearn-robustness (survival of a white-box re-finetune attack) is
reported for XTTS-v2, whose official codec LM loss makes the strongest adversary; it is
bimodal, with 4/15 splits fully recovering. Numbers are reproduced by
`scripts/run_eval.py` (see below); expect small variation across hardware/CUDA.

## Install

The three backbones have conflicting dependencies, so the repo is a light,
model-agnostic core plus per-model setup. **XTTS-v2 and Tortoise-TTS share one
environment; IndexTTS-1.5 uses its own.**

```bash
git clone https://github.com/pujariaditya/cond-id-tts-unlearning.git
cd cond-id-tts-unlearning

# XTTS-v2 + Tortoise-TTS environment
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-core.txt
pip install -e .

# IndexTTS-1.5 (isolated environment + checkpoint)
bash scripts/setup_indextts.sh
```

Tested on Python 3.10, CUDA 12.4, cuDNN 9, a single 48 GB GPU. WER uses
Faster-Whisper on **CPU by default** (portable); set `CD_FW_DEVICE=cuda` for speed.

## Reproduce the paper

```bash
# 1. Dataset (VCTK-Corpus-0.92, CC-BY-4.0) — download + prepare the 20-speaker pool
bash scripts/download_vctk.sh
python scripts/prepare_vctk.py

# 2. Checkpoints (XTTS auto-path; Tortoise self-downloads; IndexTTS via setup script)
python scripts/download_models.py --model xtts

# 3. Full pipeline per backbone: unlearning fine-tune -> relearn adversary (XTTS)
#    -> held-out scoring -> results/<model>.json
bash scripts/reproduce.sh xtts
bash scripts/reproduce.sh tortoise
models/index_tts/.venv/bin/python scripts/run_eval.py --config configs/indextts.yaml
```

A fast end-to-end smoke test (2 forget / 1 retain / 1 sentence, no relearn):

```bash
python scripts/run_eval.py --config configs/xtts.yaml --dry-run
```

Per-model hyperparameters live in `configs/{xtts,tortoise,indextts}.yaml`.
Checkpoint revisions are commit-pinned in `scripts/download_models.py`.

## Use as a library

Unlearn your own speakers and get an edited model back. You supply reference audio
for the speakers to forget and to retain; cond-ID exports the edit as a small
**adapter patch** (a few MB, no base weights — so sharing it is license-clean):

```python
import cond_id

edited = cond_id.unlearn(
    "xtts",
    forget={"alice": ["alice/a1.wav", "alice/a2.wav"]},   # or forget="forget/"
    retain={"carol": ["carol/c1.wav"]},                   # or retain="retain/"
    out="edited_xtts/",
)
wav = edited.synth("Hello, world.", reference="alice/a1.wav")   # 'alice' is misdirected
```

or from the command line:

```bash
cond-id unlearn --model xtts --forget-dir forget/ --retain-dir retain/ --out edited_xtts/
cond-id synth   --model-dir edited_xtts/ --text "Hello." --reference ref.wav --out out.wav
```

**Caveat.** On IndexTTS-1.5 the erasure is partial: BigVGAN's own speaker encoder is a
second identity path that the conditioning-path edit does not touch. Erasure is
strongest on XTTS-v2 and Tortoise-TTS, where the conditioning path is the only route
from reference audio to speaker identity.

## Configuration

All variables are optional; the defaults make the repo runnable as-is.

| Variable | Default | Purpose |
|---|---|---|
| `COND_ID_DATA_DIR` | `<repo>/data` | Root of the prepared VCTK data |
| `COND_ID_ECAPA_DIR` | `<repo>/.cache/ecapa` | Cache dir for the ECAPA checkpoint |
| `ECAPA_SOURCE` | `speechbrain/spkrec-ecapa-voxceleb` | speechbrain source id or local dir |
| `CD_EVAL_SIZE` | `20` | Speaker-subset size: `20`, `10`, or `5` |
| `CD_FORGET` | `p272` | Comma-separated forget speakers |
| `CD_RETAIN` | rest of subset | Comma-separated retain speakers |
| `CD_CPU` | unset | Set to `1` to force CPU |
| `CD_STRICT` | unset | Set to `1` for strict determinism |
| `CD_TRAIN_REF_N` | see `datasets/` | Training reference clips per speaker |
| `CD_XTTS_DIR` | `models/xtts_v2` | XTTS-v2 checkpoint directory |
| `CD_INDEXTTS_DIR` | `models/index_tts` | IndexTTS-1.5 checkpoint directory |
| `CD_FW_DEVICE` | `cpu` | Faster-Whisper device (`cuda` for speed) |
| `CD_FW_MODEL` | `large-v3` | Faster-Whisper model size |
| `VCTK_RAW_DIR` | `data/_raw` | Location of the raw VCTK download |

## Repository layout

```
cond_id/               core package (model-agnostic)
  methods/cond_id.py     the method: apply_cond_id (persistent) + applied_unlearning
  api.py                  public API: unlearn / load_edited / EditedModel
  adapter.py              save/load the conditioning-path patch (+ metadata)
  cli.py                  the `cond-id` console entry point (eval / unlearn / synth)
  hosts.py                host interface; build_host(tag)
  models/                 XTTS / Tortoise / IndexTTS host adapters
  eval/                   clone, relearn adversary, held-out scoring
  metrics/                WavLM verifier, Faster-Whisper WER, erasure
  datasets/               VCTK train / held-out clip partitions
configs/                per-model hyperparameters
scripts/                data prep, checkpoint download, run_eval, reproduce
```

## Citation

```bibtex
@misc{pujari_condid,
  title  = {cond-ID: Conditioning-Space Identity Redirection for Speaker Unlearning in Zero-Shot TTS},
  author = {Pujari, Aditya and Rattani, Ajita},
  note   = {Preprint},
}
```

See `CITATION.cff` for the machine-readable record.

## License

Code is released under the MIT License (`LICENSE`). The datasets and pretrained
checkpoints keep their own licenses — VCTK is CC-BY-4.0; **XTTS-v2 is released under
the Coqui Public Model License (CPML, non-commercial)**; review each before use. The
scripts fetch checkpoints from their original sources; unmodified, commit-pinned
mirrors are also provided for convenience
([XTTS-v2](https://huggingface.co/RootAccess4Life/cond-id-xtts-v2),
[IndexTTS-1.5](https://huggingface.co/RootAccess4Life/cond-id-indextts-1.5)) and
remain subject to their upstream licenses. See `NOTICE` for the full third-party
attribution list. The synthesized
[audio samples](https://huggingface.co/datasets/RootAccess4Life/cond-id-samples)
are derived from VCTK and are for research/demonstration only.
