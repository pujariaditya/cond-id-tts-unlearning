"""Intelligibility metric: word error rate via Faster-Whisper.

Used to verify that unlearning preserves retained-speaker speech: a method that
silences or garbles audio could trivially drop speaker similarity, so retain WER
acts as an anti-collapse gate.
"""

import torch

_FASTER_WHISPER = None
_WER_TRANSFORM = None

# Pin to an exact revision for reproducibility before a public release. Faster-Whisper
# resolves ``large-v3`` to the CTranslate2 repo ``Systran/faster-whisper-large-v3``;
# ``None`` tracks the latest upload (not reproducible).
_FW_REVISION = None  # TODO(release): pin to the evaluated Systran/faster-whisper-large-v3 commit


def _ensure_cudnn_on_path():
    """Add the pip-installed cuDNN libraries to ``LD_LIBRARY_PATH`` if present.

    CTranslate2 (the Faster-Whisper backend) needs cuDNN 9 for GPU inference. When
    cuDNN is provided by the ``nvidia-cudnn-cu12`` pip package its libraries live
    under ``site-packages/nvidia/cudnn/lib``; expose that directory portably (no
    machine-specific path). A no-op if the package is absent or cuDNN is already
    on the system path.
    """
    import os

    try:
        import nvidia.cudnn
    except Exception:
        return
    # ``__file__`` is None when nvidia.cudnn is an implicit namespace package; fall
    # back to ``__path__``. No-op if neither resolves.
    f = getattr(nvidia.cudnn, "__file__", None)
    if f:
        pkgdir = os.path.dirname(f)
    else:
        paths = list(getattr(nvidia.cudnn, "__path__", []) or [])
        pkgdir = paths[0] if paths else None
    if not pkgdir:
        return
    libdir = os.path.join(pkgdir, "lib")
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if os.path.isdir(libdir) and libdir not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{cur}" if cur else libdir


def load_faster_whisper():
    """Load the Faster-Whisper ASR model once (cached).

    Model id (``CD_FW_MODEL``, default ``large-v3``) and device
    (``CD_FW_DEVICE``, default ``cpu``) are read from the environment. CPU is the
    default so WER works on any machine without the cuDNN 9 CTranslate2 build. For
    faster transcription set ``CD_FW_DEVICE=cuda`` AND launch with the pip cuDNN on
    ``LD_LIBRARY_PATH`` (``python -c "import nvidia.cudnn,os;print(os.path.dirname(
    nvidia.cudnn.__file__)+'/lib')"``) -- the linker must see it at process start.
    """
    global _FASTER_WHISPER
    if _FASTER_WHISPER is None:
        import os

        device = os.environ.get("CD_FW_DEVICE", "cpu")
        if device == "cuda":
            _ensure_cudnn_on_path()
        from faster_whisper import WhisperModel

        model_name = os.environ.get("CD_FW_MODEL", "large-v3")
        rev = {} if _FW_REVISION is None else {"revision": _FW_REVISION}
        _FASTER_WHISPER = WhisperModel(
            model_name, device=device, compute_type="float16" if device == "cuda" else "int8", **rev
        )
    return _FASTER_WHISPER


def transcribe(wav_16k: torch.Tensor) -> str:
    """Transcribe a 16 kHz waveform tensor to lowercase text.

    Args:
        wav_16k: 1-D float32 waveform at 16 kHz.

    Returns:
        The decoded transcript (a single space if empty, to keep WER defined).
    """
    fw = load_faster_whisper()
    segments, _ = fw.transcribe(
        wav_16k.detach().cpu().numpy().astype("float32"), language="en", beam_size=5
    )
    return " ".join(s.text.strip() for s in segments).strip() or " "


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Compute word error rate between reference and hypothesis transcripts.

    Args:
        references: Ground-truth sentences.
        hypotheses: Predicted transcripts (same length as ``references``).

    Returns:
        The aggregate WER as a float.
    """
    global _WER_TRANSFORM
    import jiwer

    if _WER_TRANSFORM is None:
        _WER_TRANSFORM = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
    return float(
        jiwer.wer(
            references,
            hypotheses,
            truth_transform=_WER_TRANSFORM,
            hypothesis_transform=_WER_TRANSFORM,
        )
    )
