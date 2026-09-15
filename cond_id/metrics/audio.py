"""Audio preprocessing helpers for the evaluation metrics."""

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


def to_16k(wav_t: torch.Tensor, sr: int) -> torch.Tensor:
    """Resample a waveform tensor to 16 kHz (the speaker-encoder rate).

    Args:
        wav_t: Waveform tensor shaped ``(..., samples)``.
        sr: Source sample rate.

    Returns:
        The waveform resampled to 16 kHz (unchanged if already 16 kHz).
    """
    if sr != 16000:
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
    return wav_t


def fix_len(wav_t: torch.Tensor, seconds: float = 3.0, sr: int = 16000) -> torch.Tensor:
    """Trim or zero-pad a waveform to a fixed duration.

    Args:
        wav_t: Waveform tensor shaped ``(..., samples)``.
        seconds: Target duration in seconds.
        sr: Sample rate of ``wav_t``.

    Returns:
        The waveform cropped or right-padded to ``seconds * sr`` samples.
    """
    tlen = int(seconds * sr)
    if wav_t.size(-1) >= tlen:
        return wav_t[..., :tlen]
    return F.pad(wav_t, (0, tlen - wav_t.size(-1)))


def load_wav_16k(path: str) -> torch.Tensor:
    """Load an audio file as a fixed-length 16 kHz mono tensor.

    Args:
        path: Path to an audio file readable by ``soundfile``.

    Returns:
        A 1-D float32 tensor at 16 kHz, length-normalized by :func:`fix_len`.
    """
    wav, sr = sf.read(str(path), dtype="float32")
    wav_t = torch.from_numpy(wav).reshape(1, -1)
    return fix_len(to_16k(wav_t, sr)).squeeze(0)
