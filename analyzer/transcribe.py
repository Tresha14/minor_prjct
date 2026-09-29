"""Thin wrapper around faster-whisper.

Kept deliberately small and separate from audio_features.py: this is the
part that touches the model and the GPU, so it is the part most likely to
need patching for a specific machine. audio_features.py never imports this
module, which is what keeps the feature-computation tests fast and
hardware-independent.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from analyzer.audio_features import Word
from analyzer.config import AUDIO

logger = logging.getLogger(__name__)


def _register_nvidia_dll_dirs() -> None:
    """faster-whisper's GPU path needs cuBLAS/cuDNN DLLs. We install them as
    pip packages (nvidia-cublas-cu12, nvidia-cudnn-cu12) instead of the full
    CUDA toolkit, but on Windows pip does not put their `bin/` folders on
    PATH automatically -- so we register them ourselves before the first
    CUDA call. Safe to call even if the packages aren't installed (CPU-only
    machines just skip this).
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    # `nvidia` is a namespace package (no __file__), so use __path__ instead.
    bin_dirs = []
    for root in nvidia.__path__:
        bin_dirs.extend(str(p) for p in Path(root).glob("*/bin"))

    for bin_dir in bin_dirs:
        try:
            os.add_dll_directory(bin_dir)
        except (OSError, FileNotFoundError):
            pass

    # ctranslate2's CUDA libraries are delay-loaded by its compiled extension,
    # which does not always respect os.add_dll_directory -- PATH is the
    # documented fallback (see faster-whisper's README on this exact issue).
    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")


_register_nvidia_dll_dirs()


class Transcriber:
    """Loads a faster-whisper model, preferring the GPU but never crashing
    the pipeline if the GPU path is unavailable -- it logs a warning and
    falls back to CPU instead.
    """

    def __init__(self, model_size: str = AUDIO.whisper_model,
                 device: str = AUDIO.whisper_device,
                 compute_type: str = AUDIO.whisper_compute_type):
        self.model_size = model_size
        self.device, self.compute_type = self._load_with_fallback(device, compute_type)

    def _load_with_fallback(self, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        if device == "cuda":
            try:
                self.model = WhisperModel(self.model_size, device="cuda", compute_type=compute_type)
                logger.info("Whisper '%s' loaded on GPU (%s)", self.model_size, compute_type)
                return "cuda", compute_type
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any GPU failure should fall back
                logger.warning(
                    "GPU load failed (%s: %s) -- falling back to CPU. "
                    "Live transcription will be slower; see the synopsis, Sec 6.1.",
                    type(exc).__name__, exc,
                )
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        logger.info("Whisper '%s' loaded on CPU (int8)", self.model_size)
        return "cpu", "int8"

    def transcribe_chunk(self, audio_path_or_array, offset_s: float = 0.0,
                          language: str = "en") -> list[Word]:
        """Transcribes one chunk of audio and returns word-level timestamps
        shifted by `offset_s` so they line up with the session clock.
        """
        segments, _info = self.model.transcribe(
            audio_path_or_array,
            word_timestamps=True,
            language=language,
            vad_filter=False,  # VAD is handled separately (see vad.py) so we don't double-gate
            initial_prompt="Umm, uh, so, like, you know, I think, maybe...",  # nudges Whisper to keep fillers
        )
        words: list[Word] = []
        for seg in segments:
            for w in seg.words or []:
                words.append(Word(text=w.word.strip(), start=w.start + offset_s, end=w.end + offset_s))
        return words
