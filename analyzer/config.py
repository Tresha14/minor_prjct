"""Central, editable settings for the whole pipeline.

Keep every tunable number here instead of scattering magic numbers through
the modules, so a viva question like "why 0.7 seconds for a pause" has one
place to point to.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000          # Whisper and Silero VAD both expect 16 kHz mono
    whisper_model: str = "small"       # fits comfortably in 4 GB VRAM (see synopsis, Sec 6.1)
    whisper_device: str = "cuda"       # falls back to "cpu" automatically if CUDA is unavailable
    whisper_compute_type: str = "int8"  # int8_float16 hung on this GTX 1650 during testing; int8 is confirmed working
    pause_threshold_s: float = 0.7     # gap between words counted as a "pause"
    long_pause_threshold_s: float = 3.0  # gap that gets flagged as a notable hesitation
    chunk_seconds: float = 8.0         # length of each rolling audio chunk sent to Whisper
    chunk_overlap_s: float = 1.0       # overlap between chunks so words at the boundary aren't lost


@dataclass(frozen=True)
class VideoConfig:
    target_fps: float = 10.0           # analysis frame rate; the source video can be higher
    calibration_seconds: float = 8.0   # "look at the camera" window used to set the baseline
    ear_blink_threshold: float = 0.21  # eye-aspect-ratio below this counts as an eye closure


@dataclass(frozen=True)
class TextConfig:
    hedge_words: tuple = (
        "maybe", "i guess", "i think", "sort of", "kind of", "probably",
        "not sure", "perhaps", "possibly", "i suppose", "might be", "could be",
    )
    filler_words: tuple = (
        "um", "uh", "umm", "uhh", "erm", "hmm", "like", "you know", "so yeah",
    )


@dataclass(frozen=True)
class Paths:
    data_dir: str = "data/sessions"
    db_filename: str = "events.db"


AUDIO = AudioConfig()
VIDEO = VideoConfig()
TEXT = TextConfig()
PATHS = Paths()
