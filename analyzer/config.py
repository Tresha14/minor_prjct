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

    # Chunking: cuts are made at detected silence (via Silero VAD), not at a
    # fixed time. A fixed-time cut risks slicing mid-word, which we found
    # (by testing, not by guessing) makes Whisper silently DROP the speech
    # right after the cut rather than mis-transcribe it -- see
    # tests/test_audio_worker.py and README.md "Known gotchas".
    target_chunk_s: float = 8.0        # aim to cut once buffered audio passes this length
    max_chunk_s: float = 15.0          # force a cut here even mid-speech, so one long
                                        # monologue can't stall transcription indefinitely
    min_chunk_s: float = 2.0           # never cut a chunk shorter than this
    min_speech_fraction: float = 0.05  # skip transcribing a chunk if less than this fraction
                                        # of it is detected speech -- Whisper hallucinates short
                                        # phrases ("You", "Thank you") on near-silent audio
                                        # otherwise; found by testing against a live mic with
                                        # only ambient room noise. See README.md "Known gotchas".


@dataclass(frozen=True)
class VideoConfig:
    target_fps: float = 10.0           # analysis frame rate; the source video can be higher
    calibration_seconds: float = 8.0   # "look at the camera" window used to set the baseline
    blink_score_threshold: float = 0.5  # MediaPipe eyeBlinkLeft/Right blendshape score (0-1,
                                         # 1=fully closed) above this counts as an eye closure
    facing_yaw_tolerance_deg: float = 15.0   # allowed drift from the calibrated baseline yaw
    facing_pitch_tolerance_deg: float = 15.0  # allowed drift from the calibrated baseline pitch
    pose_window_s: float = 3.0         # rolling window used to compute head-pose stability
    metric_interval_s: float = 2.0     # how often camera_facing/pose_stability/blink_rate are written
    gaze_away_threshold: float = 0.3   # camera_facing fraction below this over an interval -> flagged event
    face_lost_min_s: float = 1.0       # minimum gap with no face detected before it's logged as an event
    model_path: str = "analyzer/models/face_landmarker.task"


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
