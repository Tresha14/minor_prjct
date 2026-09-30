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

    # Lexicon-based sentiment (analyzer/text.py) -- rule-based like every
    # other scoring layer in this project, for the same reason fusion.py
    # gives for not training a model: too little data collected here to
    # fine-tune or trust a model responsibly, and a word-list score is
    # transparent and defensible in a viva. Interview-context word choices,
    # not a general-purpose sentiment lexicon.
    positive_words: tuple = (
        "confident", "excited", "enthusiastic", "prepared", "comfortable",
        "motivated", "capable", "skilled", "successful", "effective",
        "strong", "passionate", "proud", "accomplished", "eager", "great",
        "good", "positive", "improved", "achieved",
    )
    negative_words: tuple = (
        "nervous", "unsure", "worried", "confused", "difficult", "struggled",
        "weak", "uncomfortable", "hesitant", "doubt", "anxious", "poor",
        "negative", "failed", "afraid", "stressed", "overwhelmed", "bad",
        "mistake", "problem",
    )

    # Stopwords excluded before comparing question/answer word overlap
    # (analyzer/text.py's answer_relevance) -- without this, two unrelated
    # sentences that both happen to use "the", "a", "is" etc. would look
    # artificially relevant.
    stopwords: tuple = (
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "at", "for", "and", "or", "but", "with",
        "as", "by", "that", "this", "these", "those", "it", "its", "i",
        "you", "he", "she", "they", "we", "do", "does", "did", "have",
        "has", "had", "will", "would", "should", "could", "can", "may",
        "might", "not", "no", "so", "if", "then", "than", "what", "which",
        "who", "whom", "how", "why", "when", "where", "there", "here",
        "your", "my", "our", "their",
    )


@dataclass(frozen=True)
class Paths:
    data_dir: str = "data/sessions"
    db_filename: str = "events.db"


@dataclass(frozen=True)
class FusionConfig:
    """Reference ranges used to turn raw metrics into 0-1 "how good is
    this" scores before they're combined (see analyzer/fusion.py).

    These numbers come from general public-speaking / conversational-speech
    reference points (typical conversational English is roughly 120-150
    words per minute; average resting blink rate is roughly 15-20/min) and
    from engineering judgement -- NOT from this project's own collected
    data. Week 7 of the plan (synopsis Sec 8) validates these against real
    human ratings and recalibrates if the data supports it. Until then,
    every fusion score is a transparent first approximation, not a
    validated measurement -- state that plainly if asked in a viva.
    """
    # Speaking rate (words per minute): comfortable conversational band.
    # Penalised further away in either direction -- too slow reads as
    # hesitant, too fast reads as rushed.
    wpm_floor: float = 60.0
    wpm_good_low: float = 110.0
    wpm_good_high: float = 165.0
    wpm_ceiling: float = 220.0

    # Pause rate: fraction of a segment spent in pauses >= pause_threshold_s.
    pause_rate_good: float = 0.15   # at/below this, no penalty
    pause_rate_bad: float = 0.45    # at/above this, fully penalised

    # Filler words per 100 words.
    filler_rate_good: float = 1.0
    filler_rate_bad: float = 8.0

    # Hedge phrases per 100 words. Not produced by any module yet
    # (analyzer/text.py is Week 6) -- these thresholds exist now so
    # fusion.py's shape doesn't need to change once it is.
    hedge_rate_good: float = 1.0
    hedge_rate_bad: float = 10.0

    # Head-pose stability: mean std-dev of yaw/pitch/roll (degrees) over
    # the rolling window (analyzer/video_features.py's pose_stability) --
    # lower means steadier.
    pose_stability_good: float = 2.0
    pose_stability_bad: float = 12.0

    # Blink rate: normal resting range is roughly 15-20/min; well outside
    # it in either direction is a weaker, secondary signal (see synopsis
    # Sec 5.2 -- blink rate is deliberately treated as low-weight below).
    blink_floor: float = 2.0
    blink_good_low: float = 10.0
    blink_good_high: float = 25.0
    blink_ceiling: float = 40.0

    # Sub-score weights: relative, not required to sum to 1 -- fusion.py's
    # weighted_average() renormalises over whichever inputs are actually
    # available, so a metric that isn't wired up yet (e.g. hedge_rate) is
    # simply dropped rather than guessed at.
    engagement_weights: dict = None
    fluency_weights: dict = None
    confidence_weights: dict = None

    def __post_init__(self):
        # frozen dataclass + mutable dict defaults -> set them here instead
        # of in the field defaults (a bare `dict` default would be shared
        # across instances).
        object.__setattr__(self, "engagement_weights", {
            "camera_facing": 0.45,
            "pause_rate": 0.20,
            "wpm": 0.20,
            "pose_stability": 0.15,
        })
        object.__setattr__(self, "fluency_weights", {
            "filler_rate": 0.40,
            "pause_rate": 0.40,
            "wpm": 0.20,
        })
        object.__setattr__(self, "confidence_weights", {
            "pause_rate": 0.30,
            "camera_facing": 0.30,
            "blink_rate": 0.15,
            "hedge_rate": 0.25,
        })


@dataclass(frozen=True)
class FeedbackConfig:
    """Thresholds controlling which raw metrics turn into a worded coaching
    tip (see analyzer/feedback.py). Deliberately reuses the same 0-1
    normalised "how good is this" score fusion.py already computes from
    FUSION's good/bad ranges above, rather than introducing a second set of
    raw thresholds -- so a tip's trigger point can never drift out of sync
    with what actually moved the fusion score for the same metric.
    """
    needs_work_max: float = 0.40   # normalised score at/below this -> a coaching tip is generated
    praise_min: float = 0.85       # normalised score at/above this -> a positive tip is generated
    max_tips: int = 5              # cap on how many tips generate_tips() returns at once


@dataclass(frozen=True)
class FlagConfig:
    """Thresholds for "several signals deviated together" flags
    (analyzer/flags.py). All comparisons are made in fusion.py's 0-1 "how
    good is this" space, so they line up with the scores by construction.
    """
    window_s: float = 30.0             # length of the window checked for co-occurring deviations
    step_s: float = 15.0               # windows overlap by half; overlapping flagged windows are merged
    norm_drop: float = 0.25            # a signal "deviates" when the window is at least this much worse than the speaker's OWN session median...
    norm_max: float = 0.60             # ...AND is itself on the needs-work side (so a consistently great speaker isn't flagged for noise)
    min_signals: int = 2               # this many deviating signals in one window -> flagged
    min_baseline_readings: int = 3     # need at least this many readings to trust a session median; otherwise that signal can't fire


AUDIO = AudioConfig()
VIDEO = VideoConfig()
TEXT = TextConfig()
PATHS = Paths()
FUSION = FusionConfig()
FEEDBACK = FeedbackConfig()
FLAGS = FlagConfig()
