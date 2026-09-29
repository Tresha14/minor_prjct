# Multimodal Behavioural Analysis System

A minor project: a tool that watches a **live video interview** (up to ~50 minutes) and
turns face, voice and speech signals into specific, timestamped coaching feedback —
instead of one vague "be more confident" comment.

**If you are picking this up in a fresh Claude Code session with no prior context,
read this whole file first, then see [For a new contributor / new Claude session](#for-a-new-contributor--new-claude-session)
at the bottom.**

Full design docs (more detail than this README):
- [`Project_Synopsis.docx`](Project_Synopsis.docx) — the full written synopsis (architecture, modules, evaluation plan, ethics)
- [`Project_Presentation.pptx`](Project_Presentation.pptx) — 12-slide presentation version

---

## 1. What the system actually does

- The candidate joins a normal video call (e.g. FaceTime). The interviewer's laptop
  captures the call window (video) and the system audio (candidate's voice) plus the
  interviewer's own microphone, as two **separate** channels — no speaker diarization needed.
- Three analysis modules run continuously:
  - **Video**: face/eye tracking → how much of the time the candidate faces the camera,
    head-pose stability, blink rate.
  - **Audio**: live transcript with word timestamps → pauses, words per minute, filler words.
  - **Text**: hedging language ("maybe", "I guess"), sentiment, answer relevance.
- Everything is written to a shared, timestamped **event log** (SQLite).
- A **live dashboard**, visible only to the interviewer (never the candidate), shows
  rolling indicators and alert chips during the session.
- After the session, a report gives per-question scores and a timeline of moments
  worth following up on.
- **Not for hiring/screening decisions.** This is a self-practice / coaching tool.
  See the ethics section in the synopsis.

## 2. Architecture

```
FaceTime call on the interviewer's laptop
   |  call window (virtual camera)   |  system audio (candidate)   |  microphone (interviewer)
   v                                  v                              v
 Video worker                       Audio worker (VAD + streaming Whisper)
 face/eye/head tracking             transcript, pauses, WPM, fillers
   |                                  |
   +----------------+-----------------+
                     |
     Event log (SQLite): timestamped metrics + events + question markers
          |                                   |
   Live dashboard (Streamlit,          Background scoring
   interviewer only)                   (LLM, per question)
          |                                   |
          +------------- Post-session report -+
```

Each worker (video, audio) is its own process. They never call each other directly —
they only read/write the shared event log. This means a slow step (e.g. an LLM call)
can never freeze the live dashboard.

## 3. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Best compatibility with MediaPipe / Whisper on Windows |
| Call capture | OBS Virtual Camera (video) + `soundcard` loopback (audio) | Analyse a normal video call without building our own call app |
| Video analysis | OpenCV + MediaPipe Face Landmarker | Face/gaze/head-pose landmarks, runs on CPU |
| Speech to text | faster-whisper, `small` model, GPU | Word-level timestamps, keeps up with live conversation |
| Voice activity | Silero VAD | Real-time speaking/silence detection, CPU |
| NLP | hedging lexicon + Hugging Face + LLM API (later) | Confidence language, sentiment, answer relevance |
| Fusion & stats | pandas, scikit-learn, SciPy | Rule-based scoring, correlation, agreement checks |
| Dashboard | Streamlit + Plotly | Interviewer-only live view (not built yet) |
| Storage | SQLite (`analyzer/events.py`) | Shared by all workers + the dashboard |

Full reasoning for each choice (including "why not FastAPI") is in the synopsis, Section 6.

## 4. Project structure

```
minor_project/
  analyzer/
    config.py          # every tunable number lives here (thresholds, model choice, etc.)
    events.py           DONE  the shared SQLite event log — all workers write here
    audio_features.py   DONE  pure functions: pauses, WPM, fillers, hedges (no hardware deps)
    transcribe.py        DONE  faster-whisper wrapper, GPU with automatic CPU fallback
    vad.py               DONE  Silero voice-activity detection wrapper
    audio_worker.py      DONE  live pipeline: silence-aware chunking + transcription + stats
    video_features.py     DONE  pure functions: head pose, calibrated baseline, blinks, stability
    video.py                DONE  MediaPipe FaceLandmarker wrapper (Tasks API)
    video_worker.py           DONE  live pipeline: calibration + camera-facing/pose/blink metrics
    capture.py                  DONE  live capture: webcam/loopback/mic wired into the workers
    text.py                       TODO  hedging, sentiment, answer relevance
    fusion.py                  TODO  combines everything into engagement/fluency/confidence
    feedback.py                  TODO  turns scores into coaching tips
  app.py                          TODO  Streamlit live dashboard
  run_session.py                   TODO  starts capture + worker processes
  tests/                    DONE  pytest suite — 14 tests passing, run before every commit
  evaluation/                TODO  validation, ablation, rater-agreement scripts (Week 7)
  data/sessions/               each recorded/analysed session gets its own subfolder + events.db
  requirements.txt
  Project_Synopsis.docx        the full written design doc
  Project_Presentation.pptx    12-slide presentation
```

## 5. Current status (as of this commit)

### Done and tested
- **Event log** (`analyzer/events.py`) — SQLite in WAL mode, `metric()` / `event()` / `word()`
  writers, reader methods for the dashboard. 4 tests in `tests/test_events.py`.
- **Audio feature math** (`analyzer/audio_features.py`) — pause detection, words-per-minute,
  filler counting (single- and multi-word), hedge counting. Pure functions, no dependency
  on faster-whisper or a microphone, so these tests are fast and hardware-independent.
  9 tests in `tests/test_audio_features.py`.
- **Voice activity detection** (`analyzer/vad.py`) — Silero VAD wrapper. Smoke-tested.
- **GPU transcription** (`analyzer/transcribe.py`) — faster-whisper on the GPU, confirmed
  working end to end (see "Known gotchas" below for what it took).
- **Live audio pipeline** (`analyzer/audio_worker.py`) — ties VAD + streaming Whisper +
  `audio_features.py` together: buffers incoming audio and cuts each chunk at a
  **detected silence**, not a fixed time offset (see "Known gotchas" — this came from a
  real bug found while testing, not a design guess). Writes words, WPM, pause rate,
  filler rate and long-pause events to the event log as it goes. 7 tests in
  `tests/test_audio_worker.py`, using a fake VAD + fake transcriber so they run in ~1s
  with no GPU or model needed; also confirmed against real GPU transcription of a
  spoken test clip (all 29 words recovered correctly, streamed in 1-second pieces).
- **System-audio loopback capture** — confirmed working end to end using `soundcard`:
  played a clip through the speaker, captured it via loopback, re-transcribed the capture,
  and got back the exact original words. Candidate (loopback) + interviewer (microphone)
  channels also confirmed recording **simultaneously** without conflict.
- **Plain webcam capture** via OpenCV — confirmed working as a fallback.
- **Face/head-pose tracking** (`analyzer/video.py`, `analyzer/video_features.py`) —
  MediaPipe's FaceLandmarker (Tasks API, not the older deprecated `mp.solutions`),
  giving head-pose (yaw/pitch/roll, decomposed from MediaPipe's rotation matrix via
  scipy) and blendshape scores (used for blink detection — more reliable than manually
  computing eye-aspect-ratio from raw landmarks). Confirmed working on the real webcam:
  121/121 frames detected a face over a 4-second test. 14 tests in
  `tests/test_video_features.py` — including a matrix↔euler round-trip check so the
  pose math is verified independent of the camera.
- **Live video pipeline** (`analyzer/video_worker.py`) — per-session calibration (looks
  at the camera for `VIDEO.calibration_seconds`, averages that into a baseline), then
  compares every subsequent frame against that baseline for camera-facing %, tracks a
  rolling pose-stability window, counts blinks, and raises `gaze_away` / `face_lost`
  events. 6 tests in `tests/test_video_worker.py` using a fake tracker (fast, no
  camera). Confirmed end-to-end on the real webcam: calibration completed correctly,
  camera-facing/pose-stability/blink metrics all came out with sane values, no crashes
  over an 8-second live run.
- **Live capture** (`analyzer/capture.py`) — `SessionCapture` wires a real webcam,
  real loopback audio (candidate) and a real microphone (interviewer) into
  `VideoWorker` and `AudioWorker` continuously, each source running in its own
  background thread. Confirmed working end-to-end: played the 29-word test clip through
  the speaker while the webcam ran and `SessionCapture` captured both — **all 29 words
  came out correct**, camera-facing/pose/blink metrics were written throughout, and the
  interviewer channel's speaking-fraction metric worked too. 9 tests in
  `tests/test_capture.py` (pure logic — mono conversion, interval batching, and the
  threading/queue behavior described below — no camera or mic needed to run them).
  Found and fixed a real bug in the process — see "Known gotchas".
- 49/49 tests passing.

### Not started yet
- **Call-window video capture via OBS Virtual Camera.** `capture.py`'s `VideoSource`
  works with any camera device index, so once OBS is installed its virtual camera can
  be used as-is — nothing in the code needs to change, just point `camera_index` at it
  (see `analyzer/capture.py`'s docstring). **Blocked: OBS Studio is not installed on
  the dev machine yet.** Until it is, `capture.py` has been developed and validated
  against a plain webcam instead. **This is the natural next task**, alongside
  `run_session.py` (a real entry point that starts a `SessionCapture` and runs the
  Streamlit dashboard against it — right now `SessionCapture` is only driven by an
  ad-hoc test script).
- `analyzer/text.py` — hedging/sentiment/answer-relevance (Week 4 in the plan).
- `analyzer/video.py` — MediaPipe face/gaze/head-pose tracking (Week 3 in the plan).
- `analyzer/fusion.py`, `analyzer/feedback.py`, `app.py` (Streamlit dashboard),
  `run_session.py` — all later weeks.
- `evaluation/` scripts, real dataset collection — Week 7.

### Known gotchas (read before touching audio/GPU code)
- **A slow consumer must never run on the same thread as live capture.** The first
  version of `SessionCapture` called `AudioWorker.push_audio()` directly from the
  audio-capture thread. `push_audio()` triggers GPU transcription synchronously, which
  can take a second or more — for that whole time, the thread wasn't reading from the
  microphone either, so the OS-level audio buffer overflowed and **real words were
  permanently dropped**, not just delayed (confirmed by testing: word count went from
  29/29 to 20/29 with this bug present). The fix is `capture.py`'s `QueuedConsumer`:
  the capture thread just enqueues audio and returns immediately; a separate thread
  drains the queue and does the slow transcription. If you ever add another
  slow/blocking step to a live pipeline, it needs the same pattern — a fast producer
  thread must never call directly into a slow handler.
- **Unthrottled video capture competes for CPU with everything else.** A related fix:
  `VideoSource` supports a `target_fps` throttle (so MediaPipe doesn't run on every raw
  camera frame, ~30fps), but `SessionCapture` wasn't actually passing
  `config.VIDEO.target_fps` into it — an oversight caught by the same test that found
  the threading bug above. Both fixes together were needed to get a clean 29/29 words.
- **Chunk boundaries must land on detected silence, never a fixed time offset.** An
  earlier version of `audio_worker.py` cut every 8 seconds with a 1-second overlap and
  de-duplicated by timestamp. Tested against real GPU transcription of a spoken clip,
  it silently **dropped several words at the boundary** — not garbled, just missing.
  Cause: a chunk that starts mid-word, with no leading silence for acoustic context,
  doesn't get mis-transcribed by Whisper, it gets **skipped outright**. The fix
  (`_find_cut_point` in `audio_worker.py`) uses the VAD to only cut right after a
  confirmed silence, waiting up to `AUDIO.max_chunk_s` if the speaker talks straight
  through the target chunk length. A second, subtler version of the same bug: a VAD
  run on a not-yet-complete buffer will report speech "ending" exactly at the buffer's
  current edge even when the speaker is still mid-sentence and more audio just hasn't
  arrived yet — trusting that as a cut point re-introduces word loss. The fix requires
  a bit of *confirmed* trailing silence (`_cut_margin_s`) inside the buffer before a
  candidate cut point is trusted. Both failure modes have regression tests in
  `tests/test_audio_worker.py` — don't change the cutting logic without keeping them
  green.
- **`compute_type="int8_float16"` hangs (not errors) on the dev GPU (GTX 1650, 4GB).**
  It silently freezes instead of raising an exception. Use `"int8"` — it's confirmed
  fast and reliable (model load ~4.5s, transcribed a 15s clip in 1.9s). This is set as
  the default in `analyzer/config.py`; don't change it without re-testing under a hard
  timeout (a hung process won't show an error, it'll just never return).
- **GPU transcription needs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`** (installed via
  pip, not the full CUDA toolkit). `analyzer/transcribe.py` registers their DLL folders
  on `PATH` automatically at import time — `os.add_dll_directory()` alone was **not**
  enough because ctranslate2 delay-loads its CUDA dependencies. If you ever see
  `cublas64_12.dll is not found`, it means this registration didn't run (e.g. you
  imported ctranslate2 directly instead of via `analyzer.transcribe`) or the packages
  aren't installed.
- **`soundcard` loopback recording logs "data discontinuity" warnings** with small
  `numframes` reads. Not blocking, but `audio_worker.py` should use a larger buffer or
  a dedicated capture thread so a real ~50-minute session doesn't drop audio.
- Two OpenCV packages get installed (`opencv-python` + `opencv-contrib-python`, the
  second pulled in by MediaPipe). Harmless so far; if `cv2` starts behaving oddly,
  that's the first thing to check.
- **The face-tracking model file is not in git** (`analyzer/models/face_landmarker.task`,
  ~3.7 MB of binary weights) — it's gitignored and every dev downloads their own copy.
  See the Setup section below. `analyzer/video.py`'s `FaceTracker` raises a clear
  `FileNotFoundError` (not a cryptic MediaPipe error) if it's missing.
  `mediapipe==1.0.1` uses the newer Tasks API (`mediapipe.tasks.python.vision`), not the
  older `mp.solutions.face_mesh` — if you find older MediaPipe example code online, it
  likely uses the deprecated API and won't match this codebase.
- **`video_features.py`'s yaw/pitch/roll sign convention hasn't been independently
  verified against MediaPipe's own convention** (e.g. "does yaw actually increase when
  you turn right") — the round-trip math is tested and correct, but that only proves
  `matrix_to_euler` and `euler_to_matrix` are exact inverses of each other, not what a
  positive number *means* physically. This matters less than it sounds because every
  comparison in the project is against a per-session calibrated `Baseline`, not an
  absolute "0 = facing camera" assumption — but worth a quick manual check (turn your
  head right, print the yaw, confirm it moves the direction you'd expect) before
  trusting the raw numbers in a debugger.

## 6. Setup

```bash
git clone <this repo>
cd minor_project
python -m venv .venv
```
```bash
# Windows (Git Bash / PowerShell)
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Download the face-tracking model (not committed to git — see "Known gotchas"):
```bash
mkdir -p analyzer/models
curl -L -o analyzer/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Run the tests to confirm the setup works:
```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
```
You should see `41 passed`. If GPU-related tests are added later and fail, that's a
hardware/driver issue on your machine specifically — CPU fallback should still work
(`analyzer/transcribe.py` falls back automatically and logs a warning). None of the
current 41 tests actually require a GPU, webcam or microphone — they use fake
transcriber/VAD/tracker objects, so they should pass identically on any machine.

## 7. Roadmap (8 weeks, from the synopsis)

| Week | Milestone | Status |
|---|---|---|
| 1 | Feasibility test: call capture + GPU transcription + face tracking together | **Audio: done.** Video: blocked on OBS install |
| 2 | Audio worker: VAD + streaming transcript + pauses/WPM/fillers | **Done** (`analyzer/audio_worker.py`, tested against real audio) |
| 3 | Video worker: gaze, head pose, blinks, calibration | **Done** (`analyzer/video_worker.py`, tested against a real webcam) |
| 4 | Event log + live Streamlit dashboard | Event log **and live capture** done; Streamlit dashboard not started |
| 5 | Fusion layer: rule-based scoring | Not started |
| 6 | Feedback generator + background LLM scoring | Not started |
| 7 | Data collection + evaluation (validation, rater agreement, ablation) | Not started |
| 8 | Polish, demo rehearsal, final report | Not started |

**We're ahead of the week-by-week schedule** (Weeks 2, 3, and the event-log half of
Week 4 all done) — the remaining open item from Week 1, call-window video capture, is
blocked on installing OBS Studio, not on anything technical. The next natural task is
the Streamlit dashboard (`app.py`), reading live from `SessionCapture`'s event log.

If behind schedule: drop background LLM scoring first, then the live transcript
(keep live video + voice-activity metrics; transcribe after the session instead).

## 8. Development notes

- **Run `pytest tests/ -q` before pushing.** Keep it green.
- **Put every tunable number in `analyzer/config.py`**, not scattered through modules —
  this is deliberate so a viva question like "why 0.7 seconds for a pause" has one
  place to point to.
- `analyzer/audio_features.py` is pure logic with no hardware dependency, by design —
  keep it that way so it stays fast to test. Hardware-touching code goes in
  `transcribe.py`, `vad.py`, `capture.py`, `video.py`.
- Ethics: the live dashboard is interviewer-only, never shown to the candidate.
  Keep that boundary when building `app.py`.

---

## For a new contributor / new Claude session

If you're starting a fresh Claude Code session on this repo with no prior context,
paste this to Claude:

> Read `README.md` in full, then read `analyzer/config.py`, `analyzer/events.py`,
> `analyzer/audio_features.py`, `analyzer/transcribe.py`, `analyzer/vad.py`,
> `analyzer/audio_worker.py`, `analyzer/video_features.py`, `analyzer/video.py`,
> `analyzer/video_worker.py` and `analyzer/capture.py` to see what's already built and
> tested — pay particular attention to the "Known gotchas" section and the comments in
> `audio_worker.py` and `capture.py`, which explain two real bugs (dropped words from a
> bad chunking design, then dropped words again from a threading bug) that were found
> by testing and fixed. Then run:
>   `mkdir -p analyzer/models && curl -L -o analyzer/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
> (the face-tracking model isn't committed to git) and
> `./.venv/Scripts/python.exe -m pytest tests/ -q` to confirm the setup works (expect
> 49 passed — none of these need a camera, mic or GPU, they all use fakes). Then help
> me build the Streamlit dashboard (`app.py`): a live, interviewer-only view that
> starts a `SessionCapture` (see `analyzer/capture.py`) and polls its `EventLog` every
> 1-2 seconds to show rolling camera-facing/WPM/pause/filler/blink indicators, a
> timeline of flagged events (`long_pause`, `gaze_away`, `face_lost`), and a way to
> mark question boundaries. Ask me first whether OBS Studio is installed yet — if it
> is, `SessionCapture`'s video source can point at the OBS virtual camera device
> instead of the plain webcam it's been validated against so far.
