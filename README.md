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
    capture.py            TODO  call-window video + system-audio loopback capture
    video.py               TODO  MediaPipe face/gaze/head-pose tracking
    text.py                  TODO  hedging, sentiment, answer relevance
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
- 21/21 tests passing.

### Not started yet
- `analyzer/capture.py` — call-window video capture via **OBS Virtual Camera**, plus
  wiring the confirmed-working `soundcard` loopback/mic capture into a real live feed
  for `audio_worker.py` (so far it's been fed pre-recorded clips in testing, not a
  live stream — the chunking logic doesn't care which, but this hasn't been run
  against a truly live mic yet). **Blocked: OBS Studio is not installed on the dev
  machine yet.** Until it is, video work can proceed against a plain webcam (confirmed
  working) as a stand-in. **This is the natural next task.**
- `analyzer/video.py` — MediaPipe face/gaze/head-pose tracking (Week 3 in the plan).
- `analyzer/fusion.py`, `analyzer/feedback.py`, `app.py` (Streamlit dashboard),
  `run_session.py` — all later weeks.
- `evaluation/` scripts, real dataset collection — Week 7.

### Known gotchas (read before touching audio/GPU code)
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

Run the tests to confirm the setup works:
```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
```
You should see `14 passed`. If GPU-related tests are added later and fail, that's a
hardware/driver issue on your machine specifically — CPU fallback should still work
(`analyzer/transcribe.py` falls back automatically and logs a warning).

## 7. Roadmap (8 weeks, from the synopsis)

| Week | Milestone | Status |
|---|---|---|
| 1 | Feasibility test: call capture + GPU transcription + face tracking together | **Audio: done.** Video: blocked on OBS install |
| 2 | Audio worker: VAD + streaming transcript + pauses/WPM/fillers | **Done** (`analyzer/audio_worker.py`, tested against real audio) |
| 3 | Video worker: gaze, head pose, blinks, calibration | Not started |
| 4 | Event log + live Streamlit dashboard | Event log done; dashboard not started |
| 5 | Fusion layer: rule-based scoring | Not started |
| 6 | Feedback generator + background LLM scoring | Not started |
| 7 | Data collection + evaluation (validation, rater agreement, ablation) | Not started |
| 8 | Polish, demo rehearsal, final report | Not started |

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
> `analyzer/audio_features.py`, `analyzer/transcribe.py`, `analyzer/vad.py` and
> `analyzer/audio_worker.py` to see what's already built and tested — pay particular
> attention to the "Known gotchas" section and the comments at the top of
> `audio_worker.py`, which explain a real word-dropping bug that was found and fixed
> there. Run `./.venv/Scripts/python.exe -m pytest tests/ -q` to confirm the setup
> works (expect 21 passed). Then help me build `analyzer/capture.py`: call-window
> video capture (via OBS Virtual Camera, once it's installed) and system-audio
> loopback + microphone capture using `soundcard` (already confirmed working
> end-to-end in manual testing — see the README status section for how). It needs to
> feed live audio into `AudioWorker.push_audio()` continuously. Ask me first whether
> OBS Studio is installed yet; if not, start with a plain webcam via OpenCV as a
> stand-in for video, since that's already confirmed working too.
