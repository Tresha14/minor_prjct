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
    dashboard_data.py             DONE  pure formatting/aggregation helpers for app.py
    text.py                         TODO  hedging, sentiment, answer relevance
    fusion.py                     DONE  rule-based engagement/fluency/confidence scoring
    feedback.py                     TODO  turns scores into coaching tips
  app.py                            DONE  Streamlit live dashboard (interviewer-only)
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
- **Live dashboard** (`app.py`, `analyzer/dashboard_data.py`) — a Streamlit page for
  the interviewer: Start/Stop session, a "mark next question" button, live metric
  tiles (camera-facing %, WPM, pause rate, filler rate, blink rate) that refresh every
  1.5s via `st.fragment`, rolling charts, a timeline of flagged moments, and the live
  transcript. `dashboard_data.py` holds the pure formatting/aggregation logic (13
  tests, no Streamlit/camera/mic needed); `app.py` itself has 3 tests using
  Streamlit's own `AppTest` framework, checked in its idle state only (starting a real
  session needs real hardware, so that path is checked manually — see below).
  **Actually run end-to-end via a live browser** (not just imported): started the
  server, clicked Start session, played the 29-word test clip through the speaker
  while the webcam ran, and watched it work — correct live transcript, correct WPM
  (149) and filler count (4.8/100 words, matching the clip's one "Umm"), only one
  genuine flagged pause, camera-facing chart updating, Stop session tore everything
  down cleanly with zero errors in the server log, and the session data was confirmed
  saved to disk afterward. **Found and fixed two more real bugs doing this** — see
  "Known gotchas": a frozen elapsed-time display, and Whisper hallucinating phrases on
  near-silent audio.
- **Fusion / scoring layer** (`analyzer/fusion.py`) — rule-based, not a trained model
  (the dataset is far too small to train one responsibly; see the synopsis Sec 5.5 and
  Sec 8). Turns the metrics already in the event log into three 0–100 sub-scores —
  engagement, fluency, confidence — using two rules enforced throughout: a metric that
  hasn't been observed yet is **dropped from the average**, never guessed at, and if
  every input to a sub-score is missing, that sub-score is **`None`, not a fabricated
  default**. Every score is traceable back to exactly which metrics fed it (see
  `SubScore.drivers`) — this is surfaced directly in the dashboard's "How these scores
  were built" expander, not just in code. Tested unusually heavily for this project,
  on purpose: hand-calculated "golden value" checks (not just example-based sanity
  checks), monotonicity sweeps (e.g. more camera-facing time can never lower
  engagement — checked by sweeping the whole 0–1 range and asserting the score never
  decreases), and a config sanity check that fails loudly if a weight dict in
  `config.py` ever references a metric name that doesn't exist (catches typos that
  would otherwise silently contribute nothing to a score, forever, with no error). 27
  tests in `tests/test_fusion.py`, including a `coverage` regression test built from
  the exact readings of a real "confidence looked confident with zero speech data"
  bug found by a user (see "Known gotchas"). **Verified live, not just unit-tested:** hand-computed
  the expected engagement/fluency/confidence scores from the dashboard's own live
  readings during a browser test, and the displayed scores matched the hand
  calculation exactly (down to driver-by-driver percentages), proving the whole chain
  — event log → `compute_from_log` → `compute_scores` → dashboard tiles — is wired
  correctly end to end in the actual running app.
- 95/95 tests passing.

### Not started yet
- **Call-window video capture via OBS Virtual Camera.** `capture.py`'s `VideoSource`
  works with any camera device index, so once OBS is installed its virtual camera can
  be used as-is — nothing in the code needs to change, just point `camera_index` at it
  (see `analyzer/capture.py`'s docstring). **Blocked: OBS Studio is not installed on
  the dev machine yet.** Until it is, `capture.py`/`app.py` have been developed and
  validated against a plain webcam instead. **This is the natural next task.**
- `analyzer/text.py` — hedging/sentiment/answer-relevance (Week 6 in the plan). Once
  built, its `hedge_rate` metric plugs straight into `fusion.py`'s confidence score
  with no changes needed there — the weight is already configured, just currently
  dropped because nothing produces the metric yet.
- `analyzer/feedback.py` — turning fusion scores into worded coaching tips (Week 6).
- `evaluation/` scripts, real dataset collection, validating/recalibrating
  `fusion.py`'s reference ranges against real human ratings — Week 7.

### Known gotchas (read before touching audio/GPU code)
- **A score built from half its inputs must never look like a complete one — always
  show coverage, not just the number.** A real user testing solo saw Confidence = 67
  with zero words spoken as far as the system could tell. The number itself was
  computed correctly (missing inputs are dropped, per `fusion.py`'s design), but a
  score built from only 2 of its 4 configured signals (camera_facing + blink_rate;
  pause_rate had no audio, hedge_rate doesn't exist yet) rendered as a plain "67" —
  visually indistinguishable from a fully-informed score. Fixed by adding
  `SubScore.coverage` (fraction of configured weight actually present) and showing it
  next to every score tile in `app.py`, always, not just when it happens to be low —
  the whole point is that a thin score can never again look complete by accident.
  Also added an in-session warning banner (`app.py`) that fires if no candidate speech
  has been detected after 15s, pointing directly at the solo-test-mode checkbox — the
  same root cause as the gotcha below, but this time surfaced *during* a running
  session instead of only on the idle screen, since that's when someone would
  actually notice something's wrong.
- **Calibration must be visible to whoever's on camera, or it silently ruins the
  session.** `VideoWorker`'s first `calibration_seconds` (~8s) become the "normal"
  baseline that everything else is compared against — but that window used to run with
  zero on-screen indication. Found by an actual user test: the dashboard showed
  near-constant "looking away from camera" and "not visible in frame" for almost the
  whole session, which looked like the system was broken. It wasn't — the person being
  tracked had no way to know those first 8 seconds mattered, so their baseline got set
  from whatever position they happened to be in (most likely looking at the dashboard,
  not the lens), and every bit of normal posture afterward read as a deviation from it.
  Fixed with `VideoWorker.calibration_seconds_remaining()` plus a prominent warning
  banner + progress bar in `app.py` telling the person exactly when to look at the
  camera and for how long. **If you ever see near-100% "looking away" events again,
  check whether the person looked at the camera during calibration before assuming the
  scoring math is wrong.**
- **"No speech transcribed" during a solo test is by design, not a bug.** The candidate
  (transcribed) channel is wired to speaker loopback — simulating a candidate's voice
  coming over a call — while a lone tester's own microphone is wired to the interviewer
  channel, which is only tracked for speaking presence and never transcribed. Testing
  alone by just talking into your own mic will correctly show an empty transcript.
  `SessionCapture(..., solo_test_mode=True)` (exposed as a checkbox in `app.py`, off by
  default) points the transcribed channel at your own microphone instead, purely for
  testing/demos — never use it for an actual interview, since it skips the interviewer
  channel entirely.
- **Whisper hallucinates short phrases on near-silent audio — gate transcription on
  detected speech, don't just gate the cut point.** Live-tested `app.py` against a real
  microphone with only ambient room noise (no one speaking) and the transcript showed
  "You you" — a well-known Whisper failure mode, not a bug in our code, but one that
  would otherwise pollute the live transcript and skew WPM/pause metrics as if someone
  had spoken. The VAD was already used to choose *where* to cut a chunk
  (`audio_worker.py`'s `_find_cut_point`), but nothing stopped a chunk from being sent
  to Whisper at all if it turned out to be almost entirely silence. Fixed with
  `AudioWorker._has_enough_speech()`: skip the Whisper call entirely (and log nothing)
  if less than `config.AUDIO.min_speech_fraction` (5%) of the chunk is detected speech.
  Re-tested live afterward with real speech playing: correct transcript, no
  hallucinated phrases, and only genuine pauses got flagged. Regression tests in
  `tests/test_audio_worker.py` cover both "near-silent chunk never reaches the
  transcriber" and "a normal chunk with some genuine pauses still does."
- **A Streamlit `st.fragment` only refreshes what's written inside it.** `app.py`'s
  elapsed-time caption was originally a plain `st.caption()` placed *above* the
  `@st.fragment(run_every=1.5)` block — it looked frozen at "00:00 elapsed" during live
  testing because nothing outside a fragment reruns on the fragment's own timer, only
  on a full-page rerun (i.e. a button click). Moved inside the fragment; confirmed
  ticking correctly in the same live test that caught it.
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
You should see `95 passed`. If GPU-related tests are added later and fail, that's a
hardware/driver issue on your machine specifically — CPU fallback should still work
(`analyzer/transcribe.py` falls back automatically and logs a warning). None of the
current 91 tests actually require a GPU, webcam or microphone — they use fake
transcriber/VAD/tracker/AppTest objects, so they should pass identically on any machine.

To actually see the live dashboard (needs a webcam and speakers/mic on your machine):
```bash
./.venv/Scripts/streamlit.exe run app.py
```

## 7. Roadmap (8 weeks, from the synopsis)

| Week | Milestone | Status |
|---|---|---|
| 1 | Feasibility test: call capture + GPU transcription + face tracking together | **Audio: done.** Video: blocked on OBS install |
| 2 | Audio worker: VAD + streaming transcript + pauses/WPM/fillers | **Done** (`analyzer/audio_worker.py`, tested against real audio) |
| 3 | Video worker: gaze, head pose, blinks, calibration | **Done** (`analyzer/video_worker.py`, tested against a real webcam) |
| 4 | Event log + live Streamlit dashboard | **Done** (`analyzer/events.py`, `analyzer/capture.py`, `app.py` — all tested against real hardware) |
| 5 | Fusion layer: rule-based scoring | **Done** (`analyzer/fusion.py`, wired live into `app.py`, verified against hand-calculated values) |
| 6 | Feedback generator + background LLM scoring | Not started |
| 7 | Data collection + evaluation (validation, rater agreement, ablation) | Not started |
| 8 | Polish, demo rehearsal, final report | Not started |

**We're ahead of the week-by-week schedule** — Weeks 2 through 5 are all done. The
remaining open item from Week 1, call-window video capture, is blocked on installing
OBS Studio, not on anything technical. The next natural task is Week 6: turning the
fusion scores into worded coaching tips (`analyzer/feedback.py`), and eventually
`analyzer/text.py` (hedging/sentiment/answer-relevance) to feed the confidence score's
already-configured-but-unused `hedge_rate` input.

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
- Ethics: the live dashboard is interviewer-only, never shown to the candidate. Keep
  that boundary — it's already true of `app.py`, don't add a candidate-facing view.
- When adding real-time logic (a new worker, a new stage in `capture.py`), test it
  against real hardware before calling it done, not just against fakes. Three separate
  real bugs in this project so far were invisible to fake-based tests and only showed
  up against actual audio/video/timing — see "Known gotchas" for all of them. The
  `mcp__Claude_Browser__preview_start` tool (or `streamlit run app.py` directly) plus
  actually watching the dashboard is how `app.py` itself was verified.

---

## For a new contributor / new Claude session

If you're starting a fresh Claude Code session on this repo with no prior context,
paste this to Claude:

> Read `README.md` in full, then read `analyzer/config.py`, `analyzer/events.py`,
> `analyzer/audio_features.py`, `analyzer/transcribe.py`, `analyzer/vad.py`,
> `analyzer/audio_worker.py`, `analyzer/video_features.py`, `analyzer/video.py`,
> `analyzer/video_worker.py`, `analyzer/capture.py`, `analyzer/dashboard_data.py`,
> `analyzer/fusion.py` and `app.py` to see what's already built and tested — pay
> particular attention to the "Known gotchas" section, which explains several real
> bugs (dropped words from a bad chunking design, dropped words again from a
> threading bug, Whisper hallucinating on silence, a frozen UI element) that were
> found by testing against real hardware and fixed, and to `fusion.py`'s docstring,
> which explains the two rules its scoring enforces (never fabricate a missing
> metric; a sub-score with no inputs is `None`, not a fake default). Then run:
>   `mkdir -p analyzer/models && curl -L -o analyzer/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
> (the face-tracking model isn't committed to git) and
> `./.venv/Scripts/python.exe -m pytest tests/ -q` to confirm the setup works (expect
> 95 passed — none of these need a camera, mic or GPU, they all use fakes). If you want
> to see the actual dashboard, including the live fusion scores, `./.venv/Scripts/streamlit.exe run app.py`
> needs a real webcam and speakers/mic. Then help me build `analyzer/feedback.py`
> (Week 6): turn a `FusionResult` (see `analyzer/fusion.py`) into three to five
> specific, worded coaching tips grounded only in the actual computed numbers and
> timestamps (e.g. "Camera-facing time was 42% — try holding your gaze on the lens for
> 3-5 seconds at a time"), never inventing an observation the data doesn't support. If
> a language model is used to phrase the tips, it must only see the already-computed
> metrics as input, never raw video/audio — the scores themselves stay rule-based, per
> the synopsis Sec 5.7. Alongside that, `analyzer/text.py` (hedging/sentiment/answer
> relevance) is the other open Week 6 item, and it plugs straight into `fusion.py`'s
> confidence score with no changes needed there once it writes a `hedge_rate` metric.
