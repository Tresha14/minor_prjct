# Multimodal Behavioural Analysis System — Project Report

*Final report, Week 8. Companion to `Project_Synopsis.docx` (the original proposal —
see its new Section 14 for a condensed version of this document) and `README.md`
(the continuously-updated technical reference). This document is written to be read
end-to-end by someone with no prior context — an evaluator, or you the night before
a demo.*

---

## 1. The problem

Job interview coaching usually reduces to vague advice — "be more confident," "make
eye contact." That advice isn't actionable: *which* moment, *how much*, *compared to
what*. This project builds a tool that watches a live video interview and turns face,
voice, and speech signals into **specific, timestamped, numbers-backed feedback** —
"camera-facing time was 42%, try holding your gaze on the lens for 3-5 seconds at a
time" instead of "make more eye contact."

**Explicit non-goal:** this is a self-practice / coaching tool, not a hiring or
screening tool. The live dashboard is shown only to the interviewer running a
practice session, never to a real candidate in an actual hiring decision. That
boundary is enforced in the code (see §6) and argued for in the ethics section of
`Project_Synopsis.docx` (Section 9) — real commercial tools that tried to score
candidates automatically (e.g. HireVue) walked back facial-analysis scoring under
fairness scrutiny; this project sidesteps that entire problem class by scope.

## 2. How a session actually works

1. The candidate joins a normal video call (e.g. FaceTime) with the interviewer.
2. The interviewer's laptop captures the call window (video) and two separate audio
   channels: system audio (the candidate's voice, via loopback) and the
   interviewer's own microphone. Two channels, not one, means no speaker
   diarization is needed — the code always knows whose voice it's looking at.
3. Three analysis modules run continuously against that captured stream:
   - **Video**: is the candidate facing the camera, how steady is their head, how
     often do they blink.
   - **Audio**: a live transcript with word-level timestamps, from which pauses,
     speaking rate, and filler words are derived.
   - **Text**: hedging language ("I guess", "maybe"), sentiment, and (not yet
     live-wired — see §7) how relevant an answer is to the question asked.
4. Every module writes its readings to one shared, timestamped **event log**
   (SQLite) — never to each other directly. This is the single most important
   architectural decision in the project (see §5).
5. A **live dashboard**, visible only to the interviewer, shows rolling metrics and
   flagged moments during the session.
6. A rule-based **fusion layer** turns the raw metrics into three interpretable
   0-100 scores — engagement, fluency, confidence — each traceable back to exactly
   which raw signals fed it.
7. A **coaching-tip generator** turns those scores' underlying raw numbers into
   worded, specific advice.

## 3. System architecture

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
   Live dashboard (Streamlit,          Fusion + feedback
   interviewer only)                   (rule-based scoring, then worded tips)
          |                                   |
          +------------- Post-session view ---+
```

**Why this shape, specifically:** each worker (video, audio) runs independently and
only *ever* reads/writes the shared event log — they never call each other's code
directly. That single decision is why a slow step (GPU transcription taking over a
second) can never freeze the live dashboard, and why adding the text module later
required zero changes to the audio or video workers. It also made the project
buildable module-by-module and testable module-by-module, which is a direct answer
to "how did you keep 161 tests passing across 8 weeks of work."

## 4. Why rule-based scoring, not a trained model

This is probably the single question most worth having a crisp answer to in a viva.
The dataset this project can realistically collect — a handful of mock interviews —
is far too small to train or fine-tune a model responsibly. A trained model on that
little data would either overfit or default to something close to a constant
output, and either way its output ("the model said 62") is not actionable and is
hard to defend under questioning. A transparent, additive score, by contrast:

- can always answer "why is this score 67" — the exact metrics and their weighted
  contributions are inspectable (`SubScore.drivers` in `analyzer/fusion.py`)
- degrades honestly — a score built from partial data is flagged as partial
  (`SubScore.coverage`), never silently presented as complete
- is directly actionable — "your pause rate was high" is a sentence a candidate can
  act on; a bare model score is not
- has weights that live in one place (`analyzer/config.py`'s `FUSION` block) and
  can be re-tuned later with real data (Week 7, see §8) without touching the
  scoring logic itself

The same reasoning was applied consistently to sentiment analysis
(`analyzer/text.py` uses a lexicon, not a downloaded model) — this is a project-wide
design principle, not a one-off choice, and it's worth stating that explicitly if
asked "why didn't you just use an LLM for everything."

## 5. The event log: the one thing every module agrees on

`analyzer/events.py` is a thin SQLite wrapper with three kinds of row: a `metric`
(a number at a point in time — `camera_facing = 0.9 at t=12.4s`), an `event`
(something discrete worth flagging — `long_pause` lasting 4.2s), and a `transcript`
word. Every worker writes; only the dashboard and the fusion/feedback layers read.
WAL mode lets the dashboard read while a worker is writing. This is the seam that
lets video, audio, text, fusion, and feedback all be developed, tested, and
demoed independently — and it's why adding Week 6/7 work didn't require touching
any of the Week 1-5 code.

## 6. Module-by-module

| Module | What it does | Status |
|---|---|---|
| `analyzer/events.py` | Shared SQLite event log | Done, tested, used by everything |
| `analyzer/audio_features.py` | Pure math: pauses, WPM, fillers, hedge counting | Done |
| `analyzer/transcribe.py` | faster-whisper wrapper, GPU with CPU fallback | Done, real-hardware verified |
| `analyzer/vad.py` | Silero voice-activity detection | Done |
| `analyzer/audio_worker.py` | Live pipeline: silence-aware chunking + transcription + stats | Done, real-hardware verified |
| `analyzer/video_features.py` | Pure math: head pose, calibrated baseline, blinks | Done |
| `analyzer/video.py` | MediaPipe FaceLandmarker wrapper | Done, real-hardware verified |
| `analyzer/video_worker.py` | Live pipeline: calibration + camera-facing/pose/blink metrics | Done, real-hardware verified |
| `analyzer/capture.py` | Wires real webcam/loopback/mic into the workers | Done, real-hardware verified |
| `analyzer/dashboard_data.py` | Pure formatting for the dashboard | Done |
| `app.py` | Streamlit live dashboard (interviewer-only) | Done, live-browser verified |
| `analyzer/fusion.py` | Rule-based engagement/fluency/confidence scoring | Done, hand-verified |
| `analyzer/feedback.py` | Turns scores into worded coaching tips | Done |
| `analyzer/text.py` | Hedge rate, sentiment, answer relevance | Done, **not yet live-wired** (see §7) |
| `evaluation/` | Rater agreement, correlation, ablation tooling | Tooling done, **no real data yet** (see §8) |

### Video module
MediaPipe's Face Landmarker (the modern Tasks API, not the deprecated
`mp.solutions.face_mesh` most online examples use) gives a rotation matrix per
frame, decomposed into yaw/pitch/roll via `scipy.spatial.transform.Rotation`. The
first ~8 seconds of a session are a **calibration window** — a running-average
"normal" pose is built from them, and everything afterward is compared against
*that* baseline, not an absolute "0 = facing camera" assumption. Blinks come from
MediaPipe's blendshape scores (`eyeBlinkLeft`/`eyeBlinkRight`), which proved more
reliable than manually computing eye-aspect-ratio from raw landmarks.

### Audio module
Chunk boundaries for transcription are chosen at **detected silence** (via Silero
VAD), never a fixed time offset — an earlier fixed-8-second design silently dropped
words at chunk boundaries because a chunk starting mid-word gets skipped, not
mis-transcribed, by Whisper. A chunk is also skipped entirely if the VAD finds
almost no real speech in it, because Whisper hallucinates short phrases ("You",
"Thank you") on near-silent audio otherwise.

### Text module
`hedge_rate` reuses the same hedge-phrase lexicon audio_features.py already defined
(one single definition of "what's a hedge phrase" project-wide). Sentiment is
exact-token lexicon matching (so "strongly" doesn't false-match "strong"). Answer
relevance is Jaccard word-overlap between question and answer text, after removing
stopwords.

### Fusion layer
Three sub-scores, each a weighted average of normalised (0-1) metric readings.
Two rules are enforced everywhere: a metric that hasn't been observed yet is
**dropped from the average, never guessed at**; if every input to a sub-score is
missing, the sub-score is **`None`, not a fabricated default**. A real bug this
caught: a solo tester saw "Confidence = 67" built from only 2 of 4 configured
signals — the number itself was computed correctly, but looked exactly as
confident as a fully-informed score. Fixed by adding `SubScore.coverage` and
showing it next to every score, always.

### Feedback generator
Takes the same raw metric values fusion.py scores (not a second read of the log),
reuses fusion.py's exact `band_score`/`linear_lower_is_better`/`identity01`
functions and thresholds so a tip can never describe a number differently than the
score that was computed from it, and returns 1-5 tips — worst signals first — or an
honest "not enough data yet" / "everything's comfortable" placeholder instead of an
empty list.

## 7. Known gaps (say these before an evaluator finds them)

- **Call-window capture via OBS Virtual Camera** is implemented but unexercised —
  OBS Studio isn't installed on the dev machine. The code path (`capture.py`'s
  `VideoSource` takes any camera index) needs no changes once it is; validated
  instead against a plain webcam.
- **`analyzer/text.py`'s `TextWorker` isn't wired into the live pipeline.** It's
  built and tested standalone; integrating it into `capture.py`/`app.py` the way
  audio/video workers already are is left as next-step work.
- **`answer_relevance` needs question text that nothing currently captures.** The
  interviewer asks questions out loud; `app.py`'s "mark next question" button only
  records a timestamp, not the question's wording. A text box for the interviewer
  to type/paste the question is the missing piece.
- **Background LLM-based tip phrasing** (mentioned as an option in the original
  synopsis) was not built — tips are template-based against real computed numbers
  instead, which satisfies "grounded only in real data" without an external-API
  dependency added this late.

Stating these upfront, with the reasoning, is a stronger answer in a viva than
having an evaluator discover them and ask why.

## 8. Evaluation: what's real and what isn't

**Built:** `evaluation/statistics.py` (Pearson correlation, Spearman correlation,
Cohen's kappa — each returns `None`, never a fabricated 0.0/1.0, when the input
can't support the statistic), `evaluation/ablation.py` (leave-one-signal-out
ablation, reusing `fusion.weighted_average()` directly so it can never drift from
the real scoring math), `evaluation/validation.py` (ties rater agreement and
system-vs-human correlation together, explicitly flagging small sample sizes
instead of hiding them). 27 tests, all against synthetic/textbook example data.

**Not done, and stated plainly:** no real interviews have been recorded and rated
by real people. That's Week 7's actual point, and it cannot be substituted with
invented numbers without making the whole evaluation section meaningless. If asked
for results in a viva, the honest answer is: "the measurement tooling is built and
tested; the data collection — recording sessions and getting them independently
rated — is the next step, documented in `evaluation/README.md`."

This is a deliberate, defensible position, not an oversight — and it is consistent
with how the entire project was built: every "Done" claim in this report is backed
by a specific test or a specific real-hardware run, not an assumption.

## 9. Testing philosophy

161 automated tests, in two categories:

- **Pure-logic tests** (the large majority) — no camera, microphone, or GPU needed.
  Pure functions tested with hand-calculated "golden value" expectations (not just
  "does it run without crashing"), missing-data edge cases, and — for the fusion
  layer specifically — monotonicity sweeps (e.g. more camera-facing time can never
  *lower* the engagement score, checked by sweeping the entire 0-1 range).
- **Real-hardware verification** — several real bugs in this project were invisible
  to fake-based tests and only surfaced when actually run against real
  audio/video/timing: words dropped at fixed chunk boundaries, words dropped from a
  threading bug (a slow GPU call blocking the audio-capture thread so the OS buffer
  overflowed), Whisper hallucinating on near-silent audio, a frozen elapsed-time UI
  element, and a misleading "confident-looking" score built from almost no data.
  Each is documented with its root cause in `README.md`'s "Known gotchas" section —
  worth reading verbatim before a viva, since "we found a real bug during testing
  and here's exactly what caused it and how we fixed it" is far stronger material
  than "we wrote tests and they passed."

## 10. Ethics and limitations (condensed from the synopsis, Section 9)

- The live dashboard is interviewer-only — never shown to the candidate, by design,
  and enforced in `app.py`.
- Metrics are **behavioural proxies**, not measurements of true confidence or
  engagement — stated explicitly, not implied to be more than they are.
- Signals like gaze and blink rate vary with glasses, lighting, camera position,
  and individual differences — a fairness caveat worth having ready.
- This is a coaching tool. It is explicitly not intended, and not validated, for
  screening or hiring decisions.

## 11. What to say if asked "is this finished"

No, and that's a more credible answer than "yes." Weeks 1-6 are complete and
real-hardware verified. Week 7's *tooling* is complete and tested; its *data* is
not collected, because that requires real interviews and real independent raters,
which cannot be rushed. Week 8 (this report, the testing guide, and a rehearsed
live demo) is the current work. Every claim in this report traces to a specific
test file or a specific real-hardware run — see `README.md`'s "Current status" for
the file-by-file breakdown.
