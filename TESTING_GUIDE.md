# Testing Guide — run this yourself, step by step

Goal: verify the whole system works, end to end, on your machine, before the demo.
Each step says what to run and **exactly what "working" looks like**, so you can
tell a real problem apart from something that's supposed to look that way.

Do these in order. Steps 1-3 just confirm setup (already done once tonight).
Step 4 onward is the actual hands-on test with your webcam/mic.

---

## 1. Confirm the environment

```bash
cd <path to your minor_project folder>
./.venv/Scripts/python.exe --version
```
Expect: `Python 3.11.9`. If this errors, the venv isn't set up —
`python -m venv .venv` then `./.venv/Scripts/python.exe -m pip install -r requirements.txt`.

## 2. Confirm the face-tracking model is present

```bash
ls analyzer/models/face_landmarker.task
```
Expect: a file, ~3.6 MB. If missing:
```bash
mkdir -p analyzer/models
curl -L -o analyzer/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## 3. Run the automated test suite

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
```
**Expect: `209 passed`**, no failures, finishing in well under a minute. This
doesn't touch your camera or mic — it's every pure-logic test plus a Streamlit
smoke test in its idle state. If anything fails here, fix it before moving on —
everything after this point assumes this is green.

---

## 4. Launch the live dashboard

```bash
./.venv/Scripts/streamlit.exe run app.py
```
A browser tab should open automatically (usually `http://localhost:8501`). If it
doesn't, open that URL manually. **Expect:** a page titled with the app name, a
"Start session" control, and no metrics yet (session hasn't started).

If you're testing **alone** (no second person to play "candidate"), check the
**"solo test mode"** checkbox before starting — this routes your own microphone
into the transcribed (candidate) channel instead of the loopback channel, since
normally the candidate's voice comes from the call's speaker output, not your mic.
**Never use solo test mode for an actual practice interview** — it skips the
interviewer's own channel entirely. Leave it unchecked for a real two-person run.

## 4b. Testing a real video call (no OBS)

Use this when the candidate is on another laptop, in a Zoom/Meet/Teams call on yours.
1. Join the call on your laptop; put the candidate's video large (speaker view).
2. In the app set **Video source -> Capture a window (no OBS needed)**, pick the call window.
3. Click **Preview what will be analysed** and use the **Trim** sliders until the candidate's
   face fills most of the picture and no other face is visible. (A small tile reads as
   "no face" -- that is the usual reason nothing shows up.)
4. Leave **Solo test mode** off and use headphones, so the candidate's voice comes from your
   default output (captured by loopback) and your mic stays the separate interviewer channel.
5. Start the session; the candidate looks at their own camera for the first 8 s.
It keeps working if another window is on top, but not if the call window is minimised.

## 5. Start a session and get through calibration

Click **Start session**. For the first ~8 seconds you should see a prominent
banner/progress bar telling you to look at the camera — **actually look at the
lens during this window**. This sets the "normal" baseline everything else is
compared against; if you look at the dashboard instead of the camera during these
8 seconds, the rest of the session will read as "looking away" almost constantly,
which is expected behavior given a bad calibration, not a bug (see README's
"Known gotchas" if this happens and you want to understand exactly why).

**Expect after calibration:** the banner disappears, and camera-facing metrics
start updating.

## 6. Talk, and watch the live metrics

Say a few sentences out loud (or play speech through your speakers, aimed at
whichever channel you configured). Within a few seconds you should see:

- **Live transcript** updating with the words you said
- **WPM** (words per minute) showing a plausible number (~110-165 is "comfortable")
- **Pause rate** and **filler rate** tiles updating
- **Hedges / 100 words** tile updating (say a few "maybe" / "I guess" phrases to push it up)
- **Camera-facing %** updating as you look at/away from the lens
- Charts scrolling forward in real time

If you deliberately look away from the camera for a few seconds, expect a
"Looking away from camera" flagged event to appear in the timeline once that
interval flushes (every ~2 seconds by default).

If you go quiet for a few seconds, expect a "Long pause" flagged event once the
pause crosses the threshold (~3 seconds by default).

**A known, correct (not broken) behavior:** if you're testing with only ambient
room noise and no one actually speaking, the transcript should stay empty — the
system deliberately skips sending near-silent audio to the transcriber, because
Whisper otherwise hallucinates short phrases on silence.

## 7. Check the fusion scores

After you've spoken enough for at least a few metrics to populate, look for the
**Engagement / Fluency / Confidence** score tiles. Each should show:
- a 0-100 number
- a **coverage** indicator (how many of that score's configured signals actually
  have data yet) — this should genuinely be less than 100% early in a session and
  climb as more signals come in; if a score shows a number but coverage is low,
  that's the system correctly telling you not to over-trust it yet, not a bug.
- an expander ("How these scores were built" or similar) showing exactly which
  metrics fed the score and their individual contributions

## 7b. Check the live coaching tips

Under the scores there is a **Coaching tips (live)** section. Expect 🔸 tips for weak
readings (e.g. look away from the camera for a while → a camera-facing tip), ✅ for good
ones, or an ℹ️ line if there isn't enough data yet. These use only the latest reading, so
they can flip while you talk — that's expected; the report is the steadier view.

## 8a. Automatic question detection

With **Detect questions automatically from my microphone** ticked (default, and only when
solo test mode is off), you don't mark anything: ask a question out loud, pause ~2 s, and
within a few seconds a `New question marked (Q1: <your wording>)` event appears and the
caption shows "auto question detection on (N found)". Use headphones, otherwise the
candidate's voice leaks into your mic. Statements ("thanks, great answer") are not marked;
the manual button still works for anything it misses. **Not yet tried on a live mic** --
check that the wording it captured is right, and that it doesn't mark the candidate's answers.

## 8. Mark questions (with wording) and check timing

Type a question into the box next to **Mark next question** (e.g. "Describe your
experience with databases"), then click **Mark next question**. Expect a toast
"Marked question 1 at MM:SS", the box to clear, and an entry like
`New question marked (Q1: Describe your experience with databases)` in the events list.
Answer out loud, mark a second question the same way, and answer again. Marking one
*without* typing wording is fine too — it just won't get a relevance score.

## 9. Stop the session

Click **Stop session**. Expect:
- no errors in the terminal running Streamlit
- the app returns to something like its idle/start state
- a new subfolder under `data/sessions/` with an `events.db` file inside, timestamped
  around when you started

Confirm the data actually saved:
```bash
ls data/sessions/
```
Pick the newest folder and confirm `events.db` exists and has non-trivial size
(not 0 bytes).

## 9b. Open the post-session report

In the sidebar switch **Mode → Review saved session**. Pick the session you just ran
(the newest is preselected). Expect:
- Whole-session Engagement / Fluency / Confidence, each with "built from X% of usual
  signals" (Confidence should now include the hedge signal if you spoke).
- One section per question: your typed wording, its scores, the raw averages, answer
  relevance (% of the question's key words your answer touched), tips, the flags that
  happened during it, and your answer's transcript **under the right question** — words
  spoken just before you pressed Mark should still be in the previous question.
- "Best moments to replay" — may legitimately say "No stretch had several signals worse
  than usual at once" for a short or steady session. To see one, deliberately look away
  from the camera *and* go quiet/hedge for ~30 s in the middle of a longer run.
- **Download report (HTML)** gives a file that opens on its own in any browser (no
  internet needed); Print → Save as PDF works.

## 10. Sanity-check a saved session's data directly (optional but reassuring)

```bash
./.venv/Scripts/python.exe -c "
from analyzer.events import EventLog
log = EventLog('data/sessions/<the folder you just created>')
print('words:', len(log.all_transcript()))
print('wpm readings:', len(log.recent_metrics('wpm')))
print('events:', len(log.recent_events()))
"
```
Replace `<the folder you just created>` with the actual folder name. Expect
non-zero counts if you spoke during the session.

---

## 11. Things that are SUPPOSED to look a certain way (don't "fix" these)

- **Empty transcript with only ambient noise** — correct, not a bug (see step 6).
- **Low coverage early in a session** — correct; coverage climbs as more signals
  arrive, and a caller (you) is meant to notice it, not be surprised by it.
- **Near-constant "looking away" for the whole session** — almost always means
  calibration happened while you weren't looking at the camera (step 5), not a
  scoring bug. Restart the session and actually look at the lens for the first 8s.
- **No combined flags in a short or steady session** — correct. They only appear when
  at least two signals are clearly worse than *your own* typical level in the same 30 s.
  A session that is uniformly weak also shows few, by design (the tips cover that).
- **Answer relevance shown as "not scored"** — it needs typed question wording and at
  least one transcribed answer word for that question.
- **Live tips changing from moment to moment** — they use only the latest reading; the
  report averages the whole window.

## 12. If something goes wrong

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError` mentioning `face_landmarker.task` | Step 2 wasn't done — download the model. |
| Report page says "No saved sessions found" | Nothing has been saved under `data/sessions/` yet — run and Stop a session first. |
| GPU transcription errors / falls back to CPU with a warning | Expected on a machine without the exact CUDA setup this was tuned for (see README "Known gotchas" — needs `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` via pip, not a full CUDA toolkit install). CPU fallback still works, just slower. |
| No camera-facing data at all | Check another app isn't already holding the webcam; check the correct camera index is configured. |
| Streamlit page never loads | Check the terminal for a Python traceback; check nothing else is already using port 8501. |
| `cublas64_12.dll is not found` | Means `analyzer.transcribe`'s CUDA DLL path registration didn't run, or the `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` pip packages are missing — see README "Known gotchas". |

If you hit something not on this list, check `README.md`'s "Known gotchas" section
first — several real, previously-found bugs are documented there with their exact
symptoms and root causes.
