"""The shared event log.

Every worker (video, audio, text) writes to this one SQLite file, and the
Streamlit dashboard only ever reads from it. This is what lets the workers
run as separate processes without talking to each other directly: a slow
worker can never block a fast one.

Two kinds of row:
  - a "metric" row: a single numeric reading at a point in time
    (e.g. camera_facing = 1.0 at t = 12.4s)
  - an "event" row: something discrete worth flagging
    (e.g. "long_pause" lasting 4.2s starting at t = 30.1s)

SQLite's default settings serialise writers, which is fine here: writes are
small and infrequent (roughly once a second per worker), and WAL mode lets
the dashboard read while a worker is writing.
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    t           REAL NOT NULL,      -- seconds since session start
    source      TEXT NOT NULL,      -- 'video' | 'audio' | 'text' | 'fusion'
    name        TEXT NOT NULL,      -- e.g. 'camera_facing', 'wpm', 'hedge_rate'
    value       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_t ON metrics(t);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    t_start     REAL NOT NULL,
    t_end       REAL,
    source      TEXT NOT NULL,
    kind        TEXT NOT NULL,      -- e.g. 'long_pause', 'gaze_away', 'question_marker'
    detail      TEXT                -- JSON blob: free-form extra info
);
CREATE INDEX IF NOT EXISTS idx_events_t ON events(t_start);

CREATE TABLE IF NOT EXISTS transcript (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    t_start     REAL NOT NULL,
    t_end       REAL NOT NULL,
    word        TEXT NOT NULL,
    question_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_transcript_t ON transcript(t_start);

CREATE TABLE IF NOT EXISTS session (
    key         TEXT PRIMARY KEY,
    value       TEXT
);
"""


class EventLog:
    """Thin wrapper around the session's SQLite file.

    Usage:
        log = EventLog(session_dir)
        log.session_start()          # once, at the start of the session
        log.metric("audio", "wpm", 118.0)
        log.event("audio", "long_pause", t_start=30.1, t_end=34.3)
    """

    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.session_dir / "events.db"
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _cursor(self):
        conn = self._connect()
        try:
            yield conn.cursor()
            conn.commit()
        finally:
            conn.close()

    # ---- session bookkeeping -------------------------------------------------

    def session_start(self, t0: float | None = None) -> float:
        t0 = t0 if t0 is not None else time.time()
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO session (key, value) VALUES ('t0', ?)",
                (str(t0),),
            )
        return t0

    def get_t0(self) -> float | None:
        with self._cursor() as cur:
            cur.execute("SELECT value FROM session WHERE key = 't0'")
            row = cur.fetchone()
        return float(row[0]) if row else None

    # ---- writing ---------------------------------------------------------

    def metric(self, source: str, name: str, value: float, t: float | None = None) -> None:
        t = t if t is not None else self._elapsed()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO metrics (t, source, name, value) VALUES (?, ?, ?, ?)",
                (t, source, name, value),
            )

    def event(self, source: str, kind: str, t_start: float, t_end: float | None = None,
               detail: dict | None = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO events (t_start, t_end, source, kind, detail) VALUES (?, ?, ?, ?, ?)",
                (t_start, t_end, source, kind, json.dumps(detail or {})),
            )

    def word(self, word: str, t_start: float, t_end: float, question_id: int | None = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO transcript (t_start, t_end, word, question_id) VALUES (?, ?, ?, ?)",
                (t_start, t_end, word, question_id),
            )

    def _elapsed(self) -> float:
        t0 = self.get_t0()
        return time.time() - t0 if t0 else 0.0

    # ---- reading (used by the dashboard) ----------------------------------

    def recent_metrics(self, name: str, since_t: float = 0.0, limit: int = 2000):
        with self._cursor() as cur:
            cur.execute(
                "SELECT t, value FROM metrics WHERE name = ? AND t >= ? ORDER BY t ASC LIMIT ?",
                (name, since_t, limit),
            )
            return cur.fetchall()

    def recent_events(self, since_t: float = 0.0, limit: int = 500):
        with self._cursor() as cur:
            cur.execute(
                "SELECT t_start, t_end, source, kind, detail FROM events "
                "WHERE t_start >= ? ORDER BY t_start DESC LIMIT ?",
                (since_t, limit),
            )
            return cur.fetchall()

    def all_transcript(self):
        with self._cursor() as cur:
            cur.execute("SELECT t_start, t_end, word, question_id FROM transcript ORDER BY t_start ASC")
            return cur.fetchall()
