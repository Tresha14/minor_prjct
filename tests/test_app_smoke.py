"""Smoke test for app.py using Streamlit's own AppTest framework: verifies
the dashboard loads without an exception in its idle (session not started)
state, and that the key controls are present. Deliberately does NOT click
"Start session" -- that loads the real GPU speech model and opens a real
camera/microphone, which isn't appropriate for an automated test run. That
path is instead verified manually against real hardware (see README.md's
status section for how, and app.py's own docstring).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_app_loads_without_exception_in_idle_state():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    at.run(timeout=30)

    assert not at.exception, f"app.py raised on load: {at.exception}"


def test_start_and_stop_buttons_present_when_idle():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    at.run(timeout=30)

    labels = [b.label for b in at.button]
    assert any("Start session" in l for l in labels)
    assert not any("Stop session" in l for l in labels)  # not running yet, so no stop button


def test_mark_question_button_is_disabled_when_idle():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    at.run(timeout=30)

    mark_button = next(b for b in at.button if "Mark next question" in b.label)
    assert mark_button.disabled
