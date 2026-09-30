"""Post-session review page (imported by app.py). Shows a saved session's
report and offers it as a self-contained HTML download.

The page displays exactly the HTML that gets downloaded (analyzer/report.py's
render_html), so what the interviewer reads on screen and what they hand to
the candidate afterwards can never differ.

Like the live view, this is interviewer-facing. Handing the *exported* report
to the candidate afterwards is the intended coaching use -- the ethics rule is
about not showing live scores to the candidate mid-interview.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from analyzer.config import PATHS
from analyzer.dashboard_data import format_seconds
from analyzer.report import list_sessions, load_report, render_html


def render() -> None:
    st.title("Interview Coach — Session Report")
    st.caption("Review a saved session. Nothing here needs a camera or microphone.")

    sessions = list_sessions(PATHS.data_dir)
    if not sessions:
        st.info(f"No saved sessions found under `{PATHS.data_dir}/` yet. Run a session first, then come back here.")
        return

    default = 0
    last = st.session_state.get("last_session_dir")
    if last:
        names = [s.name for s in sessions]
        from pathlib import Path
        if Path(last).name in names:
            default = names.index(Path(last).name)

    chosen = st.selectbox("Session", sessions, index=default, format_func=lambda p: p.name)

    try:
        report = load_report(chosen)
    except Exception as exc:  # a corrupt/partial session file shouldn't take the page down
        st.error(f"Could not read this session: {exc}")
        return

    if report.duration <= 0:
        st.warning("This session has no recorded data (it may have been stopped immediately).")

    page = render_html(report)
    st.download_button("⬇ Download report (HTML)", data=page, file_name=f"report_{chosen.name}.html",
                       mime="text/html", type="primary")
    st.caption(f"Length {format_seconds(report.duration)} · open the downloaded file in any browser; "
               "use Print → Save as PDF for a PDF copy.")
    components.html(page, height=900, scrolling=True)
