# meetnotes - local meeting recorder with live transcription and notes
# Copyright (C) 2026 Marc-Antoine Lalonde
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from datetime import datetime, timezone

from meetnotes import outputs

META = {"title": "Sync", "created": "2026-08-04T14:30:00", "notes": []}


def test_clock():
    assert outputs.clock(0) == "00:00:00"
    assert outputs.clock(61) == "00:01:01"
    assert outputs.clock(3725) == "01:02:05"


def test_clean_removes_english_fillers():
    assert outputs.clean_text("uh so um we should, you know, ship it") == "so we should, ship it"


def test_clean_removes_french_fillers():
    assert outputs.clean_text("euh ben on peut, genre, livrer") == "on peut, livrer"


def test_clean_collapses_stutter():
    assert outputs.clean_text("we we we need it") == "we need it"


def test_clean_keeps_ordinary_text_intact():
    text = "The migration lands Tuesday and Marc owns the rollback."
    assert outputs.clean_text(text) == text


def test_clean_does_not_eat_substrings():
    assert outputs.clean_text("the umbrella is a benchmark") == "the umbrella is a benchmark"


def test_transcript_groups_by_speaker():
    segments = [
        {"speaker": "Me", "start": 0.0, "end": 1.0, "text": "hello"},
        {"speaker": "Me", "start": 1.0, "end": 2.0, "text": "again"},
        {"speaker": "Them", "start": 2.0, "end": 3.0, "text": "hi"},
    ]
    out = outputs.render_transcript(META, segments, clean=False)
    assert out.count("**Me**") == 1
    assert out.count("**Them**") == 1
    assert "[00:00:00] hello" in out


def test_ics_todo_has_due_and_escapes():
    action = {
        "owner": "Marc",
        "task": "Send the SLA, then invoice",
        "due": "2026-08-10",
        "kind": "todo",
        "quote": "we need it by Monday",
        "at": 42.0,
    }
    ics = outputs.render_ics(META, action, "uid-1", datetime(2026, 8, 4, tzinfo=timezone.utc))
    assert "BEGIN:VTODO" in ics
    assert "DUE;VALUE=DATE:20260810" in ics
    assert "SUMMARY:Send the SLA\\, then invoice" in ics
    assert ics.endswith("\r\n")


def test_ics_event_spans_one_day():
    action = {"owner": "Marc", "task": "Review", "due": "2026-08-10", "kind": "event",
              "quote": "", "at": 0}
    ics = outputs.render_ics(META, action, "uid-2", datetime(2026, 8, 4, tzinfo=timezone.utc))
    assert "DTSTART;VALUE=DATE:20260810" in ics
    assert "DTEND;VALUE=DATE:20260811" in ics


def test_ics_folds_long_lines_under_76_octets():
    action = {"owner": "Marc", "task": "x" * 200, "due": "2026-08-10", "kind": "todo",
              "quote": "", "at": 0}
    ics = outputs.render_ics(META, action, "uid-3", datetime(2026, 8, 4, tzinfo=timezone.utc))
    for line in ics.split("\r\n"):
        assert len(line.encode("utf-8")) <= 75
