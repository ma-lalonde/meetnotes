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

import pytest

from meetnotes import audio, outputs, session, store
from meetnotes.config import Config


def test_safe_label_keeps_ordinary_names():
    assert store.safe_label("Marc-Antoine Lalonde", "Me") == "Marc-Antoine Lalonde"
    assert store.safe_label("O'Brien", "Me") == "O'Brien"


def test_safe_label_keeps_accents():
    assert store.safe_label("Chloé Gagnon", "Me") == "Chloé Gagnon"


def test_safe_label_strips_path_separators():
    # The label becomes a filename, so a slash must not create a directory.
    assert "/" not in store.safe_label("a/b", "Me")
    assert ".." not in store.safe_label("..", "Me")


def test_safe_label_falls_back_when_empty():
    assert store.safe_label("", "Me") == "Me"
    assert store.safe_label("   ", "Participants") == "Participants"
    assert store.safe_label("///", "Me") == "Me"


def test_identical_names_are_disambiguated(tmp_path, monkeypatch):
    cfg = Config()
    cfg.data_dir = str(tmp_path)
    cfg.capture.mic_source = "1"
    cfg.capture.system_source = "2"
    cfg.llm.free_vram_before_recording = False

    started = {}

    class FakeRecorder:
        def __init__(self, cmd, rate, tracks, out_dir):
            started["tracks"] = tracks
            self.paths = {label: out_dir / f"{label}.wav" for label in tracks}
            self.started_at = 0.0

        def start(self):
            pass

        def failed(self):
            return {"boom": "stop here"}

        def stop(self):
            return 0.0

    monkeypatch.setattr(audio, "Recorder", FakeRecorder)
    monkeypatch.setattr(session.time, "sleep", lambda *_: None)

    sess = session.Session(cfg)
    with pytest.raises(RuntimeError):
        sess.start("sync", mic_label="Marc", system_label="Marc")

    assert list(started["tracks"]) == ["Marc", "Marc (other)"]


def test_named_speakers_appear_as_transcript_headings():
    meta = {"title": "Sync", "created": "2026-08-04T14:30:00", "notes": []}
    segments = [
        {"speaker": "Marc", "start": 0.0, "end": 2.0, "text": "we ship Tuesday"},
        {"speaker": "Chloe", "start": 2.0, "end": 4.0, "text": "we need the SLA"},
    ]
    text = outputs.render_transcript(meta, segments, clean=True)
    assert "**Marc**" in text
    assert "**Chloe**" in text
    assert "**Me**" not in text


def test_named_speakers_reach_the_language_model():
    meta = {"title": "Sync", "created": "2026-08-04T14:30:00", "notes": []}
    segments = [{"speaker": "Marc", "start": 0.0, "end": 2.0, "text": "we ship Tuesday"}]
    source = outputs.transcript_for_llm(meta, segments)
    assert "Marc: we ship Tuesday" in source
