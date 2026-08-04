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

import threading

import pytest

from meetnotes import artifacts, llm, pipeline, session, store
from meetnotes.config import Config


@pytest.fixture
def meeting(tmp_path):
    cfg = Config()
    cfg.data_dir = str(tmp_path)
    cfg.asr.final_pass = False
    cfg.llm.model = ""
    path = store.new_meeting(tmp_path, "sync")
    store.update_meta(
        path,
        segments=[{"speaker": "Me", "start": 0.0, "end": 2.0, "text": "we ship Tuesday"}],
        notes=[],
    )
    return cfg, path


def test_pipeline_reports_each_step(meeting):
    cfg, path = meeting
    cfg.llm.model = "fake"
    steps = []
    try:
        pipeline.process(path, cfg, progress=steps.append)
    except llm.LlmError:
        pass
    assert "writing transcripts" in steps
    assert "summarizing" in steps


def test_state_on_disk_moves_through_the_stages(meeting):
    cfg, path = meeting
    seen = []

    original = store.update_meta

    def spy(target, **changes):
        if "state" in changes:
            seen.append(changes["state"])
        return original(target, **changes)

    store.update_meta = spy
    try:
        pipeline.process(path, cfg)
    finally:
        store.update_meta = original

    assert seen[0] == "transcribing"
    assert "summarizing" in seen
    # No language model configured, so it settles as transcribed rather than done.
    assert seen[-1] == "transcribed"


def test_active_map_is_populated_while_running_and_cleared_after(meeting):
    cfg, path = meeting
    done = threading.Event()
    sess = session.Session(cfg, on_state=lambda state, detail: (
        done.set() if state in ("done", "failed", "idle") else None
    ))

    observed = []
    original = pipeline.process

    def watched(target, config, **kwargs):
        progress = kwargs.get("progress")
        if progress:
            progress("halfway")
        observed.append(dict(sess.active))
        return original(target, config, **kwargs)

    pipeline.process = watched
    try:
        sess.process_async(path)
        assert done.wait(timeout=30)
    finally:
        pipeline.process = original

    assert observed and observed[0].get(path.name) == "halfway"
    assert sess.active == {}


def test_active_map_is_cleared_even_when_processing_fails(meeting, monkeypatch):
    cfg, path = meeting
    done = threading.Event()
    sess = session.Session(cfg, on_state=lambda state, detail: (
        done.set() if state in ("done", "failed", "idle") else None
    ))
    monkeypatch.setattr(
        pipeline, "process",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    sess.process_async(path)
    assert done.wait(timeout=30)
    assert sess.active == {}
    assert store.read_meta(path)["state"] == "failed"


def test_file_hash_is_cached_until_the_file_changes(tmp_path):
    target = tmp_path / "a.md"
    target.write_text("one")
    first = artifacts.file_hash(target)
    assert artifacts.file_hash(target) == first

    target.write_text("two different bytes")
    assert artifacts.file_hash(target) != first


def test_file_hash_of_a_missing_file_does_not_raise(tmp_path):
    assert artifacts.file_hash(tmp_path / "gone.md")
