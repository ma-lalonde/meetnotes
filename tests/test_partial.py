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

from meetnotes import llm, pipeline, store
from meetnotes.config import Config


@pytest.fixture
def meeting(tmp_path):
    cfg = Config()
    cfg.data_dir = str(tmp_path)
    cfg.asr.final_pass = False
    path = store.new_meeting(tmp_path, "sync")
    store.update_meta(
        path,
        segments=[{"speaker": "Me", "start": 0.0, "end": 2.0, "text": "we ship Tuesday"}],
        notes=[],
    )
    return cfg, path


def test_unconfigured_llm_keeps_the_transcripts(meeting):
    cfg, path = meeting
    cfg.llm.model = ""

    report = pipeline.process(path, cfg)

    assert (path / "raw_output/transcription.md").exists()
    assert (path / "raw_output/transcription_cleaned.md").exists()
    assert (path / "raw_output/notes.md").exists()
    assert report["raw_output/transcription.md"] == "written"
    assert report["summary.md"].startswith("skipped:")


def test_unconfigured_llm_records_why(meeting):
    cfg, path = meeting
    cfg.llm.model = ""
    pipeline.process(path, cfg)

    meta = store.read_meta(path)
    assert meta["state"] == "transcribed"
    assert "no LLM model selected" in meta["error"]


def test_unreachable_llm_is_not_fatal(meeting, monkeypatch):
    cfg, path = meeting
    cfg.llm.model = "some-model"
    monkeypatch.setattr(
        llm, "chat",
        lambda *a, **k: (_ for _ in ()).throw(llm.LlmError("connection refused")),
    )

    report = pipeline.process(path, cfg)
    assert (path / "raw_output/transcription.md").exists()
    assert "connection refused" in report["summary.md"]


def test_retry_after_configuring_the_llm_completes(meeting, monkeypatch):
    cfg, path = meeting
    cfg.llm.model = ""
    pipeline.process(path, cfg)

    cfg.llm.model = "good-model"
    monkeypatch.setattr(
        llm, "chat",
        lambda cfg_, system, user, schema=None, schema_name="r", on_token=None: (
            {"actions": []} if schema else "## Context\nfine"
        ),
    )
    report = pipeline.process(path, cfg)

    assert report["summary.md"] == "written"
    assert (path / "summary.md").exists()
    assert store.read_meta(path)["state"] == "done"
    # The transcripts were already correct, so they are left alone.
    assert report["raw_output/transcription.md"] == "skipped"
