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

"""The whole post-processing path against a stubbed OpenAI-compatible server.

Everything below the HTTP boundary is the real code: pipeline, artifacts,
outputs, store. Only the model server is faked.
"""

import json

import httpx
import pytest

from meetnotes import llm, pipeline, store
from meetnotes.config import Config

SUMMARY_TEXT = "## Context\n\nMarc and Chloe discussed the SLA.\n"
ACTIONS_JSON = {
    "actions": [
        {"owner": "Marc", "task": "send the SLA to Chloe", "due": "2026-08-10",
         "kind": "todo", "quote": "we need the SLA", "at": 2.0}
    ]
}


def sse(text: str) -> bytes:
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": chunk}}]})
        for chunk in text.split(" ")
    ]
    return ("\n".join(lines) + "\ndata: [DONE]").encode()


@pytest.fixture
def server(monkeypatch):
    """An OpenAI-compatible server that streams, like LM Studio does."""
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        wants_json = "response_format" in body
        text = json.dumps(ACTIONS_JSON) if wants_json else SUMMARY_TEXT
        if body.get("stream"):
            return httpx.Response(200, content=sse(text))
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kw: real_client(**{**kw, "transport": httpx.MockTransport(handler)}),
    )
    return seen


@pytest.fixture
def meeting(tmp_path):
    cfg = Config()
    cfg.data_dir = str(tmp_path)
    cfg.asr.final_pass = False
    cfg.llm.model = "qwen3-8b"
    path = store.new_meeting(tmp_path, "sync")
    store.update_meta(
        path,
        segments=[
            {"speaker": "Marc", "start": 0.0, "end": 2.0, "text": "we ship Tuesday"},
            {"speaker": "Chloe", "start": 2.0, "end": 4.0, "text": "we need the SLA"},
        ],
        notes=[{"at": 3.0, "text": "chase the SLA"}],
    )
    return cfg, path


def test_every_artifact_is_produced(meeting, server):
    cfg, path = meeting
    report = pipeline.process(path, cfg)

    for name in (
        "raw_output/transcription.md",
        "raw_output/transcription_cleaned.md",
        "raw_output/notes.md",
        "transcription_cleaned_with_notes.md",
        "summary.md",
        "actions.md",
        "actions.json",
    ):
        assert (path / name).exists(), name
        assert (path / name).read_text().strip(), f"{name} is empty"
    assert report["summary.md"] == "written"


def test_summary_content_reaches_the_file(meeting, server):
    cfg, path = meeting
    pipeline.process(path, cfg)
    assert "Marc and Chloe discussed the SLA." in (path / "summary.md").read_text()


def test_actions_become_markdown_and_a_calendar_entry(meeting, server):
    cfg, path = meeting
    pipeline.process(path, cfg)

    assert "send the SLA to Chloe" in (path / "actions.md").read_text()
    ics = list((path / "calendar").glob("*.ics"))
    assert len(ics) == 1
    assert "DUE;VALUE=DATE:20260810" in ics[0].read_text()


def test_the_transcript_actually_sent_contains_the_speech_and_notes(meeting, server):
    cfg, path = meeting
    pipeline.process(path, cfg)

    sent = server[0]["messages"][1]["content"]
    assert "Marc: we ship Tuesday" in sent
    assert "Chloe: we need the SLA" in sent
    assert "NOTE FROM THE NOTE-TAKER: chase the SLA" in sent


def test_state_ends_as_done(meeting, server):
    cfg, path = meeting
    pipeline.process(path, cfg)
    assert store.read_meta(path)["state"] == "done"


def test_rerunning_changes_nothing(meeting, server):
    cfg, path = meeting
    pipeline.process(path, cfg)
    before = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}

    report = pipeline.process(path, cfg)

    assert set(report.values()) == {"skipped"}
    assert {p: p.read_bytes() for p in path.rglob("*") if p.is_file()} == before


def test_a_meeting_with_no_speech_is_refused_before_calling_the_model(meeting, server):
    cfg, path = meeting
    store.update_meta(path, segments=[], notes=[])

    report = pipeline.process(path, cfg)

    assert "nothing to summarize" in report["summary.md"]
    assert not (path / "summary.md").exists()
    assert store.read_meta(path)["state"] == "transcribed"
    assert server == []


def test_a_server_returning_nothing_leaves_no_empty_file(meeting, monkeypatch):
    cfg, path = meeting
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kw: real_client(**{**kw, "transport": httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": ""}}]}
            )
        )}),
    )

    report = pipeline.process(path, cfg)

    assert "empty response" in report["summary.md"]
    assert not (path / "summary.md").exists()
    # The transcripts survive a language model failure.
    assert (path / "raw_output/transcription.md").read_text().strip()
