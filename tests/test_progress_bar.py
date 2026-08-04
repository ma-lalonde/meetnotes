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

import json

import httpx
import pytest

from meetnotes import llm, pipeline, store
from meetnotes.config import Config


def test_stage_weights_sum_to_one():
    assert sum(weight for _, weight in pipeline.STAGES) == pytest.approx(1.0)


def test_stage_fraction_increases_monotonically():
    seen = [pipeline.stage_fraction(name) for name, _ in pipeline.STAGES]
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


def test_unknown_stage_reports_the_end():
    assert pipeline.stage_fraction("nonsense") == pytest.approx(1.0)


@pytest.fixture
def meeting(tmp_path):
    cfg = Config()
    cfg.data_dir = str(tmp_path)
    cfg.asr.final_pass = False
    cfg.llm.model = "fake"
    path = store.new_meeting(tmp_path, "sync")
    store.update_meta(
        path,
        segments=[{"speaker": "Marc", "start": 0.0, "end": 2.0, "text": "we ship Tuesday"}],
        notes=[],
    )
    return cfg, path


def test_progress_never_goes_backwards(meeting, monkeypatch):
    cfg, path = meeting
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (
        {"actions": []} if k.get("schema") or (len(a) > 3 and a[3]) else "## Context\nfine"
    ))
    seen = []
    pipeline.process(path, cfg, progress=lambda step, fraction=None: (
        seen.append(fraction) if fraction is not None else None
    ))
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


def test_token_progress_stays_inside_its_stage(meeting, monkeypatch):
    cfg, path = meeting
    seen = []

    def fake_chat(cfg_, system, user, schema=None, schema_name="r", on_token=None):
        if on_token:
            for count in (1, 50, 500, 5000):
                on_token(count)
        return {"actions": []} if schema else "## Context\nfine"

    monkeypatch.setattr(llm, "chat", fake_chat)
    pipeline.process(path, cfg, progress=lambda step, fraction=None: (
        seen.append((step, fraction)) if fraction is not None else None
    ))

    summarizing = [f for step, f in seen if step.startswith("summarizing (")]
    assert summarizing == sorted(summarizing)
    floor = pipeline.stage_fraction("writing transcripts")
    ceiling = pipeline.stage_fraction("summarizing")
    assert all(floor <= f < ceiling for f in summarizing)


def test_token_counts_are_reported(meeting, monkeypatch):
    cfg, path = meeting
    steps = []

    def fake_chat(cfg_, system, user, schema=None, schema_name="r", on_token=None):
        if on_token:
            on_token(42)
        return {"actions": []} if schema else "text"

    monkeypatch.setattr(llm, "chat", fake_chat)
    pipeline.process(path, cfg, progress=lambda step, fraction=None: steps.append(step))
    assert any("42 tokens" in step for step in steps)


def _sse(chunks):
    lines = []
    for piece in chunks:
        lines.append(
            "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]})
        )
    lines.append("data: [DONE]")
    return "\n".join(lines).encode()


def test_streaming_assembles_the_whole_response(monkeypatch):
    cfg = Config()
    cfg.llm.model = "fake"
    counts = []

    def handler(request):
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=_sse(["Hello", " ", "world"]))

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kw: real_client(**{**kw, "transport": httpx.MockTransport(handler)}),
    )
    result = llm.chat(cfg, "system", "user", on_token=counts.append)
    assert result == "Hello world"
    assert counts == [1, 2, 3]


def test_streaming_ignores_malformed_chunks(monkeypatch):
    cfg = Config()
    cfg.llm.model = "fake"

    def handler(request):
        body = b"data: not json\n" + _sse(["ok"])
        return httpx.Response(200, content=body)

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kw: real_client(**{**kw, "transport": httpx.MockTransport(handler)}),
    )
    assert llm.chat(cfg, "system", "user", on_token=lambda n: None) == "ok"
