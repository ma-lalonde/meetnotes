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

from meetnotes import llm
from meetnotes.config import Config


def test_required_context_rounds_up_to_a_step():
    assert llm.required_context(100) in llm.CONTEXT_STEPS


def test_a_short_transcript_needs_the_smallest_step():
    assert llm.required_context(1000) == 4096


def test_a_long_transcript_needs_more():
    # An hour of speech is roughly 60k characters.
    assert llm.required_context(60000) >= 16384


def test_required_context_never_exceeds_the_largest_step():
    assert llm.required_context(100_000_000) == llm.CONTEXT_STEPS[-1]


def test_reserve_leaves_room_for_the_answer():
    assert llm.required_context(0, reserve=5000) >= 8192


@pytest.fixture
def loads(monkeypatch):
    """Record load attempts; fail any above a given ceiling."""
    attempts = []

    def make(ceiling):
        def fake_load(model, context, gpu="max"):
            attempts.append(context)
            if context > ceiling:
                return False, "failed to allocate KV cache"
            return True, f"loaded {model} with {context}"

        monkeypatch.setattr(llm.shutil, "which", lambda name: "/usr/bin/lms")
        monkeypatch.setattr(llm, "load_model", fake_load)
        monkeypatch.setattr(llm, "catalog", lambda cfg: [
            {"id": cfg.llm.model, "context": 131072}
        ])
        return attempts

    return make


def test_fit_context_uses_the_size_the_transcript_needs(loads):
    attempts = loads(131072)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert attempts == [llm.required_context(60000)]


def test_fit_context_steps_down_when_vram_will_not_take_it(loads):
    # Whether it fits is measured, not predicted: a failed load is the signal.
    attempts = loads(8192)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, detail = llm.fit_context(cfg, 60000)
    assert ok
    assert attempts[0] > 8192
    assert attempts[-1] == 8192
    assert attempts == sorted(attempts, reverse=True)
    assert "8192" in detail


def test_fit_context_raises_when_nothing_loads(loads):
    # Proceeding to the summary anyway only replaces this reason with an
    # opaque out-of-memory from the server.
    attempts = loads(0)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    with pytest.raises(llm.LoadFailed) as caught:
        llm.fit_context(cfg, 60000)
    assert "could not load" in str(caught.value)
    assert attempts


def test_a_failed_load_is_reported_with_the_gpu_state(loads, monkeypatch):
    loads(0)
    monkeypatch.setattr(
        llm, "vram_note", lambda: " GPU: 400 MB free of 8188 MB."
    )
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    with pytest.raises(llm.LoadFailed) as caught:
        llm.fit_context(cfg, 60000)
    assert "400 MB free" in str(caught.value)


def test_a_failed_load_is_a_partial_result_not_a_lost_transcript(loads):
    # LoadFailed is an LlmError, which the pipeline downgrades rather than
    # letting it destroy already-written transcripts.
    loads(0)
    assert issubclass(llm.LoadFailed, llm.LlmError)


def test_fit_context_evicts_the_resident_model_before_loading(loads, monkeypatch):
    # LM Studio holds several models at once and `lms load` adds an instance,
    # so without this the reload competes with the copy it replaces.
    order = []
    loads(131072)
    monkeypatch.setattr(llm, "unload_all", lambda: (order.append("unload"), (True, "ok"))[1])
    real_load = llm.load_model
    monkeypatch.setattr(
        llm, "load_model",
        lambda *a, **k: (order.append("load"), real_load(*a, **k))[1],
    )
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    llm.fit_context(cfg, 60000)
    assert order[0] == "unload"
    assert "load" in order


def test_the_model_maximum_is_respected(loads):
    attempts = loads(131072)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    llm.fit_context(cfg, 500000)
    assert max(attempts) <= 131072


def test_a_configured_ceiling_is_respected(loads):
    attempts = loads(131072)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    cfg.llm.max_context = 8192
    llm.fit_context(cfg, 500000)
    assert max(attempts) == 8192


def test_load_model_without_the_cli_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    ok, detail = llm.load_model("qwen3-8b", 8192)
    assert not ok
    assert "lms CLI" in detail


def test_fit_context_without_the_cli_makes_no_network_call(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    monkeypatch.setattr(llm, "catalog", lambda cfg: pytest.fail("should not query"))
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, detail = llm.fit_context(cfg, 60000)
    assert not ok
    assert "lms CLI" in detail
