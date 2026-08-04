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

import subprocess

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


def loaded(model, loaded_context, maximum=131072):
    """A catalog entry for a model LM Studio reports as currently loaded."""
    return {"id": model, "context": maximum, "state": "loaded",
            "loaded_context": loaded_context}


@pytest.fixture
def server(monkeypatch):
    """A fake LM Studio. Returns (load_sizes, unloaded_names), both live lists.

    `entries` is what /api/v0/models reports, so a test can put a model in the
    loaded state with a given loaded_context_length.
    """
    def make(ceiling, entries=None):
        attempts, unloaded = [], []
        state = list(entries) if entries is not None else None

        def catalog(cfg):
            if state is None:
                return [{"id": cfg.llm.model, "context": 131072,
                         "state": "", "loaded_context": 0}]
            return state

        def fake_load(model, context, gpu="max"):
            attempts.append(context)
            if context > ceiling:
                return False, "failed to allocate KV cache"
            # A real server reports the new size on the next query, which is
            # what fit_context reads back to confirm the load was honoured.
            if state is not None:
                for entry in state:
                    if entry["id"] == model:
                        entry.update(state="loaded", loaded_context=context)
            return True, f"loaded {model} with {context}"

        def fake_unload(model=""):
            unloaded.append(model)
            for entry in state or []:
                if not model or entry["id"] == model:
                    entry.update(state="not-loaded", loaded_context=0)
            return True, "ok"

        monkeypatch.setattr(llm, "lms_binary", lambda: "/home/x/.lmstudio/bin/lms")
        monkeypatch.setattr(llm, "load_model", fake_load)
        monkeypatch.setattr(llm, "unload", fake_unload)
        monkeypatch.setattr(llm, "catalog", catalog)
        return attempts, unloaded

    return make


@pytest.fixture
def loads(server):
    return lambda ceiling, entries=None: server(ceiling, entries)[0]


def test_fit_context_uses_the_size_the_transcript_needs(loads):
    attempts = loads(131072)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert attempts == [llm.required_context(60000)]


def test_fit_context_steps_down_when_vram_will_not_take_it(loads):
    # Whether it fits is measured, not predicted: a failed load is the signal.
    # The step-down still runs, but the size it lands on is too small for this
    # transcript, so it is reported rather than quietly used.
    attempts = loads(8192)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    with pytest.raises(llm.ContextTooSmall):
        llm.fit_context(cfg, 60000)
    assert attempts[0] > 8192
    assert attempts[-1] == 8192
    assert attempts == sorted(attempts, reverse=True)


def test_a_context_that_cannot_hold_the_transcript_is_refused(loads):
    # Sending it anyway means the server drops the start of the transcript
    # without saying so, and the summary silently covers only the end.
    attempts = loads(4096)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    with pytest.raises(llm.ContextTooSmall) as caught:
        llm.fit_context(cfg, 60000)
    message = str(caught.value)
    assert "4096" in message
    assert str(llm.required_context(60000)) in message
    assert attempts[-1] == 4096


def test_a_short_transcript_is_fine_at_a_small_size(loads):
    attempts = loads(4096)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, detail = llm.fit_context(cfg, 2000)
    assert ok
    assert attempts == [4096]


def test_the_lms_binary_is_found_where_lm_studio_installs_it(tmp_path, monkeypatch):
    # lms only reaches PATH when `lms bootstrap` edits a shell profile, which an
    # application launched from a desktop entry never reads. Looking only on
    # PATH is why loads and unloads silently did nothing.
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    installed = tmp_path / ".lmstudio" / "bin" / "lms"
    installed.parent.mkdir(parents=True)
    installed.write_text("#!/bin/sh\n")
    installed.chmod(0o755)
    monkeypatch.setattr(llm, "LMS_PATHS", (str(tmp_path / ".lmstudio/bin/lms"),))
    assert llm.lms_binary() == str(installed)


def test_path_wins_when_the_cli_is_on_it(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda name: "/usr/local/bin/lms")
    assert llm.lms_binary() == "/usr/local/bin/lms"


def test_a_non_executable_candidate_is_not_used(tmp_path, monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    dud = tmp_path / "lms"
    dud.write_text("")
    dud.chmod(0o644)
    monkeypatch.setattr(llm, "LMS_PATHS", (str(dud),))
    assert llm.lms_binary() == ""


def test_load_passes_only_flags_lms_accepts(monkeypatch):
    # lms load takes [path], --ttl, --gpu, --context-length, --identifier,
    # --estimate-only and --host. A stray --yes fails the whole command.
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return subprocess.CompletedProcess(args, 0, "ok", "")

    monkeypatch.setattr(llm, "lms_binary", lambda: "/home/x/.lmstudio/bin/lms")
    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    ok, _ = llm.load_model("qwen3-8b", 32768)
    assert ok
    allowed = {"--ttl", "--gpu", "--context-length", "--identifier",
               "--estimate-only", "--host"}
    flags = {a.split("=")[0] for a in seen["args"] if a.startswith("--")}
    assert flags <= allowed
    assert "--context-length=32768" in seen["args"]


def test_without_the_cli_the_reason_names_the_fix(monkeypatch):
    monkeypatch.setattr(llm, "lms_binary", lambda: "")
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, detail = llm.fit_context(cfg, 60000)
    assert not ok
    # The symptom is the server loading at its own default, so the message has
    # to point at the cause rather than at the context size.
    assert "bootstrap" in detail


def test_a_load_that_is_silently_clamped_is_caught(server, monkeypatch):
    # A zero exit says the command was accepted, not that the size was honoured.
    entry = loaded("qwen3-8b", 0)
    entry["state"] = "not-loaded"
    attempts, _ = server(131072, [entry])
    monkeypatch.setattr(
        llm, "load_model",
        lambda model, context, gpu="max": (
            entry.update(state="loaded", loaded_context=4096), (True, "ok")
        )[1],
    )
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    with pytest.raises(llm.ContextTooSmall) as caught:
        llm.fit_context(cfg, 60000)
    assert "4096" in str(caught.value)


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


def test_an_adequate_resident_model_is_used_as_is(server):
    # `lms load` adds an instance rather than replacing one, so loading a model
    # that is already loaded puts a second copy of the weights on the card.
    attempts, unloaded = server(131072, [loaded("qwen3-8b", 32768)])
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, detail = llm.fit_context(cfg, 60000)
    assert ok
    assert attempts == []
    assert unloaded == []
    assert "already loaded" in detail


def test_a_resident_model_with_too_little_context_is_replaced_not_stacked(server):
    attempts, unloaded = server(131072, [loaded("qwen3-8b", 4096)])
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert unloaded == ["qwen3-8b"]
    assert attempts == [llm.required_context(60000)]


def test_a_resident_model_of_unreported_size_is_replaced(server):
    # loaded_context_length is absent from the published example response, so
    # a loaded model with no size is treated as unknown, never as adequate.
    attempts, unloaded = server(131072, [loaded("qwen3-8b", 0)])
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    llm.fit_context(cfg, 60000)
    assert unloaded == ["qwen3-8b"]
    assert attempts == [llm.required_context(60000)]


def test_a_different_resident_model_is_evicted_to_make_room(server):
    attempts, unloaded = server(131072, [
        loaded("some-other-13b", 32768),
        {"id": "qwen3-8b", "context": 131072, "state": "not-loaded", "loaded_context": 0},
    ])
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert unloaded == ["some-other-13b"]
    assert attempts == [llm.required_context(60000)]


def test_an_adequate_resident_model_is_kept_even_with_others_loaded(server):
    # Nothing is evicted when no load is needed: the summary can go straight
    # to the instance that is already up.
    attempts, unloaded = server(131072, [
        loaded("some-other-13b", 8192),
        loaded("qwen3-8b", 65536),
    ])
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert (attempts, unloaded) == ([], [])


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
    # The cap holds, and capping below what the transcript needs is reported
    # rather than used, since the shortfall comes from a setting either way.
    with pytest.raises(llm.ContextTooSmall) as caught:
        llm.fit_context(cfg, 500000)
    assert max(attempts) == 8192
    assert "max_context" in str(caught.value)


def test_a_ceiling_above_what_is_needed_does_not_inflate_the_load(loads):
    # max_context is a cap, not a target: a transcript that fits in 8192 is
    # loaded at 8192 even when 16384 is allowed.
    attempts = loads(131072)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    cfg.llm.max_context = 16384
    ok, _ = llm.fit_context(cfg, 20000)
    assert ok
    assert attempts == [llm.required_context(20000)]
    assert attempts[0] <= 16384


def test_load_model_without_the_cli_is_reported_not_raised(monkeypatch):
    # Patch the lookup, not shutil.which: the CLI is also found off PATH, so
    # stubbing which alone still finds a real install on a developer machine.
    monkeypatch.setattr(llm, "lms_binary", lambda: "")
    ok, detail = llm.load_model("qwen3-8b", 8192)
    assert not ok
    assert "lms CLI" in detail


def test_fit_context_without_the_cli_makes_no_network_call(monkeypatch):
    monkeypatch.setattr(llm, "lms_binary", lambda: "")
    monkeypatch.setattr(llm, "catalog", lambda cfg: pytest.fail("should not query"))
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, detail = llm.fit_context(cfg, 60000)
    assert not ok
    assert "lms CLI" in detail
