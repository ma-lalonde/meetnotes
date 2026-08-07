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


def test_the_context_is_not_rounded_to_a_power_of_two():
    # Nothing requires one, and rounding 20000 up to 32768 reserves KV cache
    # for 12768 tokens that will never be used.
    asked = llm.required_context(60000)
    assert asked not in llm.CONTEXT_STEPS
    assert asked == int(60000 / llm.CHARS_PER_TOKEN * 1.2) + llm.ANSWER_RESERVE


def test_a_short_transcript_still_gets_room_to_answer():
    # Sizing the window to the transcript alone loaded a five-minute recording
    # at 4096, which the model ran out of before finishing the summary.
    assert llm.required_context(1000) == llm.MIN_CONTEXT
    assert llm.required_context(0) >= 8192


def test_a_long_transcript_needs_more():
    # An hour of speech is roughly 60k characters.
    assert llm.required_context(60000) >= 16384


def test_french_is_costed_above_the_english_rule_of_thumb():
    # Around 4 characters per token for English; French splits more often, and
    # underestimating means a prompt that silently does not fit.
    assert llm.CHARS_PER_TOKEN < 4


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
    def make(ceiling, entries=None, cost=0, free=0):
        attempts, unloaded = [], []
        state = list(entries) if entries is not None else None
        # cost/free default to zero, meaning "unknown", which is what a machine
        # without nvidia-smi reports and which must not gate anything.
        monkeypatch.setattr(llm, "estimate_load", lambda m, c, g="max": (cost, "estimate"))
        monkeypatch.setattr(llm, "free_vram_mb", lambda: free)
        monkeypatch.setattr(llm, "loaded", lambda: [])

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
        # No native REST endpoints, so the CLI fallback is what runs. The REST
        # path has its own tests.
        monkeypatch.setattr(llm, "rest_instances", lambda cfg: [])
        monkeypatch.setattr(llm, "rest_load", lambda cfg, m, c: (False, 0, "no REST"))
        return attempts, unloaded

    return make


@pytest.fixture
def loads(server):
    return lambda ceiling, entries=None: server(ceiling, entries)[0]


def test_a_model_too_big_for_the_card_is_named_as_such(server):
    # "failed to load model" is all LM Studio says. --estimate-only turns that
    # into a number that can be compared against the card.
    attempts, _ = server(0, cost=9200, free=7600)
    cfg = Config()
    cfg.llm.model = "qwen3-14b"
    with pytest.raises(llm.LoadFailed) as caught:
        llm.fit_context(cfg, 60000)
    message = str(caught.value)
    assert "9200 MB" in message
    assert "7600 MB" in message
    assert "gpu_offload" in message
    # Nothing was even attempted: the estimate ruled every size out first.
    assert attempts == []


def test_a_failed_load_is_not_retried_smaller(server):
    # A smaller context that loads is not a success. The server truncates the
    # prompt without saying so, and the result reads as a summary of the whole
    # meeting while covering only the end of it.
    attempts, _ = server(0)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    with pytest.raises(llm.LoadFailed):
        llm.fit_context(cfg, 60000)
    assert len(attempts) == 1
    assert attempts == [llm.required_context(60000)]


def test_an_unknown_estimate_does_not_block_the_load(server):
    # No nvidia-smi, or an lms build that reports nothing: fall back to trying.
    attempts, _ = server(131072, cost=0, free=0)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert attempts == [llm.required_context(60000)]


def test_a_model_that_will_not_unload_stops_the_load(server, monkeypatch):
    # Loading on top of a model that refused to go is exactly the out-of-memory
    # this is meant to prevent, and doing it anyway replaces a specific reason
    # with a generic one.
    attempts, _ = server(131072)
    monkeypatch.setattr(llm, "loaded", lambda: ["qwen3-8b  8.0 GB  loaded"])
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    with pytest.raises(llm.LoadFailed) as caught:
        llm.fit_context(cfg, 60000)
    assert "could not free the GPU" in str(caught.value)
    assert "qwen3-8b" in str(caught.value)
    assert attempts == []


def test_the_unload_is_verified_rather_than_assumed(server, monkeypatch):
    # An accepted unload is not evidence that the memory came back.
    server(131072)
    calls = []
    monkeypatch.setattr(llm, "loaded", lambda: calls.append("checked") or [])
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    llm.fit_context(cfg, 60000)
    assert calls, "nothing asked whether the unload worked"


def test_both_unload_routes_are_tried_not_just_one(monkeypatch):
    # REST for a windowed app that never sees lms on PATH, the CLI for an
    # LM Studio too old to have the endpoint.
    done = []
    monkeypatch.setattr(llm, "rest_instances", lambda cfg: [] if done else [
        {"id": "qwen3-8b", "model": "qwen3-8b", "context": 4096, "size_bytes": 0},
    ])
    monkeypatch.setattr(
        llm, "rest_unload",
        lambda cfg, i: (done.append(f"rest:{i}"), (True, "unloaded"))[1],
    )
    monkeypatch.setattr(llm, "lms_binary", lambda: "/home/x/.lmstudio/bin/lms")
    monkeypatch.setattr(llm, "unload", lambda model="": (done.append("cli"), (True, "ok"))[1])
    monkeypatch.setattr(llm, "loaded", lambda: [])

    cleared, detail = llm.unload_everything(Config())
    assert cleared
    assert done == ["rest:qwen3-8b", "cli"]


def test_the_unload_records_what_the_card_actually_gave_back(server, monkeypatch):
    server(131072, free=1000)
    lines = []
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    llm.fit_context(cfg, 60000, log=lines.append)
    assert any("free VRAM" in line for line in lines)


def test_fit_context_uses_the_size_the_transcript_needs(loads):
    attempts = loads(131072)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert attempts == [llm.required_context(60000)]


def test_a_load_that_will_not_take_the_size_aborts(loads):
    # One attempt. The alternative is a context that loads but cannot hold the
    # transcript, which produces a confident summary of part of the meeting.
    attempts = loads(8192)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    with pytest.raises(llm.LoadFailed):
        llm.fit_context(cfg, 60000)
    assert attempts == [llm.required_context(60000)]


def test_a_short_transcript_is_not_squeezed_into_the_smallest_step(loads):
    attempts = loads(131072)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 2000)
    assert ok
    assert attempts == [llm.MIN_CONTEXT]


def test_a_clamped_load_is_caught_rather_than_used(server, monkeypatch):
    # The load succeeds but the server applies a smaller context than asked.
    entry = {"id": "qwen3-8b", "context": 131072, "state": "loaded",
             "loaded_context": 4096}
    server(131072, [entry])
    monkeypatch.setattr(llm, "rest_load", lambda cfg, m, c: (True, 4096, "clamped"))
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    with pytest.raises(llm.ContextTooSmall) as caught:
        llm.fit_context(cfg, 60000)
    assert "4096" in str(caught.value)


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


def test_everything_is_unloaded_before_the_load(server):
    # `lms load` adds an instance rather than replacing one, so anything left
    # resident is a second copy of weights competing for the same card. The
    # unload is unconditional: deciding it from reported state meant that a
    # missing or stale state silently skipped it.
    attempts, unloaded = server(131072, [loaded("qwen3-8b", 4096)])
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert unloaded == [""]          # "" is unload --all
    assert attempts == [llm.required_context(60000)]


def test_a_model_already_loaded_at_the_right_size_is_still_reloaded(server):
    # Costs one reload. Buys never stacking on a card that reports nothing.
    attempts, unloaded = server(131072, [loaded("qwen3-8b", 32768)])
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert unloaded == [""]
    assert attempts == [llm.required_context(60000)]


def test_a_server_reporting_no_state_at_all_still_unloads(server):
    # The failure that kept recurring: no state field, so nothing looked
    # loaded, so nothing was evicted and the load stacked.
    attempts, unloaded = server(131072, [
        {"id": "qwen3-8b", "context": 131072},
    ])
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert unloaded == [""]
    assert attempts == [llm.required_context(60000)]


def test_a_different_resident_model_is_evicted_too(server):
    attempts, unloaded = server(131072, [
        loaded("some-other-13b", 32768),
        {"id": "qwen3-8b", "context": 131072, "state": "not-loaded", "loaded_context": 0},
    ])
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert unloaded == [""]
    assert attempts == [llm.required_context(60000)]


def test_a_transcript_beyond_the_model_maximum_is_refused(loads):
    # Clamping to the maximum would load fine and summarize part of it.
    attempts = loads(131072)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    with pytest.raises(llm.ContextTooSmall) as caught:
        llm.fit_context(cfg, 500000)
    assert "131072" in str(caught.value)
    assert attempts == []


def test_a_configured_ceiling_below_what_is_needed_is_refused(loads):
    attempts = loads(131072)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    cfg.llm.max_context = 8192
    # Within the model's own maximum, so the cap is what refuses it.
    with pytest.raises(llm.ContextTooSmall) as caught:
        llm.fit_context(cfg, 60000)
    assert "max_context" in str(caught.value)
    assert attempts == []


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


def test_an_unreachable_server_is_reported_not_raised(monkeypatch):
    # Nothing to unload, size or load, and the request may still be going
    # somewhere else. The caller records it and carries on.
    monkeypatch.setattr(llm, "lms_binary", lambda: "")
    monkeypatch.setattr(
        llm, "catalog",
        lambda cfg: (_ for _ in ()).throw(llm.LlmError("connection refused")),
    )
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, detail = llm.fit_context(cfg, 60000)
    assert not ok
    assert "cannot reach" in detail


def test_rest_is_used_before_the_cli(monkeypatch):
    # lms only reaches PATH when `lms bootstrap` has edited a shell profile,
    # which a windowed application never reads. REST needs no such thing.
    monkeypatch.setattr(llm, "lms_binary", lambda: "")
    monkeypatch.setattr(llm, "catalog", lambda cfg: [{"id": "qwen3-8b", "context": 131072}])
    monkeypatch.setattr(llm, "rest_instances", lambda cfg: [])
    monkeypatch.setattr(llm, "estimate_load", lambda m, c, g="max": (0, ""))
    monkeypatch.setattr(llm, "free_vram_mb", lambda: 0)
    monkeypatch.setattr(llm, "_resident_context", lambda cfg: 0)
    asked = {}

    def fake_rest_load(cfg, model, context):
        asked["context"] = context
        return True, context, "ok"

    monkeypatch.setattr(llm, "rest_load", fake_rest_load)
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, detail = llm.fit_context(cfg, 60000)
    assert ok
    assert asked["context"] == llm.required_context(60000)


def test_the_weights_alone_are_checked_against_free_memory(server, monkeypatch):
    # size_bytes comes from the server, so this works with no lms CLI at all.
    server(131072, free=2000)
    monkeypatch.setattr(llm, "model_size_mb", lambda cfg, model: 5200)
    monkeypatch.setattr(llm, "estimate_load", lambda m, c, g="max": (0, ""))
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    with pytest.raises(llm.LoadFailed) as caught:
        llm.fit_context(cfg, 60000)
    message = str(caught.value)
    assert "5200 MB" in message
    assert "2000 MB" in message
    assert "before any context" in message


def test_weights_that_fit_do_not_block_the_load(server, monkeypatch):
    attempts, _ = server(131072, free=8000)
    monkeypatch.setattr(llm, "model_size_mb", lambda cfg, model: 5200)
    monkeypatch.setattr(llm, "estimate_load", lambda m, c, g="max": (0, ""))
    cfg = Config()
    cfg.llm.model = "qwen3-8b"
    ok, _ = llm.fit_context(cfg, 60000)
    assert ok
    assert attempts == [llm.required_context(60000)]


def test_the_rest_url_sits_beside_the_openai_one():
    cfg = Config()
    cfg.llm.base_url = "http://localhost:1234/v1"
    assert llm._v1(cfg, "models/load") == "http://localhost:1234/api/v1/models/load"
    cfg.llm.base_url = "http://box:5000"
    assert llm._v1(cfg, "models") == "http://box:5000/api/v1/models"
