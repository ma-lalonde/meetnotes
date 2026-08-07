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

"""Tuning picks what runs, so every branch here changes what the user gets."""

import pytest

from meetnotes import llm, models, tuning
from meetnotes.config import Config
from meetnotes.tuning import Measurement


def timed(alias, realtime, error=""):
    entry = models.WHISPER_MODELS[alias]
    return Measurement(
        alias=alias, label=entry[0], device="cuda", compute_type="float16",
        seconds=realtime * 30, realtime=realtime, error=error,
    )


def test_the_live_model_must_keep_up_with_speech():
    live, _ = tuning.choose_speech([
        timed("large-v3", 0.9),
        timed("large-v3-turbo", 0.3),
        timed("small", 0.05),
    ])
    # Turbo is the most accurate of the two that run under the live bound.
    assert live == "large-v3-turbo"


def test_the_final_pass_may_be_slower_but_not_unbounded():
    _, final = tuning.choose_speech([
        timed("large-v3", 0.9),
        timed("large-v3-turbo", 0.3),
    ])
    assert final == "large-v3"


def test_a_final_model_slower_than_realtime_is_not_chosen():
    # An hour of audio would take longer than the meeting did.
    _, final = tuning.choose_speech([
        timed("large-v3", 2.4),
        timed("large-v3-turbo", 0.8),
    ])
    assert final == "large-v3-turbo"


def test_when_nothing_keeps_up_the_fastest_measured_wins():
    live, _ = tuning.choose_speech([
        timed("large-v3", 4.0),
        timed("medium", 2.0),
    ])
    assert live == "medium"


def test_models_that_failed_are_not_chosen():
    live, final = tuning.choose_speech([
        timed("large-v3", 0.0, error="CUDA out of memory"),
        timed("large-v3-turbo", 0.3),
    ])
    assert live == final == "large-v3-turbo"


def test_no_measurement_at_all_chooses_nothing():
    assert tuning.choose_speech([]) == ("", "")


def test_candidates_are_limited_by_free_vram():
    fits = tuning.candidates("cuda", free_mb=2500, compute_type="float16")
    assert "large-v3" not in fits          # about 3.9 GB
    assert "large-v3-turbo" in fits        # about 2.0 GB
    assert fits == sorted(fits, key=lambda a: -models.WHISPER_MODELS[a][1])


def test_a_card_with_nothing_free_offers_only_the_smallest():
    fits = tuning.candidates("cuda", free_mb=150, compute_type="float16")
    assert fits == ["base", "tiny"] or fits == ["tiny"]


def test_cpu_is_not_limited_by_vram():
    assert "large-v3" in tuning.candidates("cpu", free_mb=0, compute_type="int8")


def test_english_only_models_are_never_auto_selected():
    # Auto-tuning must not silently make a bilingual setup English-only.
    for device, free in (("cuda", 8000), ("cpu", 0)):
        for alias in tuning.candidates(device, free, "float16"):
            assert alias not in models.ENGLISH_ONLY


def test_int8_is_costed_lower_than_float16():
    assert (
        tuning.vram_cost_mb("large-v3-turbo", "int8")
        < tuning.vram_cost_mb("large-v3-turbo", "float16")
    )


def test_an_unclassified_model_has_no_claimed_cost():
    assert tuning.vram_cost_mb("some-local-directory", "float16") == 0


@pytest.fixture
def studio(monkeypatch):
    """A fake LM Studio catalog with per-model, per-context memory costs."""
    def make(free, costs, entries=None):
        monkeypatch.setattr(llm, "lms_binary", lambda: "/home/x/.lmstudio/bin/lms")
        monkeypatch.setattr(llm, "free_vram_mb", lambda: free)
        monkeypatch.setattr(
            llm, "catalog",
            lambda cfg: entries if entries is not None
            else [{"id": name, "type": "llm"} for name in costs],
        )
        monkeypatch.setattr(
            llm, "estimate_load",
            lambda model, context, gpu="max": (costs.get(model, {}).get(context, 0), ""),
        )
    return make


def test_the_biggest_context_that_fits_is_chosen(studio):
    studio(7600, {"qwen3-9b": {32768: 7000, 16384: 6200, 8192: 5800}})
    model, context, notes = tuning.choose_summary(Config())
    assert (model, context) == ("qwen3-9b", 32768)
    assert notes == []


def test_a_gigabyte_more_vram_changes_the_answer(studio):
    # The exact friction reported: PRIME on-demand freed about 1 GB, and that
    # was the difference between a 4k context and a usable one.
    costs = {"qwen3-9b": {32768: 7000, 16384: 6200, 8192: 5800, 4096: 5600}}
    studio(5700, costs)
    assert tuning.choose_summary(Config())[1] == 4096
    studio(6800, costs)
    assert tuning.choose_summary(Config())[1] == 16384


def test_a_model_that_does_not_fit_at_all_is_skipped(studio):
    studio(5000, {
        "too-big-70b": {32768: 40000, 16384: 39000, 8192: 38000, 4096: 37000},
        "qwen3-4b": {32768: 4500, 16384: 4200, 8192: 4000, 4096: 3900},
    })
    model, context, _ = tuning.choose_summary(Config())
    assert model == "qwen3-4b"
    assert context == 32768


def test_nothing_fitting_is_reported_rather_than_guessed(studio):
    studio(2000, {"qwen3-9b": {32768: 7000, 16384: 6200, 8192: 5800, 4096: 5600}})
    model, context, notes = tuning.choose_summary(Config())
    assert (model, context) == ("", 0)
    assert any("2000 MB" in note for note in notes)


def test_embedding_models_are_not_offered_as_summarizers(studio):
    studio(7600, {"nomic-embed-text": {8192: 500}}, entries=[
        {"id": "nomic-embed-text", "type": "embeddings"},
    ])
    assert tuning.choose_summary(Config())[0] == ""


def test_without_the_cli_nothing_is_claimed(monkeypatch):
    monkeypatch.setattr(llm, "lms_binary", lambda: "")
    model, context, notes = tuning.choose_summary(Config())
    assert (model, context) == ("", 0)
    assert notes


def test_apply_writes_every_field_it_decided(tmp_path, monkeypatch):
    from meetnotes import config as config_module

    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.json")
    cfg = Config()
    plan = tuning.Plan(
        live="large-v3-turbo", final="large-v3", device="cuda",
        compute_type="float16", summary_model="qwen3-9b", summary_context=32768,
    )
    tuning.apply(plan, cfg)

    reloaded = Config.load()
    assert reloaded.asr.live_model == "large-v3-turbo"
    assert reloaded.asr.final_model == "large-v3"
    assert reloaded.asr.device == "cuda"
    assert reloaded.llm.model == "qwen3-9b"
    assert reloaded.llm.max_context == 32768


def test_apply_does_not_clear_a_model_it_could_not_choose(tmp_path, monkeypatch):
    from meetnotes import config as config_module

    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.json")
    cfg = Config()
    cfg.llm.model = "already-chosen"
    tuning.apply(tuning.Plan(live="small", final="small", summary_model=""), cfg)
    assert Config.load().llm.model == "already-chosen"
