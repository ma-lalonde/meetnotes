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

from meetnotes import llm, models
from meetnotes.config import Config


def test_full_name_resolves_to_the_real_repository():
    assert models.full_name("large-v3") == "Systran/faster-whisper-large-v3"
    assert models.full_name("large-v3-turbo").endswith("faster-whisper-large-v3-turbo")


def test_unknown_name_passes_through():
    # faster-whisper also accepts a repo id or a local directory.
    assert models.full_name("/home/me/my-ct2-model") == "/home/me/my-ct2-model"


def test_choices_expose_full_repository_names():
    choices = models.whisper_choices()
    assert choices
    for choice in choices:
        assert choice["repo"]
        assert "/" in choice["repo"] or choice["repo"] == choice["alias"]


def test_choices_do_not_duplicate_aliases():
    aliases = [choice["alias"] for choice in models.whisper_choices()]
    assert len(aliases) == len(set(aliases))


def test_bare_aliases_are_not_offered_twice():
    aliases = {choice["alias"] for choice in models.whisper_choices()}
    assert "turbo" not in aliases  # duplicate of large-v3-turbo
    assert "large" not in aliases  # duplicate of large-v3


def test_english_only_models_are_marked():
    for choice in models.whisper_choices():
        if choice["alias"] in models.ENGLISH_ONLY:
            assert choice["label"].endswith(" EN")
        else:
            assert not choice["label"].endswith(" EN")


def test_a_larger_model_that_is_no_better_is_not_offered():
    # Turbo is large-v2 quality at half the size, so nothing picks large-v2 on
    # purpose. This is the case the frontier exists for.
    offered = {choice["alias"] for choice in models.whisper_choices()}
    assert "large-v3-turbo" in offered
    assert "large-v2" not in offered
    assert "large-v3" in offered  # genuinely more accurate, so it stays


def test_distil_v3_is_dropped_for_v3_5():
    offered = {choice["alias"] for choice in models.whisper_choices()}
    assert "distil-large-v3" not in offered
    assert "distil-large-v3.5" in offered


def test_the_frontier_is_computed_not_listed():
    # A hand-maintained list goes stale the moment a model is added.
    assert models.dominated("large-v2", ["large-v2", "large-v3-turbo"])
    assert not models.dominated("large-v2", ["large-v2"])
    assert not models.dominated("large-v3", list(models.WHISPER_MODELS))


def test_english_and_multilingual_are_ranked_separately():
    # A small English-only model is not "better than" a multilingual one; they
    # do different jobs, so neither can dominate the other.
    assert not models.dominated("small", ["small", "distil-large-v3.5"])
    assert not models.dominated("tiny.en", ["tiny.en", "tiny"])


def test_every_offered_model_keeps_a_real_size():
    for choice in models.whisper_choices():
        assert choice["params_m"] > 0


def test_the_full_list_still_reaches_everything_installed():
    curated = {choice["alias"] for choice in models.whisper_choices()}
    everything = {choice["alias"] for choice in models.whisper_choices(all_models=True)}
    assert curated < everything
    assert "large-v2" in everything


def test_english_only_models_are_hidden_on_a_bilingual_setup():
    # They cannot transcribe French at all, so on a fr+en setup they are
    # choices that can only go wrong.
    cfg = Config()
    cfg.asr.language_mode = "restrict"
    cfg.asr.languages = ["fr", "en"]
    offered = {choice["alias"] for choice in models.whisper_choices(cfg)}
    assert not offered & models.ENGLISH_ONLY
    assert "large-v3-turbo" in offered


def test_english_only_models_are_offered_when_only_english_is_expected():
    cfg = Config()
    cfg.asr.language_mode = "restrict"
    cfg.asr.languages = ["en"]
    offered = {choice["alias"] for choice in models.whisper_choices(cfg)}
    assert "distil-large-v3.5" in offered
    assert "tiny.en" in offered


def test_primary_english_counts_as_english_only():
    cfg = Config()
    cfg.asr.language_mode = "primary"
    cfg.asr.language = "en"
    assert models.english_only_setup(cfg)
    cfg.asr.language = "fr"
    assert not models.english_only_setup(cfg)


def test_detect_anything_is_not_an_english_only_setup():
    # auto can be handed any language, so an .en model would be wrong.
    cfg = Config()
    cfg.asr.language_mode = "auto"
    assert not models.english_only_setup(cfg)


def test_an_english_model_costs_the_same_as_its_twin():
    # Same parameter count, better at English, so on an English-only setup the
    # multilingual one has nothing left to offer.
    for base, twin in models.ENGLISH_TWIN.items():
        assert models.WHISPER_MODELS[twin][1] == models.WHISPER_MODELS[base][1]


def test_english_only_replaces_the_twins_rather_than_listing_both():
    cfg = Config()
    cfg.asr.language_mode = "restrict"
    cfg.asr.languages = ["en"]
    offered = {choice["alias"] for choice in models.whisper_choices(cfg)}
    assert "tiny.en" in offered and "tiny" not in offered
    assert "base.en" in offered and "base" not in offered
    # Turbo and Large v3 have no .en build, so they stay either way.
    assert "large-v3-turbo" in offered
    assert "large-v3" in offered


def test_no_size_is_offered_twice_on_an_english_setup():
    cfg = Config()
    cfg.asr.language_mode = "restrict"
    cfg.asr.languages = ["en"]
    sizes = [choice["params_m"] for choice in models.whisper_choices(cfg)]
    assert len(sizes) == len(set(sizes))


def test_compute_types_are_reported_for_cpu():
    found = models.compute_types("cpu")
    assert "int8" in found or "float32" in found


def test_embedding_models_are_not_offered_for_summarizing():
    assert not models.is_language_model({"id": "nomic-embed-text-v1.5", "type": "embeddings"})
    assert not models.is_language_model({"id": "text-embedding-3-small", "type": ""})
    assert models.is_language_model({"id": "qwen3-8b-instruct", "type": "llm"})


def test_gguf_note_names_both_reasons():
    note = models.gguf_note()
    assert "CTranslate2" in note
    assert "audio/transcriptions" in note


def test_presets_cover_the_common_servers():
    names = {preset["name"] for preset in llm.PRESETS}
    assert {"LM Studio", "Ollama", "Open WebUI"} <= names


def test_applying_a_preset_sets_url_key_and_ttl():
    cfg = Config()
    assert llm.apply_preset(cfg, "Ollama")
    assert cfg.llm.base_url == "http://localhost:11434/v1"
    assert cfg.llm.ttl_seconds == 0


def test_lm_studio_preset_enables_idle_unload():
    cfg = Config()
    llm.apply_preset(cfg, "LM Studio")
    assert cfg.llm.ttl_seconds > 0


def test_unknown_preset_changes_nothing():
    cfg = Config()
    before = cfg.llm.base_url
    assert not llm.apply_preset(cfg, "Nonexistent")
    assert cfg.llm.base_url == before
