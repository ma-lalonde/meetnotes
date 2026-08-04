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
        if choice["alias"].endswith(".en") or choice["alias"].startswith("distil-"):
            assert "ENGLISH ONLY" in choice["note"] or choice["note"] == ""


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
