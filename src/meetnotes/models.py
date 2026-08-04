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

"""The models that can be selected for each step.

No size or memory estimates: faster-whisper reports none, LM Studio reports
none, and guessing them from a model name was worse than useless.
"""

# Fallback if faster-whisper's table cannot be read. Kept in sync by
# resolve_catalog(), which prefers the installed library's own mapping.
FALLBACK_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
}

# Presentation order and a one-line character note per model.
WHISPER_ORDER = [
    ("large-v3-turbo", "multilingual, fast, close to large-v2 quality"),
    ("large-v3", "multilingual, most accurate, slowest"),
    ("large-v2", "multilingual, superseded by v3"),
    ("medium", "multilingual, middle ground"),
    ("small", "multilingual, fast enough for live on CPU"),
    ("base", "multilingual, low accuracy"),
    ("tiny", "multilingual, lowest accuracy"),
    ("distil-large-v3.5", "ENGLISH ONLY, very fast"),
    ("distil-large-v3", "ENGLISH ONLY, very fast"),
    ("medium.en", "ENGLISH ONLY"),
    ("small.en", "ENGLISH ONLY"),
    ("base.en", "ENGLISH ONLY"),
    ("tiny.en", "ENGLISH ONLY"),
]

ENGLISH_ONLY = {
    alias for alias, note in WHISPER_ORDER if "ENGLISH ONLY" in note
}


def repos() -> dict[str, str]:
    """Alias to HuggingFace repository, from the installed faster-whisper."""
    try:
        from faster_whisper import utils

        table = getattr(utils, "_MODELS", None)
        if isinstance(table, dict) and table:
            return dict(table)
    except ImportError:
        pass
    return dict(FALLBACK_REPOS)


def full_name(alias: str) -> str:
    """The repository a shorthand name resolves to.

    Anything unrecognised is passed through, since faster-whisper also accepts
    a repository id or a local directory.
    """
    return repos().get(alias, alias)


def whisper_choices() -> list[dict]:
    known = repos()
    out = []
    for alias, note in WHISPER_ORDER:
        if alias not in known:
            continue
        out.append({"alias": alias, "repo": known[alias], "note": note})
    # Anything the installed library knows about that is not in the order list.
    for alias, repo in known.items():
        if alias in {choice["alias"] for choice in out} or alias in ("large", "turbo"):
            continue
        out.append({"alias": alias, "repo": repo, "note": ""})
    return out


def compute_types(device: str) -> list[str]:
    try:
        import ctranslate2

        found = sorted(ctranslate2.get_supported_compute_types(device))
        return found or ["float32"]
    except Exception:
        return ["float32", "int8"]


def is_language_model(entry: dict) -> bool:
    """Embedding models cannot summarize, so they are not offered."""
    kind = (entry.get("type") or "").lower()
    if kind in ("embeddings", "embedding"):
        return False
    identifier = (entry.get("id") or "").lower()
    return "embed" not in identifier


def gguf_note() -> str:
    return (
        "LM Studio's whisper builds are GGUF, which this cannot use.\n"
        "\n"
        "Two separate reasons. CTranslate2, the engine behind faster-whisper, "
        "converts from Fairseq, Marian, OpenNMT and Transformers into its own "
        "model.bin format; GGUF is llama.cpp's format and is not among them. "
        "And LM Studio has no /v1/audio/transcriptions endpoint, so it cannot "
        "serve a speech model to anything regardless of format.\n"
        "\n"
        "The size saving you are after is available here as Precision: int8 "
        "roughly halves float16 with little quality cost on turbo. Running the "
        "GGUF itself would mean adding whisper.cpp as a second engine."
    )
