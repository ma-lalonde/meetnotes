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

# alias -> (short label, parameters in millions, accuracy rank, note)
#
# Parameters are OpenAI's published counts for the Whisper family; turbo is
# large-v3 with the decoder cut from 32 layers to 4. Accuracy rank is an
# ordinal, not a score: higher is better, and equal ranks mean "no material
# difference", which is the case for turbo against large-v2 (OpenAI: "across
# languages, the turbo model performs similarly to large-v2").
#
# Ranks are within a language class. An English-only model is not comparable to
# a multilingual one, so the two are ranked separately and the frontier is
# computed per class.
WHISPER_MODELS = {
    "tiny": ("Tiny", 39, 1, "lowest accuracy, runs anywhere"),
    "base": ("Base", 74, 2, "low accuracy"),
    "small": ("Small", 244, 3, "fast enough for live on CPU"),
    "medium": ("Medium", 769, 4, "middle ground"),
    "large-v3-turbo": ("Turbo", 809, 5, "large-v2 quality at half the size"),
    "large-v2": ("Large v2", 1550, 5, "superseded by Turbo"),
    "large-v3": ("Large v3", 1550, 6, "most accurate, slowest"),
    "tiny.en": ("Tiny EN", 39, 1, "lowest accuracy"),
    "base.en": ("Base EN", 74, 2, "low accuracy"),
    "small.en": ("Small EN", 244, 3, ""),
    "medium.en": ("Medium EN", 769, 4, ""),
    "distil-large-v3": ("Distil v3 EN", 756, 5, "superseded by Distil v3.5"),
    "distil-large-v3.5": ("Distil v3.5 EN", 756, 6, "very fast"),
}

ENGLISH_ONLY = {alias for alias in WHISPER_MODELS if alias.endswith(".en")
                or alias.startswith("distil-")}


def dominated(alias: str, others) -> bool:
    """True when another model is no larger and no less accurate.

    Large v2 against Turbo is the case that matters: same accuracy, twice the
    size, so nothing chooses it on purpose. Computed rather than hand-listed so
    adding a model cannot leave a stale recommendation behind.
    """
    entry = WHISPER_MODELS.get(alias)
    if not entry:
        return False
    _, size, rank, _ = entry
    english = alias in ENGLISH_ONLY
    for other in others:
        if other == alias or (other in ENGLISH_ONLY) != english:
            continue
        rival = WHISPER_MODELS.get(other)
        if not rival:
            continue
        _, rival_size, rival_rank, _ = rival
        if rival_size <= size and rival_rank >= rank and (
            rival_size < size or rival_rank > rank
        ):
            return True
    return False


def frontier(aliases=None) -> list[str]:
    """The models worth offering: everything not dominated, largest last."""
    pool = list(aliases if aliases is not None else WHISPER_MODELS)
    kept = [alias for alias in pool if not dominated(alias, pool)]
    return sorted(kept, key=lambda a: (a in ENGLISH_ONLY, WHISPER_MODELS[a][1]))


def label(alias: str) -> str:
    entry = WHISPER_MODELS.get(alias)
    return entry[0] if entry else alias


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


def whisper_choices(all_models: bool = False) -> list[dict]:
    """What to offer for a speech step, best-value first.

    Dominated models are dropped by default: a model that is both larger and no
    more accurate than another is a choice with no upside, and offering it only
    invites picking it.
    """
    known = repos()
    pool = [alias for alias in WHISPER_MODELS if alias in known]
    keep = pool if all_models else frontier(pool)
    out = []
    for alias in sorted(keep, key=lambda a: (a in ENGLISH_ONLY, -WHISPER_MODELS[a][1])):
        short, size, _, note = WHISPER_MODELS[alias]
        out.append({
            "alias": alias, "label": short, "repo": known[alias],
            "params_m": size, "note": note,
        })
    if not all_models:
        return out
    # Anything the installed library knows about that is not classified above.
    # No size or rank is claimed for these, so they cannot be placed on the
    # frontier and are only shown when the full list is asked for. "large" and
    # "turbo" are aliases of models already listed.
    for alias, repo in known.items():
        if alias in WHISPER_MODELS or alias in ("large", "turbo"):
            continue
        out.append({"alias": alias, "label": alias, "repo": repo,
                    "params_m": 0, "note": "unclassified"})
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
