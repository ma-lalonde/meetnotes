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
    "medium": ("Medium", 769, 4, "superseded by Turbo, which is 5% larger"),
    "large-v3-turbo": ("Turbo", 809, 5, "large-v2 quality at half the size"),
    "large-v2": ("Large v2", 1550, 5, "superseded by Turbo"),
    "large-v3": ("Large v3", 1550, 6, "most accurate, slowest"),
    # Same parameter count as their multilingual siblings, and better at
    # English for it, but they cannot transcribe anything else. Whisper's
    # README: the .en models "tend to perform better, especially for the
    # tiny.en and base.en models", and "the difference becomes less
    # significant for the small.en and medium.en models" - which is why the
    # gain is only worth noting on the two smallest.
    # Notes describe each model on its own terms: an .en model and the
    # multilingual model it was cut from are never offered together, so a note
    # comparing them would name something not in the list.
    "tiny.en": ("Tiny EN", 39, 1, "lowest accuracy, runs anywhere"),
    "base.en": ("Base EN", 74, 2, "low accuracy"),
    "small.en": ("Small EN", 244, 3, "fast enough for live on CPU"),
    "medium.en": ("Medium EN", 769, 4, "superseded by Distil v3.5, which is smaller"),
    "distil-large-v3": ("Distil v3 EN", 756, 5, "superseded by Distil v3.5"),
    "distil-large-v3.5": ("Distil v3.5 EN", 756, 6, "very fast"),
}

ENGLISH_ONLY = {alias for alias in WHISPER_MODELS if alias.endswith(".en")
                or alias.startswith("distil-")}

# The one cross-class comparison Whisper's README actually supports: an .en
# model against the multilingual model it was cut from, same parameters, better
# at English. Only these exact pairs, because relative English accuracy between,
# say, distil-large-v3.5 and Turbo is not something published anywhere and
# guessing it would put a made-up number in front of a real choice.
ENGLISH_TWIN = {
    "tiny": "tiny.en",
    "base": "base.en",
    "small": "small.en",
    "medium": "medium.en",
}


# How much larger a model may be and still count as displacing a smaller one.
# Strict domination misses the case that matters most in practice: Turbo is 809
# M against Medium's 769 M, five percent more, for a jump from Medium quality to
# large-v2 quality. Being marginally smaller is not a reason to pick a
# materially worse model, so a rival within this margin still displaces it.
SIZE_TOLERANCE = 0.10


def dominated(alias: str, others) -> bool:
    """True when another model is better and not meaningfully larger.

    Large v2 against Turbo is the exact case: same accuracy, twice the size, so
    nothing chooses it on purpose. Medium against Turbo is the near case, which
    the tolerance covers. Computed rather than hand-listed so adding a model
    cannot leave a stale recommendation behind.
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
        if rival_rank < rank:
            continue
        # Same accuracy only displaces at no extra cost; better accuracy
        # displaces if the extra size is within the margin.
        budget = size * (1 + SIZE_TOLERANCE) if rival_rank > rank else size
        if rival_size <= budget and (rival_size < size or rival_rank > rank):
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


def english_only_setup(cfg) -> bool:
    """Whether this configuration will only ever be given English.

    An .en model is genuinely better than its multilingual twin at the same
    size, but it cannot transcribe anything else at all, so it is only ever an
    option for someone who has said they only need English.
    """
    mode = cfg.asr.language_mode
    if mode == "auto":
        return False
    if mode == "primary":
        return cfg.asr.language == "en"
    codes = {code.strip() for code in cfg.asr.languages if code.strip()}
    return codes == {"en"}


def whisper_choices(cfg=None, all_models: bool = False) -> list[dict]:
    """What to offer for a speech step, best-value first.

    Dominated models are dropped by default: a model that is both larger and no
    more accurate than another is a choice with no upside, and offering it only
    invites picking it. Given a config, English-only models are dropped too
    unless the configured languages are English and nothing else, since they
    cannot transcribe the other language at all.
    """
    known = repos()
    pool = [alias for alias in WHISPER_MODELS if alias in known]
    if cfg is not None and not english_only_setup(cfg):
        pool = [alias for alias in pool if alias not in ENGLISH_ONLY]
    elif cfg is not None:
        # English and nothing else, so the two classes collapse into one and a
        # multilingual model whose .en twin is available is strictly the worse
        # of the two: same size, worse at the only language being spoken.
        twinned = {base for base, twin in ENGLISH_TWIN.items() if twin in pool}
        pool = [alias for alias in pool if alias not in twinned]
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


# alias -> (label, what it actually does)
#
# CTranslate2 accepts seven of these, but there are only two decisions worth
# making per device: the normal one, and the one that halves the weights. The
# rest differ in which precision the *non-quantized* layers use, which for a
# Whisper model converted from fp16 weights is not a difference anyone can hear.
PRECISION = {
    "float32": ("Exact", "full precision, twice the memory of float16"),
    "float16": ("Balanced", "half precision throughout, the usual choice on a GPU"),
    "bfloat16": ("Balanced (bf16)", "same size as float16, wider range, needs Ampere or newer"),
    "int8_float16": ("Half memory", "int8 weights, float16 for everything else"),
    "int8_bfloat16": ("Half memory (bf16)", "int8 weights, bfloat16 for everything else"),
    "int8_float32": ("Half memory (fp32 elsewhere)", "int8 weights, float32 for everything else"),
    "int8": ("Smallest", "int8 weights, the rest at the model's own precision"),
    "int16": ("16-bit integer", "Intel CPUs with the MKL backend only"),
}

# The ones worth offering per device, default first.
PRECISION_CURATED = {
    "cuda": ["float16", "int8_float16"],
    "cpu": ["int8", "float32"],
}

# CTranslate2: "float16, bfloat16: Convert to float32 on CPU". Choosing one on
# CPU therefore costs float32 memory and buys nothing, which is worth saying out
# loud rather than leaving in the list looking like an option.
PRECISION_TRAPS = {
    "cpu": {
        "float16": "converted to float32 on CPU, so this costs float32 memory",
        "bfloat16": "converted to float32 on CPU, so this costs float32 memory",
    },
}


def precision_choices(device: str, all_types: bool = False) -> list[dict]:
    """Compute types worth offering for this device, best default first.

    Only what CTranslate2 reports as supported here, so an option that would
    silently fall back to something else is never presented as a choice.
    """
    supported = compute_types(device)
    traps = PRECISION_TRAPS.get(device, {})
    wanted = PRECISION_CURATED.get(device, PRECISION_CURATED["cpu"])
    order = [a for a in wanted if a in supported]
    if not order:
        # An old card, or a CUDA build that reports neither float16 nor
        # int8_float16. Offering only "Automatic" would hide the choice
        # entirely, so fall back to whatever this machine does support.
        order = [a for a in supported if a not in traps]
    if all_types:
        order += [a for a in supported if a not in order]
    out = [{"alias": "auto", "label": "Automatic", "note": "chosen from the device"}]
    for alias in order:
        label, note = PRECISION.get(alias, (alias, ""))
        if alias in traps:
            note = traps[alias]
        out.append({"alias": alias, "label": label, "note": note})
    return out


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
