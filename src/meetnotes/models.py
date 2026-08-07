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
    "large-v1": ("Large v1", 1550, 4, "superseded by v2 and v3 at the same size"),
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
    "distil-large-v2": ("Distil v2 EN", 756, 4, "superseded by Distil v3.5"),
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

# Relative speed from Whisper's README, large = 1x. The number that matters:
# Turbo is ~8x against Tiny's ~10x, so on hardware that can hold Turbo the
# smaller models buy a quarter more speed for several tiers of accuracy. They
# exist to fit in less memory, not to go faster.
RELATIVE_SPEED = {
    "tiny": 10, "tiny.en": 10,
    "base": 7, "base.en": 7,
    "small": 4, "small.en": 4,
    "medium": 2, "medium.en": 2,
    "large-v3-turbo": 8,
    "large-v1": 1, "large-v2": 1, "large-v3": 1,
    "distil-large-v2": 6, "distil-large-v3": 6, "distil-large-v3.5": 6,
}


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


# Bytes of VRAM per million parameters at float16, plus activations and
# workspace. Calibrated against Turbo (809 M, about 2.0 GB resident) and
# extrapolated linearly; it decides which models are worth offering and timing,
# not what actually gets loaded.
MB_PER_MILLION_FP16 = 2.5
COMPUTE_SCALE = {"float16": 1.0, "int8_float16": 0.6, "int8": 0.55, "float32": 2.0}

# The model a GPU is expected to run. Anything less accurate than this exists to
# fit in less memory, so once it fits there is no reason to offer them.
GPU_ANCHOR = "large-v3-turbo"

def vram_budget(total_mb: int, free_mb: int = 0) -> int:
    """What the card has, not what happens to be free right now.

    Sizing the menu to a snapshot is why an 8 GB card stops offering Large v3
    the moment LM Studio loads a model. By the time recognition runs, the
    language model has been unloaded and recognition has its own process, so
    the card is what it always was. A list that changes because another
    application was resident when the tab opened is the friction this is meant
    to remove.
    """
    return total_mb or free_mb


def vram_cost_mb(alias: str, compute_type: str) -> int:
    """Rough resident size for a speech model. Zero when the size is unknown."""
    entry = WHISPER_MODELS.get(alias)
    if not entry:
        return 0
    return int(entry[1] * MB_PER_MILLION_FP16 * COMPUTE_SCALE.get(compute_type, 1.0))


def hardware_pool(device: str, free_mb: int, pool: list[str]) -> tuple[list[str], str]:
    """Narrow the list to what this machine should actually be offered.

    On a GPU, speed is never the binding constraint: Turbo runs at roughly 8x
    against Tiny's 10x, so a card with room for Turbo gains nothing from the
    smaller models and loses several tiers of accuracy to them. Memory is the
    only real limit, so the list becomes Turbo and anything more accurate that
    still fits.

    On CPU it is the opposite. The live pass has to beat speech in absolute
    terms, not relative to another model, and how close a given CPU gets is not
    something that can be read off a table. The ladder stays, and
    `meetnotes tune --record` is how it gets settled by measurement.
    """
    if device != "cuda" or not free_mb:
        return pool, ""

    fits = [a for a in pool if vram_cost_mb(a, "float16") <= free_mb]
    if GPU_ANCHOR not in fits:
        # Not enough room for the model this would otherwise settle on, so the
        # smaller ones are the point rather than clutter.
        return fits or pool, ""

    anchor_rank = WHISPER_MODELS[GPU_ANCHOR][2]
    kept = [a for a in fits if WHISPER_MODELS[a][2] >= anchor_rank]
    return kept, (
        f"This GPU has room for {WHISPER_MODELS[GPU_ANCHOR][0]}, which is about "
        f"as fast as the small models and far more accurate, so only it and "
        f"anything better are offered."
    )


def why_not_offered(alias: str, cfg=None, device: str = "", vram_mb: int = 0) -> str:
    """Why a configured model is missing from the list. Empty if it is not.

    A model that is set but unlisted needs a reason next to it, or the entry
    reads as an error in the application rather than a considered exclusion.
    """
    if alias not in WHISPER_MODELS:
        return "not a model this knows about; left as configured"
    installed = [a for a in WHISPER_MODELS if a in repos()]
    if alias not in frontier(installed):
        _, _, _, note = WHISPER_MODELS[alias]
        return f"retired: {note}" if note else "retired: another model beats it for less"
    if cfg is not None and alias in ENGLISH_ONLY and not english_only_setup(cfg):
        return "English only, and more than English is expected"
    if cfg is not None and alias not in ENGLISH_ONLY and english_only_setup(cfg):
        twin = ENGLISH_TWIN.get(alias)
        if twin:
            return f"{WHISPER_MODELS[twin][0]} is the same size and better at English"
    cost = vram_cost_mb(alias, "float16")
    if device == "cuda" and vram_mb and cost > vram_mb:
        return f"needs about {cost} MB, and this GPU has {vram_mb} MB"
    if device == "cuda" and vram_mb:
        anchor = WHISPER_MODELS[GPU_ANCHOR][0]
        return f"{anchor} fits on this GPU and is faster as well as more accurate"
    return "not offered for this machine"


def whisper_choices(cfg=None, all_models: bool = False, device: str = "",
                    free_mb: int = 0) -> list[dict]:
    """What to offer for a speech step, best-value first.

    Two kinds of filtering, and only one of them is negotiable.

    A model another one beats without costing meaningfully more is never the
    right answer on any hardware, in any language, so it is not offered at all
    and `all_models` does not bring it back. Large v2 and Medium are both that:
    Turbo matches large-v2 at half the size and beats Medium for five percent
    more.

    The rest is situational. English-only models are wrong for a bilingual
    setup and right for an English one; the small models are wrong on a card
    with room for Turbo and right on one without. Those are what `all_models`
    turns off, for someone who wants to choose against the recommendation.
    """
    known = repos()
    installed = [alias for alias in WHISPER_MODELS if alias in known]
    # Applied first and never waived: no setting makes these the right choice.
    pool = frontier(installed)
    if cfg is not None and not english_only_setup(cfg):
        pool = [alias for alias in pool if alias not in ENGLISH_ONLY]
    elif cfg is not None:
        # English and nothing else, so the two classes collapse into one and a
        # multilingual model whose .en twin is available is strictly the worse
        # of the two: same size, worse at the only language being spoken.
        twinned = {base for base, twin in ENGLISH_TWIN.items() if twin in pool}
        pool = [alias for alias in pool if alias not in twinned]
    if not all_models and device:
        pool, _ = hardware_pool(device, free_mb, pool)
    # Re-run: collapsing the English twins can leave a new model dominated.
    keep = frontier(pool)
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
