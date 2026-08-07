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

"""Pick models by measuring this machine rather than asking the user to guess.

The friction this removes is real and specific: switching NVIDIA PRIME to
on-demand frees around a gigabyte, which is the difference between a 4k context
and a usable one. Nobody should have to discover that by hitting a wall, and a
setting that only reveals itself as "failed to load model" will lose people.

Two halves, deliberately different. The speech model is *timed*, because what
matters for live transcription is whether it keeps up with speech, and that
depends on the machine, not on a number anyone publishes. The language model is
*estimated*, because LM Studio will tell us what a load costs without loading
it, and timing every candidate would mean loading each one.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import asr, hardware, llm, models


# Model sizing lives with the model table, since that is where it is also used
# to decide which models are worth offering at all.
vram_cost_mb = models.vram_cost_mb


def speech_vram_mb() -> int:
    """The card's size, for deciding which speech models are worth trying.

    Not free memory: recognition runs in its own process, after the language
    model has been unloaded, so what is resident when tuning runs does not
    limit what recognition can load later.
    """
    gpus = hardware.nvidia()
    if not gpus:
        return 0
    return models.vram_budget(gpus[0]["vram_mb"], gpus[0].get("free_mb", 0))

# A live model must transcribe faster than speech arrives or it falls behind
# without limit. Below this share of realtime it keeps up with headroom.
LIVE_REALTIME_TARGET = 0.5

# The final pass runs after the meeting, so latency is cheap but not free: at
# worse than realtime an hour of audio takes more than an hour to transcribe,
# which nobody waits for.
FINAL_REALTIME_TARGET = 1.0


@dataclass
class Measurement:
    alias: str
    label: str
    device: str
    compute_type: str
    seconds: float = 0.0
    realtime: float = 0.0
    vram_mb: int = 0
    error: str = ""

    @property
    def keeps_up(self) -> bool:
        return bool(self.realtime) and self.realtime <= LIVE_REALTIME_TARGET


@dataclass
class Plan:
    live: str = ""
    final: str = ""
    device: str = "cpu"
    compute_type: str = "int8"
    summary_model: str = ""
    summary_context: int = 0
    measurements: list[Measurement] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def candidates(device: str, free_mb: int, compute_type: str, cfg=None) -> list[str]:
    """Speech models worth timing, largest first.

    Only the frontier, and on GPU only what fits in the memory actually free
    right now, which is the number that changed when PRIME switched mode.
    English-only models are included only for an English-only configuration,
    and never when no configuration is given: silently making a bilingual setup
    English-only is worse than picking a slightly smaller model.
    """
    pool = [c["alias"] for c in models.whisper_choices(cfg, device=device, free_mb=free_mb)
            if cfg is not None or c["alias"] not in models.ENGLISH_ONLY]
    if device == "cuda" and free_mb:
        pool = [a for a in pool if vram_cost_mb(a, compute_type) <= free_mb]
    return sorted(pool, key=lambda a: -models.WHISPER_MODELS[a][1])


def time_model(sample: Path, alias: str, cfg, device: str, compute_type: str) -> Measurement:
    """Transcribe the sample once and report how long it took, in a child.

    In a child because CTranslate2 never returns GPU memory to the driver while
    the process lives, so timing four models in one process would measure the
    fourth against a card the first three are still holding.
    """
    entry = models.WHISPER_MODELS.get(alias, (alias, 0, 0, ""))
    out = Measurement(
        alias=alias, label=entry[0], device=device, compute_type=compute_type,
        vram_mb=vram_cost_mb(alias, compute_type),
    )
    plan = {"final_model": alias, "live_model": alias,
            "device": device, "compute_type": compute_type}
    started = time.monotonic()
    try:
        segments = asr.transcribe_file_isolated(sample, cfg, plan)
    except Exception as exc:
        out.error = f"{type(exc).__name__}: {exc}"
        return out
    out.seconds = time.monotonic() - started
    spoken = max((s.get("end", 0.0) for s in segments), default=0.0)
    if spoken:
        out.realtime = round(out.seconds / spoken, 3)
    return out


def measure(sample: Path, cfg, log=None) -> list[Measurement]:
    """Time every candidate speech model on one recording."""
    device = "cuda" if hardware.cuda_runtime_ok() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    free = speech_vram_mb() if device == "cuda" else 0
    found = []
    for alias in candidates(device, free, compute_type, cfg):
        if log:
            log(f"timing {models.label(alias)} on {device}")
        result = time_model(sample, alias, cfg, device, compute_type)
        if log:
            log(f"  {result.realtime or '?'}x realtime"
                + (f", {result.error}" if result.error else ""))
        found.append(result)
    return found


def choose_speech(found: list[Measurement]) -> tuple[str, str]:
    """(live, final) from measurements: the most accurate that is fast enough.

    The two passes have different budgets, not different goals. Live has to run
    faster than speech arrives; the final pass only has to finish in less time
    than the meeting took. Both pick the most accurate model inside that bound,
    and fall back to the fastest measured when nothing meets it.
    """
    worked = [m for m in found if not m.error and m.realtime]
    if not worked:
        return "", ""
    rank = {m.alias: models.WHISPER_MODELS.get(m.alias, ("", 0, 0, ""))[2] for m in worked}

    def best_within(limit: float) -> Measurement:
        inside = [m for m in worked if m.realtime <= limit]
        if not inside:
            return min(worked, key=lambda m: m.realtime)
        return max(inside, key=lambda m: (rank[m.alias], -m.realtime))

    return (
        best_within(LIVE_REALTIME_TARGET).alias,
        best_within(FINAL_REALTIME_TARGET).alias,
    )


def choose_summary(cfg, want_context: int = 32768, log=None) -> tuple[str, int, list[str]]:
    """Largest language model that fits, with the biggest context it allows.

    Asks lms what each load would cost rather than loading anything, so this
    stays cheap enough to run whenever the machine changes.
    """
    notes = []
    if not llm.lms_binary():
        return "", 0, ["the lms CLI was not found, so nothing can be measured"]
    free = llm.free_vram_mb()
    if not free:
        return "", 0, ["no NVIDIA GPU reported, so there is nothing to size against"]

    try:
        entries = [e for e in llm.catalog(cfg) if models.is_language_model(e)]
    except llm.LlmError as exc:
        return "", 0, [str(exc)]

    sizes = [s for s in llm.CONTEXT_STEPS if s <= want_context] or [llm.CONTEXT_STEPS[0]]
    best = ("", 0, 0)
    for entry in entries:
        name = entry.get("id") or ""
        if not name:
            continue
        for size in reversed(sizes):
            cost, _ = llm.estimate_load(name, size, cfg.llm.gpu_offload)
            if not cost:
                continue
            if cost > free:
                continue
            if log:
                log(f"{name} at {size}: {cost} MB of {free} MB free")
            # Prefer more context first, then the model that uses more of the
            # card, since that is the larger model at the same context.
            if (size, cost) > (best[1], best[2]):
                best = (name, size, cost)
            break
    if not best[0]:
        notes.append(f"nothing in the catalog fits in {free} MB of free VRAM")
    return best[0], best[1], notes


def tune(cfg, sample: Path | None = None, log=None) -> Plan:
    """Work out what this machine should run, measuring where it can."""
    device = "cuda" if hardware.cuda_runtime_ok() else "cpu"
    plan = Plan(
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )
    # Speech models are sized against the card; the language model against what
    # is free, because that one loads into current conditions.
    card = speech_vram_mb()
    free = llm.free_vram_mb()
    if device == "cuda" and card:
        plan.notes.append(f"{card} MB of VRAM on this card, {free} MB free right now")

    if sample and sample.exists():
        plan.measurements = measure(sample, cfg, log=log)
        plan.live, plan.final = choose_speech(plan.measurements)
    if not plan.live:
        # No sample, or nothing ran. The profiles encode the speed judgement
        # that timing would otherwise establish, so the live model keeps them
        # and memory only ever takes options away.
        profile = hardware.PROFILES["gpu" if device == "cuda" else "cpu"]
        plan.live = profile["live_model"]
        plan.final = profile["final_model"]
        fits = candidates(device, card, plan.compute_type, cfg)
        if device == "cuda" and fits:
            if plan.live not in fits:
                # fits is largest first, so this is the best that will load.
                # Speed is not the binding constraint on a GPU; memory is.
                plan.live = fits[0]
            # The final pass runs after the meeting, so it takes the most
            # accurate model that fits rather than the profile default. The
            # passes no longer share weights - each runs in its own process -
            # so there is nothing to gain by reusing the live model here.
            plan.final = max(fits, key=lambda a: models.WHISPER_MODELS[a][2])
            if plan.final != profile["final_model"]:
                plan.notes.append(
                    f"{models.label(plan.final)} fits on this card, so the final "
                    f"pass uses it instead of {models.label(profile['final_model'])}"
                )
        if not sample:
            plan.notes.append(
                "no sample recording, so speech models were sized rather than timed"
            )

    plan.summary_model, plan.summary_context, notes = choose_summary(cfg, log=log)
    plan.notes.extend(notes)
    return plan


def apply(plan: Plan, cfg) -> None:
    cfg.asr.live_model = plan.live
    cfg.asr.final_model = plan.final
    # Both, and in step. hardware.plan gives the explicit device precedence over
    # the profile, so writing only the device left the Profile combo unable to
    # change anything afterwards.
    cfg.asr.profile = "gpu" if plan.device == "cuda" else "cpu"
    cfg.asr.device = plan.device
    cfg.asr.compute_type = plan.compute_type
    if plan.summary_model:
        cfg.llm.model = plan.summary_model
    if plan.summary_context:
        cfg.llm.max_context = plan.summary_context
    cfg.save()
