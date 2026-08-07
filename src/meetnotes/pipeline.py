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

import inspect
import json
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import artifacts, asr, hardware, llm, outputs, prompts, store


def _audio_fingerprint(path: Path, meta: dict, plan: dict, cfg) -> str:
    stats = []
    for label, filename in sorted(meta.get("tracks", {}).items()):
        target = path / "audio" / filename
        stat = target.stat() if target.exists() else None
        stats.append([label, filename, stat.st_size if stat else 0])
    return artifacts.sha(
        stats, plan["final_model"], cfg.asr.language, cfg.asr.final_beam_size,
        sorted(cfg.asr.vocabulary),
    )


# Rough share of total time per stage, so the bar moves at a believable rate
# rather than jumping. Transcription dominates; the calendar is instant.
STAGES = [
    ("transcribing", 0.55),
    ("writing transcripts", 0.05),
    ("summarizing", 0.20),
    ("extracting actions", 0.18),
    ("writing calendar", 0.02),
]


RAW = "raw_output"
# Artifacts that used to sit at the top level of a meeting folder.
MOVED = {
    "transcription.md": f"{RAW}/transcription.md",
    "transcription_cleaned.md": f"{RAW}/transcription_cleaned.md",
    "notes.md": f"{RAW}/notes.md",
}


def migrate_layout(path: Path, meta: dict) -> list[str]:
    """Move previously written files into raw_output.

    Anything hand-edited moves with its edits intact rather than being
    regenerated, and is left flagged as hand-edited in its new location.
    """
    moved = []
    records = meta.setdefault("artifacts", {})
    for old, new in MOVED.items():
        source = path / old
        if not source.exists():
            records.pop(old, None)
            continue
        target = path / new
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            if old in records:
                records[new] = records[old]
            moved.append(new)
        elif records.get(old, {}).get("output_hash") == artifacts.file_hash(source):
            # Already regenerated in the new place and this copy is untouched.
            source.unlink()
        records.pop(old, None)
    return moved


def _reporter(progress):
    """Accept both progress(step) and progress(step, fraction).

    The fraction was added later; callers that only want the label should not
    have to change.
    """
    if progress is None:
        return None
    try:
        wanted = len(inspect.signature(progress).parameters)
    except (TypeError, ValueError):
        wanted = 2
    if wanted >= 2:
        return progress
    return lambda step, fraction=None: progress(step)


def stage_fraction(name: str) -> float:
    """Fraction complete once the named stage has finished."""
    total = 0.0
    for stage, weight in STAGES:
        total += weight
        if stage == name:
            return total
    return total


def _slug(text: str, index: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "action"
    return f"{index:02d}-{base}.ics"


def transcribe(path: Path, cfg, force: bool = False, progress=None) -> list[dict]:
    """Full-file pass over every track. Returns merged segments."""
    progress = _reporter(progress)
    meta = store.read_meta(path)
    plan = hardware.plan(cfg)
    if not plan["final_model"]:
        return meta.get("segments", [])

    fingerprint = artifacts.sha(
        _audio_fingerprint(path, meta, plan, cfg), cfg.asr.multilingual
    )
    record = meta.get("artifacts", {}).get("segments", {})
    if not force and record.get("fingerprint") == fingerprint and meta.get("segments"):
        return meta["segments"]

    segments = []
    for label, filename in meta.get("tracks", {}).items():
        track = path / "audio" / filename
        if not track.exists():
            continue
        if progress:
            progress(f"transcribing {label}", stage_fraction("transcribing") * 0.5)
        # On GPU this runs in a child process: its exit is the only thing that
        # gives the VRAM back before the language model needs it.
        run = asr.transcribe_file_isolated if asr.isolated(cfg, plan) else asr.transcribe_file
        for seg in run(track, cfg, plan):
            segments.append({**seg, "speaker": label})
    segments.sort(key=lambda s: s["start"])

    meta["segments"] = segments
    meta["run"] = hardware.provenance(cfg, plan)
    meta.setdefault("artifacts", {})["segments"] = {
        "fingerprint": fingerprint,
        "output_hash": artifacts.sha(segments),
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    store.write_meta(path, meta)
    return segments


def process(path: Path, cfg, force: bool = False, with_llm: bool = True, progress=None) -> dict:
    """Idempotent post-processing. Safe to call repeatedly."""
    report: dict[str, str] = {}
    progress = _reporter(progress)
    with store.exclusive(path):
        store.update_meta(path, state="transcribing", error="")
        segments = transcribe(path, cfg, force=force, progress=progress)
        meta = store.read_meta(path)
        if progress:
            progress("writing transcripts", stage_fraction("transcribing"))

        migrate_layout(path, meta)

        base = artifacts.sha(segments, meta.get("notes", []), meta.get("run", {}))
        report[f"{RAW}/transcription.md"] = artifacts.ensure(
            path, meta, f"{RAW}/transcription.md", base,
            lambda: outputs.render_transcript(meta, segments, clean=False), force,
        )
        report[f"{RAW}/transcription_cleaned.md"] = artifacts.ensure(
            path, meta, f"{RAW}/transcription_cleaned.md",
            artifacts.sha(base, outputs.filler_version()),
            lambda: outputs.render_transcript(meta, segments, clean=True), force,
        )
        report[f"{RAW}/notes.md"] = artifacts.ensure(
            path, meta, f"{RAW}/notes.md", artifacts.sha(meta.get("notes", [])),
            lambda: outputs.render_notes(meta), force,
        )
        report["transcription_cleaned_with_notes.md"] = artifacts.ensure(
            path, meta, "transcription_cleaned_with_notes.md",
            artifacts.sha(base, outputs.filler_version(), "interleaved-split"),
            lambda: outputs.render_transcript_with_notes(meta, segments), force,
        )
        store.write_meta(path, meta)

        if not with_llm:
            store.update_meta(path, state="done")
            return report

        # Transcripts are already on disk. A missing or unreachable language
        # model must not undo them, so LLM failures downgrade to a partial
        # result instead of propagating.
        try:
            report.update(_summarize(path, meta, segments, cfg, force, progress))
            store.update_meta(path, state="done", error="")
        except llm.LlmError as exc:
            store.write_meta(path, meta)
            store.update_meta(path, state="transcribed", error=str(exc))
            report["summary.md"] = f"skipped: {exc}"
        return report


@contextmanager
def _model_log(path: Path, cfg, source: str):
    """Record how the context size was decided, into the meeting folder.

    Every step here is a subprocess or an HTTP call whose result is otherwise
    invisible once the run is over, which makes a load that did not do what it
    was told impossible to tell apart from one that did.
    """
    lines = [
        f"model            {cfg.llm.model}",
        f"base url         {cfg.llm.base_url}",
        f"lms binary       {llm.lms_binary() or 'NOT FOUND'}",
        f"transcript       {len(source)} characters",
        f"prompt           {len(cfg.llm.summary_prompt)} characters",
        f"context wanted   {llm.required_context(len(source) + len(cfg.llm.summary_prompt))}",
        f"max_context cap  {cfg.llm.max_context or 'none'}",
        f"gpu             {llm.vram_note().strip() or 'no NVIDIA GPU reported'}",
    ]
    for proc in hardware.gpu_processes():
        lines.append(f"  holding        {proc['used_mb']} MB  {proc['name']} (pid {proc['pid']})")
    try:
        yield lines.append
    finally:
        for entry in llm.loaded():
            lines.append(f"lms ps           {entry}")
        target = path / RAW / "model_load.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        store.write_atomic(target, "\n".join(lines) + "\n")


def _summarize(path: Path, meta: dict, segments: list[dict], cfg, force, progress) -> dict:
    report: dict[str, str] = {}
    store.update_meta(path, state="summarizing")
    # Always: the two models never need the card at the same time, and holding
    # one while the other loads is the whole reason summarizing runs out of
    # memory. Only meaningful for in-process CPU models; GPU recognition
    # released its memory by exiting, and nothing in this process can free it.
    asr.unload_all()
    if progress:
        progress(f"speech model released.{llm.vram_note()}")

    spoken = [s for s in segments if outputs.clean_text(s.get("text", ""))]
    if not spoken and not meta.get("notes"):
        raise llm.LlmError(
            "nothing to summarize: this meeting has no transcript segments. "
            "Check that the audio contains speech, then run process --force."
        )

    source = outputs.transcript_for_llm(meta, segments)
    meta.update(store.read_meta(path))

    if cfg.llm.auto_context and cfg.llm.model:
        # A transcript that overruns the context window is silently truncated,
        # leaving the model no room to answer. Size the window to the input.
        # Unloads first, then loads once: a load fails rather than stepping
        # down, because a smaller context that fits summarizes part of the
        # meeting while presenting itself as all of it.
        needed = len(source) + len(cfg.llm.summary_prompt)
        with _model_log(path, cfg, source) as note:
            ok, detail = llm.fit_context(cfg, needed, log=note)
            note(f"result: ok={ok} {detail}")
        if progress:
            progress(detail)
        if not ok:
            # Not fatal, since another server may size its own context, but it
            # has to survive the run: a truncated prompt otherwise shows up only
            # as a summary that quietly covers less than the whole meeting.
            store.update_meta(path, warning=detail)
            meta["warning"] = detail
    elif cfg.llm.model:
        # Without auto_context nothing above runs, and the request then reaches
        # a server that already has this model resident at some other context.
        # LM Studio answers that by loading a second instance rather than
        # reusing the first, which is two copies of the weights on one card.
        cleared, detail = llm.unload_everything(cfg)
        if progress:
            progress(f"freed the GPU: {detail}")

    def ticker(stage: str):
        if not progress:
            return None
        base = stage_fraction(stage) - dict(STAGES)[stage]
        span = dict(STAGES)[stage]

        def report_tokens(count: int):
            # No way to know the total ahead of time, so approach the end of
            # this stage asymptotically rather than pretending to a percentage.
            share = 1.0 - (1.0 / (1.0 + count / 250.0))
            progress(f"{stage} ({count} tokens)", base + span * share)

        return report_tokens

    if progress:
        progress("summarizing", stage_fraction("writing transcripts"))
    report["summary.md"] = artifacts.ensure(
        path, meta, "summary.md",
        # The reserve is an input: raising it after a truncated answer has to
        # invalidate the artifact, or re-running would report it as fresh.
        artifacts.sha(source, cfg.llm.summary_prompt, cfg.llm.model,
                      cfg.llm.answer_reserve),
        lambda: llm.chat(
            cfg, cfg.llm.summary_prompt, source, on_token=ticker("summarizing")
        ) + "\n",
        force,
    )

    if progress:
        progress("extracting actions", stage_fraction("summarizing"))
    actions_print = artifacts.sha(
        source, cfg.llm.actions_prompt, cfg.llm.model,
        prompts.ACTIONS_SCHEMA_VERSION, cfg.llm.answer_reserve,
    )
    report["actions.json"] = artifacts.ensure(
        path, meta, "actions.json", actions_print,
        lambda: json.dumps(
            llm.chat(
                cfg, cfg.llm.actions_prompt, source, prompts.ACTIONS_SCHEMA, "actions",
                on_token=ticker("extracting actions"),
            ),
            indent=2, ensure_ascii=False,
        ) + "\n",
        force,
    )

    actions = []
    actions_file = path / "actions.json"
    if actions_file.exists():
        try:
            actions = json.loads(actions_file.read_text()).get("actions", [])
        except json.JSONDecodeError:
            actions = []

    report["actions.md"] = artifacts.ensure(
        path, meta, "actions.md", artifacts.sha(actions),
        lambda: outputs.render_actions(meta, actions), force,
    )
    if progress:
        progress("writing calendar", stage_fraction("extracting actions"))
    _write_calendar(path, meta, actions)
    store.write_meta(path, meta)
    if progress:
        progress("done", 1.0)
    return report


def _write_calendar(path: Path, meta: dict, actions: list[dict]) -> None:
    # DTSTAMP must be stable across runs, or every regeneration rewrites
    # identical events. The meeting's own creation time is the natural anchor.
    try:
        stamp = datetime.fromisoformat(meta["created"]).astimezone(timezone.utc)
    except (KeyError, ValueError):
        stamp = datetime.now(timezone.utc)
    files = {}
    for index, action in enumerate(a for a in actions if a.get("due")):
        name = f"calendar/{_slug(action['task'], index + 1)}"
        uid = f"{uuid.uuid5(uuid.NAMESPACE_URL, path.name + name)}@meetnotes"
        files[name] = outputs.render_ics(meta, action, uid, stamp)
    (path / "calendar").mkdir(exist_ok=True)
    artifacts.replace_set(path, meta, "calendar_files", files)
