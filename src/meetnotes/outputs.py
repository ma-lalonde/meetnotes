import re
from datetime import datetime, timedelta

from . import prompts

FILLERS = [
    r"\buh+m*\b", r"\bum+\b", r"\bah+\b", r"\ber+m*\b", r"\bhm+\b", r"\bmm+\b",
    r"\beuh+\b", r"\bben\b", r"\bbah\b", r"\bhein\b",
    r"\byou know\b", r"\bi mean\b", r"\bsort of\b", r"\bkind of\b",
    r"\btu sais\b", r"\bgenre\b", r"\bfait que\b",
]
FILLER_RE = re.compile("|".join(FILLERS), re.IGNORECASE)
REPEAT_RE = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def clean_text(text: str) -> str:
    text = FILLER_RE.sub("", text)
    text = REPEAT_RE.sub(r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Removing a parenthetical filler ("we should, you know, ship") strands
    # the punctuation that framed it.
    text = re.sub(r"([,;:])(\s*[,;:])+", r"\1", text)
    text = re.sub(r"[,;:]\s*([.!?])", r"\1", text)
    text = re.sub(r"^[\s,;:]+", "", text)
    return text.strip()


def render_provenance(meta: dict) -> list[str]:
    run = meta.get("run") or {}
    if not run:
        return []
    languages = sorted({s.get("language") for s in meta.get("segments", []) if s.get("language")})
    lines = [
        "> Transcribed with `{live}` live and `{final}` final on {device} ({compute}).".format(
            live=run.get("live_model", "?"),
            final=run.get("final_model", "?"),
            device=run.get("device", "?"),
            compute=run.get("compute_type", "?"),
        )
    ]
    if run.get("gpu") and run["gpu"] != "none":
        lines[0] = lines[0][:-1] + f" on {run['gpu']}."
    detected = ", ".join(languages) if languages else run.get("language", "auto")
    lines.append(f"> Language: {detected}.")
    return lines + [""]


def render_transcript(meta: dict, segments: list[dict], clean: bool) -> str:
    lines = [f"# {meta['title']}", "", f"Recorded {meta['created']}", ""]
    lines += render_provenance(meta)
    last_speaker = None
    for seg in segments:
        text = clean_text(seg["text"]) if clean else seg["text"]
        if not text:
            continue
        if seg["speaker"] != last_speaker:
            lines.append("")
            lines.append(f"**{seg['speaker']}**")
            lines.append("")
            last_speaker = seg["speaker"]
        lines.append(f"[{clock(seg['start'])}] {text}")
    return "\n".join(lines).strip() + "\n"


def render_notes(meta: dict) -> str:
    lines = [f"# Notes - {meta['title']}", ""]
    notes = meta.get("notes", [])
    for note in notes:
        lines.append(f"- [{clock(note['at'])}] {note['text']}")
    if not notes:
        lines.append("None")
    return "\n".join(lines) + "\n"


def merge_notes(segments: list[dict], notes: list[dict]) -> list[dict]:
    """Interleave notes with speech on one timeline.

    A note sorts to the moment it was typed, so it lands next to whatever was
    being said then. Ties put the note after the speech that prompted it.
    """
    timeline = [{**s, "kind": "speech"} for s in segments]
    timeline += [{"kind": "note", "start": n["at"], "text": n["text"]} for n in notes]
    timeline.sort(key=lambda item: (item["start"], item["kind"] == "note"))
    return timeline


def render_transcript_with_notes(meta: dict, segments: list[dict]) -> str:
    lines = [f"# {meta['title']}", "", f"Recorded {meta['created']}", ""]
    lines += render_provenance(meta)
    lines.append("Notes taken during the meeting appear inline, marked NOTE.")
    lines.append("")

    last_speaker = None
    for item in merge_notes(segments, meta.get("notes", [])):
        if item["kind"] == "note":
            lines.append("")
            lines.append(f"> **NOTE** [{clock(item['start'])}] {item['text']}")
            lines.append("")
            last_speaker = None
            continue
        text = clean_text(item["text"])
        if not text:
            continue
        if item["speaker"] != last_speaker:
            lines.append("")
            lines.append(f"**{item['speaker']}**")
            lines.append("")
            last_speaker = item["speaker"]
        lines.append(f"[{clock(item['start'])}] {text}")
    return "\n".join(lines).strip() + "\n"


def transcript_for_llm(meta: dict, segments: list[dict]) -> str:
    """What the language model reads.

    Notes are interleaved rather than appended: a note only means something
    next to the speech that prompted it, and models align timecodes poorly
    when asked to do it themselves.
    """
    body = []
    for item in merge_notes(segments, meta.get("notes", [])):
        if item["kind"] == "note":
            body.append(f"[{clock(item['start'])}] NOTE FROM THE NOTE-TAKER: {item['text']}")
            continue
        text = clean_text(item["text"])
        if text:
            body.append(f"[{clock(item['start'])}] {item['speaker']}: {text}")

    header = f"Meeting: {meta['title']}\nDate: {meta['created']}\n\n"
    if meta.get("notes"):
        header += (
            "Lines marked NOTE FROM THE NOTE-TAKER were typed by hand during the "
            "meeting and were never spoken aloud. Treat them as what the note-taker "
            "considered important, not as dialogue, and never attribute them to a "
            "speaker.\n\n"
        )
    return header + "\n".join(body)


def render_actions(meta: dict, actions: list[dict]) -> str:
    todos = [a for a in actions if a.get("kind") != "event"]
    events = [a for a in actions if a.get("kind") == "event"]
    dated = [a for a in actions if a.get("due")]

    lines = [f"# Actions - {meta['title']}", "", "## Action items", ""]
    lines += _bullets(todos) or ["None"]
    lines += ["", "## Scheduled", ""]
    lines += _bullets(events) or ["None"]
    lines += ["", "## Dates mentioned", ""]
    lines += [f"- {a['due']} - {a['task']}" for a in dated] or ["None"]
    return "\n".join(lines) + "\n"


def _bullets(actions: list[dict]) -> list[str]:
    out = []
    for action in actions:
        due = f" (due {action['due']})" if action.get("due") else ""
        stamp = f" [{clock(action['at'])}]" if action.get("at") is not None else ""
        out.append(f"- **{action.get('owner') or 'Unassigned'}**: {action['task']}{due}{stamp}")
    return out


def _ics_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def _fold(line: str) -> str:
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return line
    chunks, start = [], 0
    while start < len(raw):
        end = min(start + (75 if not chunks else 74), len(raw))
        while end < len(raw) and (raw[end] & 0xC0) == 0x80:
            end -= 1
        chunks.append(raw[start:end].decode("utf-8"))
        start = end
    return "\r\n ".join(chunks)


def render_ics(meta: dict, action: dict, uid: str, stamp: datetime) -> str:
    due = datetime.strptime(action["due"], "%Y-%m-%d").date()
    summary = _ics_escape(f"{action['task']}")
    owner = action.get("owner") or "Unassigned"
    description = _ics_escape(
        f"Owner: {owner}\nFrom meeting: {meta['title']} ({meta['created']})\n"
        f'Quote: "{action.get("quote", "")}"'
    )
    dtstamp = stamp.strftime("%Y%m%dT%H%M%SZ")

    body = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//meetnotes//EN",
        "CALSCALE:GREGORIAN",
    ]
    if action.get("kind") == "event":
        body += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{dtstamp}",
            f"DTSTART;VALUE=DATE:{due:%Y%m%d}",
            f"DTEND;VALUE=DATE:{due + timedelta(days=1):%Y%m%d}",
            f"SUMMARY:{summary}",
            f"DESCRIPTION:{description}",
            "END:VEVENT",
        ]
    else:
        body += [
            "BEGIN:VTODO",
            f"UID:{uid}",
            f"DTSTAMP:{dtstamp}",
            f"DUE;VALUE=DATE:{due:%Y%m%d}",
            f"SUMMARY:{summary}",
            f"DESCRIPTION:{description}",
            "STATUS:NEEDS-ACTION",
            "END:VTODO",
        ]
    body.append("END:VCALENDAR")
    return "\r\n".join(_fold(line) for line in body) + "\r\n"


def filler_version() -> int:
    return prompts.FILLER_RULESET_VERSION
