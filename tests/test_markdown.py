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

import re

from meetnotes import outputs

META = {
    "title": "Sync",
    "created": "2026-08-04T14:30:00",
    "notes": [{"at": 3.0, "text": "chase the SLA"}],
    "run": {"live_model": "large-v3-turbo", "final_model": "large-v3-turbo",
            "device": "cuda", "compute_type": "float16", "gpu": "RTX 4060"},
    "segments": [],
}

SEGMENTS = [
    {"speaker": "Marc", "start": 0.0, "end": 2.0, "text": "we ship Tuesday"},
    {"speaker": "Marc", "start": 2.0, "end": 4.0, "text": "the SLA is pending"},
    {"speaker": "Chloe", "start": 5.0, "end": 7.0, "text": "I will confirm"},
]

TIMECODE = re.compile(r"^\[\d\d:\d\d:\d\d\]")


def adjacent_timecodes(text: str) -> list[tuple[str, str]]:
    """Timecoded lines that a Markdown renderer would fold into one paragraph."""
    lines = text.splitlines()
    return [
        (lines[i], lines[i + 1])
        for i in range(len(lines) - 1)
        if TIMECODE.match(lines[i]) and lines[i + 1].strip()
    ]


def test_transcript_separates_every_timecode():
    assert adjacent_timecodes(outputs.render_transcript(META, SEGMENTS, clean=True)) == []


def test_verbatim_transcript_separates_every_timecode():
    assert adjacent_timecodes(outputs.render_transcript(META, SEGMENTS, clean=False)) == []


def test_interleaved_transcript_separates_every_timecode():
    assert adjacent_timecodes(outputs.render_transcript_with_notes(META, SEGMENTS)) == []


def test_consecutive_lines_from_one_speaker_are_separate_blocks():
    text = outputs.render_transcript(META, SEGMENTS, clean=True)
    assert "[00:00:00] we ship Tuesday\n\n[00:00:02] the SLA is pending" in text


def test_provenance_stays_one_blockquote():
    text = outputs.render_transcript(META, SEGMENTS, clean=True)
    assert "> Transcribed with" in text
    quote = [line for line in text.splitlines() if line.startswith("> ")]
    assert len(quote) == 2
    # Adjacent quote lines are one block in Markdown, which is what we want.
    assert "\n".join(quote) in text


def test_note_is_its_own_block():
    text = outputs.render_transcript_with_notes(META, SEGMENTS)
    assert "\n\n> **NOTE** [00:00:03] chase the SLA\n\n" in text


def test_headings_are_separated_from_their_content():
    text = outputs.render_transcript(META, SEGMENTS, clean=True)
    assert "**Marc**\n\n[00:00:00]" in text
    assert "**Chloe**\n\n[00:00:05]" in text


def test_notes_file_keeps_list_items_together():
    text = outputs.render_notes(META)
    # List items are already separate blocks, so they stay on adjacent lines.
    assert "- [00:00:03] chase the SLA" in text


def test_actions_file_separates_headings_from_lists():
    actions = [
        {"owner": "Marc", "task": "send the SLA", "due": "2026-08-10",
         "kind": "todo", "quote": "", "at": 1.0},
        {"owner": "Chloe", "task": "book the room", "due": None,
         "kind": "todo", "quote": "", "at": 2.0},
    ]
    text = outputs.render_actions(META, actions)
    assert "## Action items\n\n- **Marc**" in text
    assert "- **Chloe**: book the room" in text
    assert "## Scheduled\n\nNone" in text


def test_every_document_ends_with_exactly_one_newline():
    for text in (
        outputs.render_transcript(META, SEGMENTS, clean=True),
        outputs.render_transcript_with_notes(META, SEGMENTS),
        outputs.render_notes(META),
        outputs.render_actions(META, []),
    ):
        assert text.endswith("\n")
        assert not text.endswith("\n\n")


def test_no_document_starts_with_blank_lines():
    for text in (
        outputs.render_transcript(META, SEGMENTS, clean=True),
        outputs.render_transcript_with_notes(META, SEGMENTS),
    ):
        assert text.startswith("# Sync")
