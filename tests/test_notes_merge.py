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

from meetnotes import outputs

META = {
    "title": "Client sync",
    "created": "2026-08-04T14:30:00",
    "notes": [
        {"at": 5.0, "text": "chase the SLA"},
        {"at": 61.0, "text": "budget decision pending"},
    ],
}

SEGMENTS = [
    {"speaker": "Me", "start": 0.0, "end": 4.0, "text": "uh so we ship Tuesday"},
    {"speaker": "Them", "start": 6.0, "end": 9.0, "text": "we need the SLA first"},
    {"speaker": "Me", "start": 60.0, "end": 64.0, "text": "the budget is not signed"},
]


def test_notes_land_in_time_order():
    merged = outputs.merge_notes(SEGMENTS, META["notes"])
    # The note at 61.0 falls inside the 60.0-64.0 segment, so that segment is
    # cut in two around it. Without word timestamps the cut is proportional.
    assert [item["kind"] for item in merged] == [
        "speech", "note", "speech", "speech", "note", "speech"
    ]
    assert [item["start"] for item in merged][:3] == [0.0, 5.0, 6.0]


def test_a_note_at_a_segment_start_still_follows_it():
    segments = [{"speaker": "Me", "start": 10.0, "end": 12.0, "text": "hello"}]
    merged = outputs.merge_notes(segments, [{"at": 10.0, "text": "important"}])
    assert [item["kind"] for item in merged] == ["speech", "note"]


def test_notes_only_still_produces_a_timeline():
    merged = outputs.merge_notes([], [{"at": 3.0, "text": "solo"}])
    assert len(merged) == 1 and merged[0]["kind"] == "note"


def test_rendered_file_marks_notes_distinctly():
    text = outputs.render_transcript_with_notes(META, SEGMENTS)
    assert "> **NOTE** [00:00:05] chase the SLA" in text
    assert "> **NOTE** [00:01:01] budget decision pending" in text


def test_rendered_file_places_a_note_between_the_right_lines():
    text = outputs.render_transcript_with_notes(META, SEGMENTS)
    first_speech = text.index("we ship Tuesday")
    note = text.index("chase the SLA")
    second_speech = text.index("we need the SLA first")
    assert first_speech < note < second_speech


def test_rendered_file_is_cleaned():
    text = outputs.render_transcript_with_notes(META, SEGMENTS)
    assert "uh so" not in text
    assert "we ship Tuesday" in text


def test_speaker_heading_reappears_after_a_note():
    text = outputs.render_transcript_with_notes(META, SEGMENTS)
    # Once at the start, once after each note that interrupts Me speaking.
    assert text.count("**Me**") == 3


def test_llm_source_interleaves_and_labels_notes():
    source = outputs.transcript_for_llm(META, SEGMENTS)
    assert "NOTE FROM THE NOTE-TAKER: chase the SLA" in source
    assert source.index("we ship Tuesday") < source.index("chase the SLA")


def test_llm_source_warns_against_attributing_notes_to_speakers():
    source = outputs.transcript_for_llm(META, SEGMENTS)
    assert "never attribute them to a speaker" in source


def test_llm_source_without_notes_is_plain_speech():
    source = outputs.transcript_for_llm({**META, "notes": []}, SEGMENTS)
    assert "NOTE FROM THE NOTE-TAKER" not in source
    assert "we ship Tuesday" in source
