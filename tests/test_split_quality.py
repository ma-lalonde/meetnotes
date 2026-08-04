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


def spoken(words, speaker="Marc"):
    """Build a segment from (word, duration) pairs, one second apart."""
    entries = []
    clock = 10.0
    for word in words:
        entries.append([clock, clock + 1.0, word])
        clock += 1.0
    return {
        "speaker": speaker,
        "start": entries[0][0],
        "end": entries[-1][1],
        "text": "".join(w for _, _, w in entries).strip(),
        "words": entries,
    }


CLAUSED = spoken(
    ["we", " should", " ship", " Tuesday,", " but", " the", " SLA", " is", " not", " signed"]
)
PLAIN = spoken(["one", " two", " three", " four", " five", " six", " seven", " eight"])


def test_cut_prefers_a_comma_over_the_exact_moment():
    # The note lands during "but", but the comma one word earlier reads better.
    before, after = outputs.split_segment(CLAUSED, 15.5)
    assert before["text"] == "we should ship Tuesday,"
    assert after["text"] == "but the SLA is not signed"


def test_comma_is_ignored_when_it_is_far_away():
    # The comma ends at 14.0, well outside the window from a note at 17.5.
    parts = outputs.split_segment(CLAUSED, 17.5, min_share=0.1)
    assert parts is not None
    before, _ = parts
    assert before["text"] != "we should ship Tuesday,"


def test_a_clause_ending_after_the_note_is_not_used():
    # The comma ends at 14.0, two and a half seconds after a note at 11.5.
    # Cutting there would put the note ahead of words not yet spoken.
    parts = outputs.split_segment(CLAUSED, 11.5, min_share=0.05)
    assert parts is not None
    before, _ = parts
    assert before["text"] == "we"


def test_a_clause_ending_just_after_the_note_is_still_used():
    # Within the short lookahead: the note was almost certainly about the
    # clause that was finishing as it was typed.
    before, after = outputs.split_segment(CLAUSED, 13.5)
    assert before["text"] == "we should ship Tuesday,"


def test_no_cut_when_a_fragment_would_be_tiny():
    # A note one word in would strand "one" on its own.
    assert outputs.split_segment(PLAIN, 11.5) is None


def test_no_cut_near_the_end_either():
    assert outputs.split_segment(PLAIN, 17.5) is None


def test_cut_happens_near_the_middle():
    parts = outputs.split_segment(PLAIN, 14.5)
    assert parts is not None
    before, after = parts
    assert before["text"] == "one two three four"
    assert after["text"] == "five six seven eight"


def test_threshold_is_adjustable():
    assert outputs.split_segment(PLAIN, 11.5, min_share=0.05) is not None


def test_early_note_goes_before_the_whole_sentence():
    timeline = outputs.merge_notes([PLAIN], [{"at": 11.5, "text": "early thought"}])
    assert [item["kind"] for item in timeline] == ["note", "speech"]
    assert timeline[1]["text"] == PLAIN["text"]


def test_late_note_goes_after_the_whole_sentence():
    timeline = outputs.merge_notes([PLAIN], [{"at": 17.5, "text": "late thought"}])
    assert [item["kind"] for item in timeline] == ["speech", "note"]
    assert timeline[0]["text"] == PLAIN["text"]


def test_unsplit_sentence_is_never_marked_continued():
    timeline = outputs.merge_notes([PLAIN], [{"at": 11.5, "text": "early"}])
    assert not any(item.get("continued") for item in timeline)


def test_no_speech_is_lost_whichever_branch_is_taken():
    for at in (11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5):
        timeline = outputs.merge_notes([CLAUSED], [{"at": at, "text": "note"}])
        spoken_text = " ".join(i["text"] for i in timeline if i["kind"] == "speech")
        assert spoken_text == CLAUSED["text"], at


def test_plain_text_segments_also_avoid_tiny_fragments():
    segment = {"speaker": "Marc", "start": 0.0, "end": 10.0,
               "text": "one two three four five six seven eight"}
    assert outputs.split_segment(segment, 0.6) is None
    assert outputs.split_segment(segment, 5.0) is not None


def test_plain_text_segments_prefer_punctuation():
    segment = {"speaker": "Marc", "start": 0.0, "end": 10.0,
               "text": "we should ship Tuesday, but the SLA is not signed"}
    before, after = outputs.split_segment(segment, 5.2)
    assert before["text"] == "we should ship Tuesday,"
    assert after["text"] == "but the SLA is not signed"
