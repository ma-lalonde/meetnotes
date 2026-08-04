from meetnotes import outputs

META = {"title": "Sync", "created": "2026-08-04T14:30:00", "notes": []}


def worded(start, end, pairs, speaker="Me"):
    return {
        "speaker": speaker,
        "start": start,
        "end": end,
        "text": "".join(word for _, _, word in pairs).strip(),
        "words": [[a, b, word] for a, b, word in pairs],
    }


SENTENCE = worded(
    10.0, 16.0,
    [(10.0, 11.0, "we"), (11.0, 12.0, " should"), (12.0, 13.0, " ship"),
     (13.0, 14.0, " the"), (14.0, 15.0, " SLA"), (15.0, 16.0, " Tuesday")],
)


def test_split_cuts_after_the_last_completed_word():
    # "the" spans 13.0-14.0, so at 13.5 it was still being spoken and belongs
    # to the part after the note.
    before, after = outputs.split_segment(SENTENCE, 13.5)
    assert before["text"] == "we should ship"
    assert after["text"] == "the SLA Tuesday"
    assert before["end"] == 13.0
    assert after["start"] == 13.0


def test_split_never_cuts_inside_a_word():
    for at in (10.5, 11.4, 12.9, 15.2):
        parts = outputs.split_segment(SENTENCE, at)
        if parts is None:
            continue
        before, after = parts
        assert not before["text"].endswith(" ")
        joined = before["text"] + " " + after["text"]
        assert joined == SENTENCE["text"]


def test_split_marks_the_tail_as_continued():
    _, after = outputs.split_segment(SENTENCE, 13.5)
    assert after["continued"] is True


def test_no_split_outside_the_segment():
    assert outputs.split_segment(SENTENCE, 9.0) is None
    assert outputs.split_segment(SENTENCE, 20.0) is None


def test_no_split_before_the_first_word_ends():
    # Everything would land on one side, so the segment is left whole.
    assert outputs.split_segment(SENTENCE, 10.2) is None


def test_split_without_word_timestamps_snaps_to_a_space():
    plain = {"speaker": "Me", "start": 0.0, "end": 10.0, "text": "we should ship the SLA Tuesday"}
    before, after = outputs.split_segment(plain, 5.0)
    assert " " not in before["text"][-1:]
    assert (before["text"] + " " + after["text"]) == plain["text"]


def test_note_inside_a_sentence_splits_it():
    notes = [{"at": 13.5, "text": "confirm the date"}]
    timeline = outputs.merge_notes([SENTENCE], notes)
    kinds = [item["kind"] for item in timeline]
    assert kinds == ["speech", "note", "speech"]
    assert timeline[0]["text"] == "we should ship"
    assert timeline[2]["text"] == "the SLA Tuesday"


def test_note_between_sentences_does_not_split_anything():
    second = worded(20.0, 22.0, [(20.0, 21.0, "agreed"), (21.0, 22.0, " then")], speaker="Them")
    timeline = outputs.merge_notes([SENTENCE, second], [{"at": 18.0, "text": "gap"}])
    assert [item["kind"] for item in timeline] == ["speech", "note", "speech"]
    assert timeline[0]["text"] == SENTENCE["text"]


def test_two_notes_inside_one_sentence_split_twice():
    notes = [{"at": 12.5, "text": "one"}, {"at": 14.5, "text": "two"}]
    timeline = outputs.merge_notes([SENTENCE], notes)
    assert [item["kind"] for item in timeline] == ["speech", "note", "speech", "note", "speech"]
    spoken = " ".join(i["text"] for i in timeline if i["kind"] == "speech")
    assert spoken == SENTENCE["text"]


def test_no_speech_is_lost_when_splitting():
    notes = [{"at": 11.5, "text": "a"}, {"at": 13.5, "text": "b"}, {"at": 15.5, "text": "c"}]
    timeline = outputs.merge_notes([SENTENCE], notes)
    spoken = " ".join(i["text"] for i in timeline if i["kind"] == "speech")
    assert spoken == SENTENCE["text"]


def test_notes_before_and_after_all_speech_are_kept():
    timeline = outputs.merge_notes(
        [SENTENCE], [{"at": 0.0, "text": "early"}, {"at": 99.0, "text": "late"}]
    )
    assert timeline[0]["text"] == "early"
    assert timeline[-1]["text"] == "late"


def test_rendered_output_marks_the_continuation():
    meta = {**META, "notes": [{"at": 13.5, "text": "confirm the date"}]}
    text = outputs.render_transcript_with_notes(meta, [SENTENCE])
    assert "> **NOTE** [00:00:13] confirm the date" in text
    assert "...continued the SLA Tuesday" in text
    assert text.index("we should ship") < text.index("confirm the date")
    assert text.index("confirm the date") < text.index("the SLA Tuesday")


def test_llm_source_sees_the_split_too():
    meta = {**META, "notes": [{"at": 13.5, "text": "confirm the date"}]}
    source = outputs.transcript_for_llm(meta, [SENTENCE])
    assert source.index("we should ship") < source.index("confirm the date")
    assert source.index("confirm the date") < source.index("the SLA Tuesday")
