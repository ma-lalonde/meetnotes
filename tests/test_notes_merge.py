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
    assert [item["start"] for item in merged] == [0.0, 5.0, 6.0, 60.0, 61.0]


def test_a_note_follows_the_speech_it_shares_a_timestamp_with():
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
    assert text.count("**Me**") == 2


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
