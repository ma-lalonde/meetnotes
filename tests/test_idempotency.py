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

import json

import pytest

from meetnotes import pipeline, store
from meetnotes.config import Config


@pytest.fixture
def meeting(tmp_path):
    cfg = Config()
    cfg.data_dir = str(tmp_path)
    cfg.asr.final_pass = False
    path = store.new_meeting(tmp_path, "sync")
    store.update_meta(
        path,
        segments=[
            {"speaker": "Me", "start": 0.0, "end": 2.0, "text": "uh so we ship Tuesday"},
            {"speaker": "Them", "start": 2.0, "end": 4.0, "text": "we need the SLA"},
        ],
        notes=[{"at": 3.0, "text": "chase the SLA"}],
    )
    return cfg, path


def snapshot(path):
    """Generated artifacts only, with mtimes.

    meeting.json is deliberately excluded: it carries the live state marker
    that crash recovery reads, so it is rewritten as a run progresses. Its
    *content* settling back to the same value is asserted separately.
    """
    return {
        p.relative_to(path).as_posix(): (p.stat().st_mtime_ns, p.read_bytes())
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name not in (store.META, store.LOCK)
    }


RAW_TRANSCRIPT = "raw_output/transcription.md"


def test_first_run_writes_then_second_run_skips(meeting):
    cfg, path = meeting
    first = pipeline.process(path, cfg, with_llm=False)
    assert first[RAW_TRANSCRIPT] == "written"

    before = snapshot(path)
    before_meta = (path / store.META).read_bytes()
    second = pipeline.process(path, cfg, with_llm=False)
    assert set(second.values()) == {"skipped"}
    assert snapshot(path) == before
    assert (path / store.META).read_bytes() == before_meta


def test_no_temp_files_and_no_duplicates(meeting):
    cfg, path = meeting
    pipeline.process(path, cfg, with_llm=False)
    pipeline.process(path, cfg, with_llm=False)
    names = [p.name for p in path.rglob("*") if p.is_file()]
    assert not [n for n in names if n.endswith(".tmp")]
    assert len(names) == len(set(names))
    assert not [n for n in names if "-2." in n]


def test_hand_edited_file_is_never_clobbered(meeting):
    cfg, path = meeting
    pipeline.process(path, cfg, with_llm=False)
    edited = path / "raw_output/transcription.md"
    edited.write_text("my own words\n")

    report = pipeline.process(path, cfg, with_llm=False)
    assert report["raw_output/transcription.md"] == "hand-edited"
    assert edited.read_text() == "my own words\n"


def test_force_overwrites_hand_edited(meeting):
    cfg, path = meeting
    pipeline.process(path, cfg, with_llm=False)
    (path / "raw_output/transcription.md").write_text("my own words\n")

    report = pipeline.process(path, cfg, with_llm=False, force=True)
    assert report["raw_output/transcription.md"] == "written"
    assert "my own words" not in (path / "raw_output/transcription.md").read_text()


def test_changed_input_regenerates_only_dependents(meeting):
    cfg, path = meeting
    pipeline.process(path, cfg, with_llm=False)
    before = snapshot(path)

    store.update_meta(path, notes=[{"at": 5.0, "text": "new note"}])
    report = pipeline.process(path, cfg, with_llm=False)

    assert report["raw_output/notes.md"] == "written"
    assert report["raw_output/transcription.md"] == "written"
    assert (path / "raw_output/notes.md").read_bytes() != before["raw_output/notes.md"][1]


def test_filler_ruleset_bump_regenerates_cleaned_only(meeting, monkeypatch):
    cfg, path = meeting
    pipeline.process(path, cfg, with_llm=False)

    monkeypatch.setattr("meetnotes.outputs.filler_version", lambda: 99)
    report = pipeline.process(path, cfg, with_llm=False)
    assert report["raw_output/transcription_cleaned.md"] == "written"
    assert report["raw_output/transcription.md"] == "skipped"
    assert report["raw_output/notes.md"] == "skipped"


def test_old_layout_files_move_into_raw_output(meeting):
    cfg, path = meeting
    pipeline.process(path, cfg, with_llm=False)

    # Recreate the previous layout: files at the top level, recorded there.
    meta = store.read_meta(path)
    for old, new in pipeline.MOVED.items():
        (path / new).replace(path / old)
        meta["artifacts"][old] = meta["artifacts"].pop(new)
    store.write_meta(path, meta)

    report = pipeline.process(path, cfg, with_llm=False)

    assert set(report.values()) == {"skipped"}
    for old, new in pipeline.MOVED.items():
        assert not (path / old).exists()
        assert (path / new).exists()
    assert not set(pipeline.MOVED) & set(store.read_meta(path)["artifacts"])


def test_migration_preserves_hand_edits(meeting):
    cfg, path = meeting
    pipeline.process(path, cfg, with_llm=False)

    meta = store.read_meta(path)
    (path / RAW_TRANSCRIPT).replace(path / "transcription.md")
    meta["artifacts"]["transcription.md"] = meta["artifacts"].pop(RAW_TRANSCRIPT)
    store.write_meta(path, meta)
    (path / "transcription.md").write_text("my own words\n")

    report = pipeline.process(path, cfg, with_llm=False)

    assert (path / RAW_TRANSCRIPT).read_text() == "my own words\n"
    assert report[RAW_TRANSCRIPT] == "hand-edited"
    assert not (path / "transcription.md").exists()


def test_second_caller_gets_busy_not_a_duplicate_run(meeting):
    cfg, path = meeting
    with store.exclusive(path):
        with pytest.raises(store.Busy):
            pipeline.process(path, cfg, with_llm=False)


def test_recover_resets_interrupted_meeting(meeting):
    cfg, path = meeting
    store.update_meta(path, state="summarizing")
    assert store.recover(cfg.root) == [path.name]
    assert store.read_meta(path)["state"] == "pending"


def test_calendar_orphans_are_removed(meeting):
    cfg, path = meeting
    meta = store.read_meta(path)
    actions = [
        {"owner": "Marc", "task": "send SLA", "due": "2026-08-10", "kind": "todo",
         "quote": "", "at": 1.0},
        {"owner": "Ann", "task": "book room", "due": "2026-08-11", "kind": "event",
         "quote": "", "at": 2.0},
    ]
    pipeline._write_calendar(path, meta, actions)
    store.write_meta(path, meta)
    assert len(list((path / "calendar").glob("*.ics"))) == 2

    meta = store.read_meta(path)
    pipeline._write_calendar(path, meta, actions[:1])
    store.write_meta(path, meta)
    assert len(list((path / "calendar").glob("*.ics"))) == 1


def test_calendar_rewrite_is_byte_stable(meeting):
    cfg, path = meeting
    actions = [{"owner": "Marc", "task": "send SLA", "due": "2026-08-10", "kind": "todo",
                "quote": "", "at": 1.0}]
    meta = store.read_meta(path)
    pipeline._write_calendar(path, meta, actions)
    store.write_meta(path, meta)
    first = snapshot(path / "calendar")

    meta = store.read_meta(path)
    pipeline._write_calendar(path, meta, actions)
    store.write_meta(path, meta)
    assert snapshot(path / "calendar") == first


def test_actions_json_drives_actions_md(meeting):
    cfg, path = meeting
    (path / "actions.json").write_text(
        json.dumps({"actions": [{"owner": "Marc", "task": "send SLA", "due": None,
                                 "kind": "todo", "quote": "", "at": 1.0}]})
    )
    meta = store.read_meta(path)
    text = pipeline.outputs.render_actions(meta, json.loads((path / "actions.json").read_text())["actions"])
    assert "**Marc**: send SLA" in text
