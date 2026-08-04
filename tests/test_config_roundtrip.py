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

from meetnotes import config, prompts
from meetnotes.config import Config


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    return path


def test_vocabulary_survives_a_restart(config_file):
    cfg = Config()
    cfg.asr.vocabulary = ["Godspeed You! Black Emperor", "Chloé Gagnon", "Catena"]
    cfg.save()

    assert Config.load().asr.vocabulary == [
        "Godspeed You! Black Emperor", "Chloé Gagnon", "Catena"
    ]


def test_an_edited_prompt_survives_a_restart(config_file):
    cfg = Config()
    cfg.llm.summary_prompt = "Only bullet points, no prose."
    cfg.llm.actions_prompt = "Only overdue items."
    cfg.save()

    reloaded = Config.load()
    assert reloaded.llm.summary_prompt == "Only bullet points, no prose."
    assert reloaded.llm.actions_prompt == "Only overdue items."


def test_saving_one_screen_does_not_lose_another_screens_fields(config_file):
    cfg = Config()
    cfg.asr.vocabulary = ["Catena"]
    cfg.llm.summary_prompt = "Custom."
    cfg.capture.mic_label = "Marc"
    cfg.save()

    # A later save from a different screen writes the whole file again.
    cfg = Config.load()
    cfg.llm.model = "qwen3-8b"
    cfg.save()

    reloaded = Config.load()
    assert reloaded.asr.vocabulary == ["Catena"]
    assert reloaded.llm.summary_prompt == "Custom."
    assert reloaded.capture.mic_label == "Marc"
    assert reloaded.llm.model == "qwen3-8b"


def test_accents_survive_the_round_trip(config_file):
    cfg = Config()
    cfg.asr.vocabulary = ["Chloé", "Ægir", "Straße"]
    cfg.save()
    assert Config.load().asr.vocabulary == ["Chloé", "Ægir", "Straße"]
    # Written as real characters, not escapes.
    assert "Chloé" in config_file.read_text(encoding="utf-8")


def test_an_empty_vocabulary_round_trips(config_file):
    cfg = Config()
    cfg.asr.vocabulary = []
    cfg.save()
    assert Config.load().asr.vocabulary == []


def test_a_default_prompt_is_upgraded_but_an_edited_one_is_not(config_file):
    cfg = Config()
    cfg.save()
    # An untouched prompt is stored as the current default and stays current.
    assert Config.load().llm.summary_prompt == prompts.SUMMARY

    saved = json.loads(config_file.read_text())
    saved["llm"]["summary_prompt"] = prompts.SUPERSEDED["summary_prompt"][0]
    config_file.write_text(json.dumps(saved))
    assert Config.load().llm.summary_prompt == prompts.SUMMARY

    saved["llm"]["summary_prompt"] = "Mine."
    config_file.write_text(json.dumps(saved))
    assert Config.load().llm.summary_prompt == "Mine."


def test_a_corrupt_config_falls_back_to_defaults(config_file):
    config_file.write_text("{ not json")
    assert Config.load().llm.summary_prompt == prompts.SUMMARY


def test_unknown_keys_from_a_newer_version_are_ignored(config_file):
    config_file.write_text(json.dumps({
        "asr": {"vocabulary": ["Catena"], "some_future_field": 1},
        "unknown_section": {"a": 1},
    }))
    cfg = Config.load()
    assert cfg.asr.vocabulary == ["Catena"]
