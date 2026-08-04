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

from meetnotes import config, prompts


def test_both_prompts_ask_for_specifics():
    for prompt in (prompts.SUMMARY, prompts.ACTIONS):
        assert "Be specific" in prompt
        assert "names of people" in prompt


def test_both_prompts_explain_recognition_errors():
    for prompt in (prompts.SUMMARY, prompts.ACTIONS):
        assert "speech recognition" in prompt
        assert "(uncertain)" in prompt


def test_prompts_permit_correcting_but_not_inventing():
    for prompt in (prompts.SUMMARY, prompts.ACTIONS):
        assert "Recovering a garbled word is expected" in prompt
        assert "inventing a fact nobody mentioned is not" in prompt


def test_old_blanket_instruction_is_gone():
    # "Invent nothing" made models drop anything they were unsure of, including
    # the garbled proper nouns that matter most.
    assert "Invent nothing" not in prompts.SUMMARY
    assert "Invent nothing" not in prompts.ACTIONS


def test_summary_handles_a_single_speaker():
    assert "monologue" in prompts.SUMMARY


def test_actions_asks_for_a_specific_task():
    assert "too vague" in prompts.ACTIONS


def test_migrate_replaces_a_superseded_default():
    old = prompts.SUPERSEDED["summary_prompt"][0]
    result = prompts.migrate({"summary_prompt": old})
    assert result["summary_prompt"] == prompts.SUMMARY


def test_migrate_keeps_a_customised_prompt():
    mine = "Summarise in haiku."
    assert prompts.migrate({"summary_prompt": mine})["summary_prompt"] == mine


def test_migrate_leaves_other_keys_alone():
    result = prompts.migrate({"model": "qwen3-8b", "temperature": 0.4})
    assert result == {"model": "qwen3-8b", "temperature": 0.4}


def test_loading_an_old_config_upgrades_the_prompts(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "llm": {
            "model": "qwen3-8b",
            "summary_prompt": prompts.SUPERSEDED["summary_prompt"][0],
            "actions_prompt": prompts.SUPERSEDED["actions_prompt"][0],
        }
    }))
    monkeypatch.setattr(config, "CONFIG_PATH", path)

    cfg = config.Config.load()
    assert cfg.llm.summary_prompt == prompts.SUMMARY
    assert cfg.llm.actions_prompt == prompts.ACTIONS
    assert cfg.llm.model == "qwen3-8b"


def test_loading_a_config_keeps_deliberate_edits(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"summary_prompt": "Only bullet points."}}))
    monkeypatch.setattr(config, "CONFIG_PATH", path)

    assert config.Config.load().llm.summary_prompt == "Only bullet points."
