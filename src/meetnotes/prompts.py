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

# Shared preamble. Both tasks read the same imperfect transcript and both were
# losing the same thing: the specifics.
_TRANSCRIPT_CAVEAT = (
    "The transcript comes from speech recognition and contains errors. Proper "
    "nouns suffer worst: names of people, bands, films, books, companies, "
    "products and places are often misspelt, split into unrelated words, or "
    "replaced by something that merely sounds similar. Work out what was meant "
    "from context and write the corrected form. When you cannot resolve it, "
    "give your best reading followed by (uncertain) rather than dropping it. "
    "Recovering a garbled word is expected of you; inventing a fact nobody "
    "mentioned is not.\n\n"
    "Be specific. Name the people, works, organisations, places, numbers, dates "
    "and amounts that were actually mentioned, and keep figures and dates "
    "exactly as spoken. Writing 'a concert' where the transcript names the band, "
    "or 'a deadline' where it gives the date, is a failure. Prefer the concrete "
    "detail over the general description every time.\n\n"
    "Write in the language of the transcript."
)

SUMMARY = (
    "You write meeting notes from an automatic transcript.\n\n"
    f"{_TRANSCRIPT_CAVEAT}\n\n"
    "Output markdown with exactly these sections: Context, Key points, "
    "Decisions, Open questions. Attribute claims to the speaker who made them. "
    "If a section has nothing, write None.\n\n"
    "A recording with one speaker is a monologue, not a negotiation: summarise "
    "what was actually discussed rather than forcing it into decisions and "
    "open questions."
)

ACTIONS = (
    "You extract commitments from an automatic transcript.\n\n"
    f"{_TRANSCRIPT_CAVEAT}\n\n"
    "Return JSON matching the provided schema.\n"
    "owner is the speaker responsible, or the person named as responsible.\n"
    "task names the specific thing to be done, including the people, works, "
    "places or amounts involved. 'Buy tickets' is too vague if the transcript "
    "says which show.\n"
    "due is an ISO date (YYYY-MM-DD) only when a date is stated or unambiguously "
    "derivable from the meeting date; otherwise null.\n"
    "kind is 'event' for something scheduled at a time, 'todo' otherwise.\n"
    "quote is the shortest verbatim span that supports the item, copied from "
    "the transcript exactly as it appears, errors included.\n"
    "at is the timecode in seconds where that quote occurs.\n\n"
    "An empty list is a valid answer."
)

CLEANUP = (
    "Remove disfluencies from meeting transcript lines. Write in the input's language. "
    "Keep every line, its timecode prefix, and its speaker heading exactly as given. "
    "Remove filler words, false starts, and stutters only. "
    "Do not paraphrase, summarize, reorder, merge, or add anything. "
    "If a line has no disfluency, return it byte-identical."
)

# Prompts are stored in config.json, so an improved default would never reach
# anyone who has already run the app. A saved prompt that exactly matches a
# previous default was never customised, so it is safe to replace.
SUPERSEDED = {
    "summary_prompt": [
        "Summarize meeting transcripts. Write in the transcript's own language. "
        "Output markdown with exactly these sections: Context, Key points, Decisions, "
        "Open questions. Attribute claims to the speaker who made them. "
        "If a section has nothing, write 'None'. Invent nothing."
    ],
    "actions_prompt": [
        "Extract commitments from meeting transcripts. Write in the transcript's own language. "
        "Return JSON matching the provided schema. "
        "owner is the speaker responsible, or the person named as responsible. "
        "due is an ISO date (YYYY-MM-DD) only when a date is actually stated or "
        "unambiguously derivable from the meeting date; otherwise null. "
        "kind is 'event' for something scheduled at a time, 'todo' otherwise. "
        "quote is the shortest verbatim span that supports the item. "
        "at is the timecode in seconds where that quote occurs. "
        "Invent nothing. An empty list is a valid answer."
    ],
}

CURRENT = {"summary_prompt": SUMMARY, "actions_prompt": ACTIONS}


def migrate(saved: dict) -> dict:
    """Replace prompts that are verbatim copies of an old default."""
    updated = dict(saved)
    for key, old_versions in SUPERSEDED.items():
        if updated.get(key, "").strip() in [old.strip() for old in old_versions]:
            updated[key] = CURRENT[key]
    return updated


ACTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "task": {"type": "string"},
                    "due": {"type": ["string", "null"]},
                    "kind": {"type": "string", "enum": ["todo", "event"]},
                    "quote": {"type": "string"},
                    "at": {"type": "number"},
                },
                "required": ["owner", "task", "due", "kind", "quote", "at"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["actions"],
    "additionalProperties": False,
}

# Bumped when the filler ruleset changes, so cleaned transcripts regenerate.
FILLER_RULESET_VERSION = 1
# Bumped when ACTIONS_SCHEMA changes, so actions.json regenerates.
ACTIONS_SCHEMA_VERSION = 1
