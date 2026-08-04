SUMMARY = (
    "Summarize meeting transcripts. Write in the transcript's own language. "
    "Output markdown with exactly these sections: Context, Key points, Decisions, "
    "Open questions. Attribute claims to the speaker who made them. "
    "If a section has nothing, write 'None'. Invent nothing."
)

ACTIONS = (
    "Extract commitments from meeting transcripts. Write in the transcript's own language. "
    "Return JSON matching the provided schema. "
    "owner is the speaker responsible, or the person named as responsible. "
    "due is an ISO date (YYYY-MM-DD) only when a date is actually stated or "
    "unambiguously derivable from the meeting date; otherwise null. "
    "kind is 'event' for something scheduled at a time, 'todo' otherwise. "
    "quote is the shortest verbatim span that supports the item. "
    "at is the timecode in seconds where that quote occurs. "
    "Invent nothing. An empty list is a valid answer."
)

CLEANUP = (
    "Remove disfluencies from meeting transcript lines. Write in the input's language. "
    "Keep every line, its timecode prefix, and its speaker heading exactly as given. "
    "Remove filler words, false starts, and stutters only. "
    "Do not paraphrase, summarize, reorder, merge, or add anything. "
    "If a line has no disfluency, return it byte-identical."
)

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
