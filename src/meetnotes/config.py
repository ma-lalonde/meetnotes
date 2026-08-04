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
import os
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

from . import prompts

CONFIG_PATH = Path(
    os.environ.get("MEETNOTES_CONFIG", Path.home() / ".config/meetnotes/config.json")
)


@dataclass
class Capture:
    mic_source: str = ""
    system_source: str = ""
    mic_label: str = "Me"
    system_label: str = "Participants"
    sample_rate: int = 16000
    record_cmd: str = "pw-record --target={target} --rate={rate} --channels=1 --format=s16 {path}"


@dataclass
class Asr:
    profile: str = "auto"
    live_model: str = ""
    final_model: str = ""
    final_pass: bool = True
    device: str = "auto"
    compute_type: str = "auto"
    # "auto"      detect freely, any language
    # "restrict"  detect per chunk but only among `languages`
    # "primary"   pin `language`; foreign words are transcribed in place
    language_mode: str = "restrict"
    language: str = "fr"
    languages: list[str] = field(default_factory=lambda: ["fr", "en"])
    multilingual: bool = True
    detect_chunk_seconds: float = 20.0
    window_seconds: float = 6.0
    commit_seconds: float = 2.0
    live_beam_size: int = 1
    final_beam_size: int = 5
    # Needed to split a sentence at the exact word a note was typed against.
    word_timestamps: bool = True
    # Names and jargon the recogniser should expect. Whisper has no speaker
    # enrolment, but it does accept lexical hints.
    vocabulary: list[str] = field(default_factory=list)
    # Run GPU recognition in a child process. CTranslate2's caching allocator
    # never returns VRAM to the driver while the process lives, so without this
    # the speech model holds the card for as long as the app runs and the
    # language model has nothing left to load into. Off only for debugging.
    isolate_gpu: bool = True
    worker_timeout: float = 3600.0


@dataclass
class Diarization:
    enabled: bool = False
    model: str = "pyannote/speaker-diarization-community-1"
    hf_token: str = ""
    min_speakers: int = 0
    max_speakers: int = 0


@dataclass
class Llm:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = ""
    temperature: float = 0.2
    timeout: float = 600.0
    keep_asr_loaded: bool = False
    ttl_seconds: int = 300
    free_vram_before_recording: bool = True
    # Reload the model with a context length sized to the transcript. Needs the
    # lms CLI; ignored without it.
    auto_context: bool = True
    max_context: int = 0
    # How much of the model to put on the GPU: "max", "off", or 0-1. Lowering
    # this is what makes a model too big for the card loadable at all, at the
    # cost of speed.
    gpu_offload: str = "max"
    summary_prompt: str = prompts.SUMMARY
    actions_prompt: str = prompts.ACTIONS
    cleanup_prompt: str = prompts.CLEANUP


@dataclass
class Config:
    data_dir: str = str(Path.home() / "Meetings")
    auto_process: bool = True
    start_in_tray: bool = False
    minimize_on_quit: bool = True
    capture: Capture = field(default_factory=Capture)
    asr: Asr = field(default_factory=Asr)
    diarization: Diarization = field(default_factory=Diarization)
    llm: Llm = field(default_factory=Llm)

    @classmethod
    def load(cls) -> "Config":
        if not CONFIG_PATH.exists():
            return cls()
        try:
            raw = json.loads(CONFIG_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        """Rebuild from as_dict output, ignoring anything unrecognised.

        Also how a worker process receives the running settings, which may
        differ from what is on disk if they were changed but not saved.
        """
        kwargs = {}
        for spec in fields(cls):
            if spec.name not in raw:
                continue
            value = raw[spec.name]
            sub = spec.default_factory() if spec.default_factory is not MISSING else None
            if is_dataclass(sub):
                if isinstance(value, dict):
                    if spec.name == "llm":
                        value = prompts.migrate(value)
                    known = {f.name for f in fields(sub)}
                    kwargs[spec.name] = type(sub)(**{k: v for k, v in value.items() if k in known})
            else:
                kwargs[spec.name] = value
        return cls(**kwargs)

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, CONFIG_PATH)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def root(self) -> Path:
        return Path(self.data_dir).expanduser()
