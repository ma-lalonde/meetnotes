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

import threading
import time
from pathlib import Path

import numpy as np

from . import audio

_models: dict[tuple, object] = {}
_model_lock = threading.Lock()


def language_args(cfg, detected: str | None = None) -> dict:
    """Whisper decodes one language per call, so code-switching needs help.

    primary  - pin one language. Embedded foreign words still get transcribed
               where they fall, which is what "mostly French with the odd
               English term" actually wants; per-segment switching would only
               fragment it.
    restrict - detect per chunk, but only among the languages that can occur.
               Stops a mumbled passage being labelled Welsh.
    auto     - let faster-whisper detect per segment, unconstrained.
    """
    mode = cfg.asr.language_mode
    if mode == "primary" and cfg.asr.language:
        return {"language": cfg.asr.language}
    if mode == "restrict" and detected:
        return {"language": detected}
    if mode == "restrict":
        return {"language": cfg.asr.languages[0] if cfg.asr.languages else None}
    return {"language": None, "multilingual": cfg.asr.multilingual}


def detect_restricted(model, samples, cfg) -> str:
    """Most likely language among the allowed set, not among all 99.

    detect_language returns probabilities for every language, so restricting
    is a matter of ignoring the ones that cannot occur rather than trusting
    the unconstrained argmax.
    """
    allowed = [code.strip() for code in cfg.asr.languages if code.strip()]
    if not allowed:
        return ""
    if len(allowed) == 1:
        return allowed[0]
    try:
        _, _, probabilities = model.detect_language(
            audio=samples, vad_filter=True,
            language_detection_segments=1,
        )
    except Exception:
        return allowed[0]
    ranked = {code: prob for code, prob in probabilities}
    return max(allowed, key=lambda code: ranked.get(code, 0.0))


def get_model(name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    key = (name, device, compute_type)
    with _model_lock:
        if key not in _models:
            _models[key] = WhisperModel(name, device=device, compute_type=compute_type)
        return _models[key]


def unload_all() -> None:
    with _model_lock:
        _models.clear()


def _pack(segments, offset: float, language: str) -> list[dict]:
    packed = []
    for s in segments:
        if not s.text.strip():
            continue
        entry = {
            "start": round(s.start + offset, 2),
            "end": round(s.end + offset, 2),
            "text": s.text.strip(),
            "language": language or getattr(s, "language", "") or "",
        }
        words = getattr(s, "words", None)
        if words:
            # Compact triples rather than dicts: an hour of speech is roughly
            # ten thousand of these and they only exist to locate a boundary.
            entry["words"] = [
                [round(w.start + offset, 2), round(w.end + offset, 2), w.word] for w in words
            ]
        packed.append(entry)
    return packed


def transcribe_file(path: Path, cfg, plan: dict) -> list[dict]:
    model = get_model(plan["final_model"], plan["device"], plan["compute_type"])
    allowed = [code for code in cfg.asr.languages if code.strip()]

    if cfg.asr.language_mode != "restrict" or len(allowed) < 2:
        segments, _ = model.transcribe(
            str(path),
            beam_size=cfg.asr.final_beam_size,
            vad_filter=True,
            word_timestamps=cfg.asr.word_timestamps,
            **language_args(cfg),
        )
        return _pack(segments, 0.0, language_args(cfg).get("language") or "")

    # Restricted mode needs the language decided per passage, and transcribe()
    # only accepts one language per call, so the file is walked in chunks.
    from faster_whisper import decode_audio

    rate = 16000
    samples = decode_audio(str(path), sampling_rate=rate)
    step = max(int(cfg.asr.detect_chunk_seconds * rate), rate)
    out = []
    for start in range(0, len(samples), step):
        piece = samples[start : start + step]
        if piece.size < rate // 2:
            continue
        language = detect_restricted(model, piece, cfg)
        segments, _ = model.transcribe(
            piece,
            beam_size=cfg.asr.final_beam_size,
            vad_filter=True,
            word_timestamps=cfg.asr.word_timestamps,
            language=language or None,
        )
        out.extend(_pack(segments, start / rate, language))
    out.sort(key=lambda s: s["start"])
    return out


def diarize(path: Path, cfg) -> list[dict]:
    """Seam for speaker diarization of a single track.

    Dual-track capture already gives exact speaker identity for one-on-one
    calls, so this is only needed when several people share one stream.
    Wiring it means pyannote.audio 4.x plus the gated
    pyannote/speaker-diarization-community-1 model, which pulls in torch.
    """
    raise NotImplementedError("install the 'diarize' extra and wire pyannote here")


class LiveTrack(threading.Thread):
    """Tails a growing WAV file and emits committed transcript segments."""

    def __init__(self, path: Path, speaker: str, cfg, plan: dict, sink):
        super().__init__(daemon=True, name=f"live-{speaker}")
        self.path = path
        self.speaker = speaker
        self.cfg = cfg
        self.plan = plan
        self.sink = sink
        self.stop_flag = threading.Event()
        self.error = ""
        self.header: int | None = None
        self.offset = 0
        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_start = 0.0

    def _read_new(self) -> np.ndarray:
        if not self.path.exists():
            return np.zeros(0, dtype=np.float32)
        if self.header is None:
            self.header = audio.data_offset(self.path)
            if self.header == 0 and self.path.stat().st_size < 64:
                self.header = None
                return np.zeros(0, dtype=np.float32)
        with self.path.open("rb") as fh:
            fh.seek(self.header + self.offset)
            raw = fh.read()
        if len(raw) < 2:
            return np.zeros(0, dtype=np.float32)
        raw = raw[: len(raw) - (len(raw) % 2)]
        self.offset += len(raw)
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def run(self) -> None:
        rate = self.cfg.capture.sample_rate
        window = int(self.cfg.asr.window_seconds * rate)
        commit = self.cfg.asr.commit_seconds
        try:
            model = get_model(self.plan["live_model"], self.plan["device"], self.plan["compute_type"])
            while not self.stop_flag.is_set():
                self.buffer = np.concatenate([self.buffer, self._read_new()])
                if len(self.buffer) < window:
                    time.sleep(0.4)
                    continue
                self._flush(model, rate, commit)

            self.buffer = np.concatenate([self.buffer, self._read_new()])
            if len(self.buffer) > rate // 2:
                self._flush(model, rate, 0.0, final=True)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def _flush(self, model, rate: int, commit: float, final: bool = False) -> None:
        samples = self.buffer
        duration = len(samples) / rate
        detected = ""
        if self.cfg.asr.language_mode == "restrict":
            detected = detect_restricted(model, samples, self.cfg)
        segments, _ = model.transcribe(
            samples,
            beam_size=self.cfg.asr.live_beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            **language_args(self.cfg, detected),
        )
        cutoff = duration - commit
        keep_from = duration
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if final or seg.end <= cutoff:
                self.sink(
                    {
                        "speaker": self.speaker,
                        "start": round(self.buffer_start + seg.start, 2),
                        "end": round(self.buffer_start + seg.end, 2),
                        "text": text,
                        "language": detected or getattr(seg, "language", "") or "",
                    }
                )
            else:
                keep_from = min(keep_from, seg.start)

        if final:
            self.buffer = np.zeros(0, dtype=np.float32)
            return
        drop = int(max(0.0, keep_from) * rate)
        self.buffer = samples[drop:]
        self.buffer_start += drop / rate

    def stop(self) -> None:
        self.stop_flag.set()
