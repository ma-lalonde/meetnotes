import threading
import time
from pathlib import Path

from . import asr, audio, hardware, llm, pipeline, store


class Session:
    """Owns one recording at a time, plus the background finalizer."""

    def __init__(self, cfg, on_segment=None, on_state=None, on_note=None):
        self.cfg = cfg
        self.on_segment = on_segment or (lambda seg: None)
        self.on_state = on_state or (lambda state, detail: None)
        self.on_note = on_note or (lambda note: None)
        self.lock = threading.Lock()
        self.path: Path | None = None
        self.recorder: audio.Recorder | None = None
        self.tracks: list[asr.LiveTrack] = []
        self.live: list[dict] = []
        self.notes: list[dict] = []
        self.started_at = 0.0

    @property
    def recording(self) -> bool:
        return self.recorder is not None

    def elapsed(self) -> float:
        return time.time() - self.started_at if self.recording else 0.0

    def _sink(self, segment: dict) -> None:
        with self.lock:
            self.live.append(segment)
            self.live.sort(key=lambda s: s["start"])
        self.on_segment(segment)

    def start(self, title: str) -> Path:
        if self.recording:
            raise RuntimeError("already recording")
        cap = self.cfg.capture
        if not cap.mic_source and not cap.system_source:
            raise RuntimeError("no audio sources selected")

        root = self.cfg.root
        root.mkdir(parents=True, exist_ok=True)
        self.path = store.new_meeting(root, title)
        self.live = []
        self.notes = []

        self.recorder = audio.Recorder(
            cap.record_cmd,
            cap.sample_rate,
            {cap.mic_label: cap.mic_source, cap.system_label: cap.system_source},
            self.path / "audio",
        )
        if self.cfg.llm.free_vram_before_recording:
            # The language model has no reason to hold VRAM during a meeting.
            freed, detail = llm.unload_all()
            if freed:
                self.on_state("idle", f"freed GPU memory: {detail}")

        self.recorder.start()
        self.started_at = self.recorder.started_at
        time.sleep(0.6)

        dead = self.recorder.failed()
        if dead:
            self.recorder.stop()
            self.recorder = None
            store.update_meta(self.path, state="failed", error=str(dead))
            path, self.path = self.path, None
            raise RuntimeError(f"capture failed: {dead}")

        plan = hardware.plan(self.cfg)
        self.tracks = [
            asr.LiveTrack(track_path, label, self.cfg, plan, self._sink)
            for label, track_path in self.recorder.paths.items()
        ]
        for track in self.tracks:
            track.start()
        self.on_state("recording", self.path.name)
        return self.path

    def add_note(self, text: str) -> dict:
        if not self.recording:
            raise RuntimeError("not recording")
        note = {"at": round(self.elapsed(), 2), "text": text.strip()}
        with self.lock:
            self.notes.append(note)
        store.update_meta(self.path, notes=self.notes)
        self.on_note(note)
        return note

    def stop(self, post_process: bool | None = None) -> Path:
        if not self.recording:
            raise RuntimeError("not recording")
        duration = self.recorder.stop()
        for track in self.tracks:
            track.stop()
        for track in self.tracks:
            track.join(timeout=120)

        errors = {t.speaker: t.error for t in self.tracks if t.error}
        path = self.path
        store.update_meta(
            path,
            duration=round(duration, 2),
            tracks={label: p.name for label, p in self.recorder.paths.items()},
            notes=self.notes,
            segments=sorted(self.live, key=lambda s: s["start"]),
            state="pending",
            error=str(errors) if errors else "",
        )
        self.recorder = None
        self.tracks = []
        self.path = None

        if post_process is None:
            post_process = self.cfg.auto_process
        if post_process:
            self.process_async(path)
        else:
            self.on_state("idle", path.name)
        return path

    def process_async(self, path: Path, force: bool = False) -> None:
        def run():
            try:
                self.on_state("processing", path.name)
                report = pipeline.process(
                    path, self.cfg, force=force,
                    progress=lambda step: self.on_state("processing", f"{path.name}: {step}"),
                )
                self.on_state("done", f"{path.name}: {_summarise(report)}")
            except store.Busy:
                self.on_state("idle", f"{path.name} already running")
            except Exception as exc:
                store.update_meta(path, state="failed", error=f"{type(exc).__name__}: {exc}")
                self.on_state("failed", f"{path.name}: {exc}")

        threading.Thread(target=run, daemon=True, name="finalizer").start()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "recording": self.recording,
                "elapsed": round(self.elapsed(), 1),
                "id": self.path.name if self.path else None,
                "segments": list(self.live),
                "notes": list(self.notes),
            }


def _summarise(report: dict) -> str:
    written = [name for name, action in report.items() if action == "written"]
    edited = [name for name, action in report.items() if action == "hand-edited"]
    parts = []
    if written:
        parts.append(f"{len(written)} written")
    if edited:
        parts.append(f"{len(edited)} kept (hand-edited)")
    if not parts:
        parts.append("nothing to do")
    return ", ".join(parts)
