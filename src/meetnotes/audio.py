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
import math
import shlex
import shutil
import signal
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Source:
    name: str
    description: str
    kind: str
    target: str = ""
    pulse_name: str = ""

    @property
    def label(self) -> str:
        return self.description or self.name


def _pw_dump() -> list[dict]:
    if not shutil.which("pw-dump"):
        return []
    try:
        out = subprocess.run(
            ["pw-dump"], capture_output=True, text=True, timeout=4, check=True
        ).stdout
        return json.loads(out)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return []


_cache: tuple[float, list["Source"]] | None = None
_cache_lock = threading.Lock()
CACHE_SECONDS = 3.0


def list_sources(refresh: bool = False) -> list[Source]:
    """Cached, because every caller otherwise spawns another pw-dump.

    Startup used to run it three times before the window appeared.
    """
    global _cache
    with _cache_lock:
        if not refresh and _cache and (time.monotonic() - _cache[0]) < CACHE_SECONDS:
            return _cache[1]
    found = _list_sources_uncached()
    with _cache_lock:
        _cache = (time.monotonic(), found)
    return found


def _list_sources_uncached() -> list[Source]:
    """Enumerate capture targets as PipeWire nodes.

    Deliberately not pactl: PulseAudio exposes system audio as a synthetic
    "<sink>.monitor" source that has no matching PipeWire node, so passing that
    name to pw-record matches nothing and silently falls back to the default
    input. In PipeWire a sink is captured by targeting the sink node itself,
    which links to its monitor ports.
    """
    sources = []
    for entry in _pw_dump():
        if entry.get("type") != "PipeWire:Interface:Node":
            continue
        props = (entry.get("info") or {}).get("props") or {}
        media_class = props.get("media.class", "")
        if media_class not in ("Audio/Source", "Audio/Sink"):
            continue
        name = props.get("node.name", "")
        if not name:
            continue
        serial = props.get("object.serial")
        is_sink = media_class == "Audio/Sink"
        sources.append(
            Source(
                name=name,
                description=props.get("node.description") or props.get("node.nick") or name,
                kind="system" if is_sink else "mic",
                target=str(serial) if serial is not None else name,
                # PulseAudio's view of the same thing, for backends that speak
                # pulse rather than PipeWire.
                pulse_name=f"{name}.monitor" if is_sink else name,
            )
        )
    sources.sort(key=lambda s: (s.kind != "mic", s.description.lower()))
    return sources


def default_sources() -> tuple[str, str]:
    """Prefer the nodes the session actually defaults to, not list order."""
    defaults = _default_node_names()
    mic = system = ""
    for source in list_sources():
        if source.kind == "mic":
            if source.name == defaults.get("source"):
                mic = source.target
            elif not mic:
                mic = source.target
        elif source.kind == "system":
            if source.name == defaults.get("sink"):
                system = source.target
            elif not system:
                system = source.target
    return mic, system


def _default_node_names() -> dict[str, str]:
    if not shutil.which("pactl"):
        return {}
    try:
        out = subprocess.run(
            ["pactl", "info"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    found = {}
    for line in out.splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if key == "Default Sink":
            found["sink"] = value
        elif key == "Default Source":
            # pipewire-pulse reports the monitor when a sink is the default
            # source; strip the suffix back to the underlying node name.
            found["source"] = value[: -len(".monitor")] if value.endswith(".monitor") else value
    return found


class Ambiguous(ValueError):
    pass


def resolve(query: str, kind: str) -> str:
    """Match a source by node name, serial, or substring. Returns a target."""
    candidates = [s for s in list_sources() if s.kind == kind]
    if not candidates:
        raise ValueError(f"no {kind} sources found")

    for source in candidates:
        if query in (source.name, source.target):
            return source.target

    needle = query.casefold()
    matches = [
        s for s in candidates if needle in s.name.casefold() or needle in s.description.casefold()
    ]
    if not matches:
        raise ValueError(f"no {kind} source matching {query!r}")
    if len(matches) > 1:
        names = "\n  ".join(f"{s.description}  ({s.name})" for s in matches)
        raise Ambiguous(f"{query!r} matches {len(matches)} {kind} sources:\n  {names}")
    return matches[0].target


def describe(target: str) -> str:
    for source in list_sources():
        if source.target == target or source.name == target:
            return f"{source.description} [{source.kind}]"
    return target or "(none)"


def data_offset(path: Path) -> int:
    """Byte offset of WAV sample data, or 0 for headerless raw captures.

    The canonical header is 44 bytes, but writers are free to insert LIST or
    fact chunks, so the offset has to be read rather than assumed.
    """
    try:
        with path.open("rb") as fh:
            if fh.read(4) != b"RIFF":
                return 0
            fh.seek(8)
            if fh.read(4) != b"WAVE":
                return 0
            while True:
                header = fh.read(8)
                if len(header) < 8:
                    return 0
                chunk_id, size = struct.unpack("<4sI", header)
                if chunk_id == b"data":
                    return fh.tell()
                fh.seek(size + (size & 1), 1)
    except OSError:
        return 0


RECORD_BACKENDS = {
    "pw-record": "pw-record --target={target} --rate={rate} --channels=1 --format=s16 {path}",
    "pw-record-sink": (
        "pw-record --capture-sink --target={target} --rate={rate} "
        "--channels=1 --format=s16 {path}"
    ),
    "parec": (
        "parec --device={pulse} --rate={rate} --channels=1 --format=s16le "
        "--file-format=wav {path}"
    ),
    "ffmpeg-pulse": (
        "ffmpeg -hide_banner -loglevel error -f pulse -i {pulse} "
        "-ac 1 -ar {rate} -y {path}"
    ),
}


def build_command(template: str, target: str, pulse: str, rate: int, path: Path) -> str:
    return template.format(target=target, pulse=pulse or target, rate=rate, path=str(path))


def rms_dbfs(samples: "np.ndarray") -> float:
    import numpy as np

    if samples.size == 0:
        return -120.0
    value = float(np.sqrt(np.mean(np.square(samples))))
    return 20.0 * math.log10(value) if value > 1e-6 else -120.0


def measure(cmd: str, path: Path, seconds: float = 2.5) -> dict:
    """Run a capture command briefly and report whether audio actually arrived.

    Uses the same command shape as real recording, so a positive result here
    means recording will work rather than merely that a process started.
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    try:
        proc = subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except (OSError, ValueError) as exc:
        return {"ok": False, "peak": -120.0, "rms": -120.0, "error": str(exc), "bytes": 0}

    time.sleep(seconds)
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""

    if not path.exists():
        return {"ok": False, "peak": -120.0, "rms": -120.0, "error": err or "no file", "bytes": 0}
    offset = data_offset(path)
    raw = path.read_bytes()[offset:]
    raw = raw[: len(raw) - (len(raw) % 2)]
    if not raw:
        return {"ok": False, "peak": -120.0, "rms": -120.0, "error": err or "no samples", "bytes": 0}
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return {
        "ok": True,
        "bytes": len(raw),
        "rms": rms_dbfs(samples),
        "peak": 20.0 * math.log10(peak) if peak > 1e-6 else -120.0,
        "error": err,
    }


def backend_supports_capture_sink() -> bool:
    if not shutil.which("pw-record"):
        return False
    try:
        out = subprocess.run(
            ["pw-record", "--help"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "capture-sink" in (out.stdout + out.stderr)


def candidate_backends(kind: str) -> list[str]:
    names = []
    if shutil.which("pw-record"):
        if kind == "system" and backend_supports_capture_sink():
            names.append("pw-record-sink")
        names.append("pw-record")
    if shutil.which("parec"):
        names.append("parec")
    if shutil.which("ffmpeg"):
        names.append("ffmpeg-pulse")
    return names


class Meter(threading.Thread):
    """Continuously reports the level of one source, using the real command."""

    def __init__(self, template: str, rate: int, source: Source, workdir: Path, sink):
        super().__init__(daemon=True, name=f"meter-{source.kind}")
        self.template = template
        self.rate = rate
        self.source = source
        self.path = workdir / f"meter-{source.kind}.wav"
        self.sink = sink
        self.stop_flag = threading.Event()
        self.error = ""

    def run(self) -> None:
        import numpy as np

        cmd = build_command(self.template, self.source.target, self.source.pulse_name,
                            self.rate, self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        try:
            proc = subprocess.Popen(
                shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            self.sink(-120.0)
            return

        offset = None
        read = 0
        try:
            while not self.stop_flag.is_set():
                time.sleep(0.1)
                if proc.poll() is not None:
                    self.error = (
                        proc.stderr.read().decode(errors="replace").strip()
                        if proc.stderr else "capture stopped"
                    )
                    break
                if not self.path.exists():
                    continue
                if offset is None:
                    offset = data_offset(self.path)
                with self.path.open("rb") as fh:
                    fh.seek(offset + read)
                    chunk = fh.read()
                chunk = chunk[: len(chunk) - (len(chunk) % 2)]
                if not chunk:
                    continue
                read += len(chunk)
                samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
                self.sink(rms_dbfs(samples))
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self.path.unlink(missing_ok=True)

    def stop(self) -> None:
        self.stop_flag.set()


class FileMeter(threading.Thread):
    """Reports the level of a recording already in progress.

    Used while recording so the meters keep moving without opening a second
    capture stream on the same device.
    """

    def __init__(self, path: Path, sink, interval: float = 0.15):
        super().__init__(daemon=True, name=f"filemeter-{path.stem}")
        self.path = path
        self.sink = sink
        self.interval = interval
        self.stop_flag = threading.Event()
        self.error = ""

    def run(self) -> None:
        import numpy as np

        offset = None
        read = 0
        while not self.stop_flag.is_set():
            time.sleep(self.interval)
            if not self.path.exists():
                continue
            if offset is None:
                offset = data_offset(self.path)
            try:
                with self.path.open("rb") as fh:
                    fh.seek(offset + read)
                    chunk = fh.read()
            except OSError:
                continue
            chunk = chunk[: len(chunk) - (len(chunk) % 2)]
            if not chunk:
                continue
            read += len(chunk)
            samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
            self.sink(rms_dbfs(samples))

    def stop(self) -> None:
        self.stop_flag.set()


class Recorder:
    def __init__(self, cmd_template: str, rate: int, tracks: dict[str, str], out_dir: Path):
        self.cmd_template = cmd_template
        self.rate = rate
        self.tracks = {label: target for label, target in tracks.items() if target}
        self.out_dir = out_dir
        self.procs: dict[str, subprocess.Popen] = {}
        self.paths: dict[str, Path] = {}
        self.started_at = 0.0

    def start(self) -> None:
        if not self.tracks:
            raise RuntimeError("no capture sources configured")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        known = {s.target: s for s in list_sources()}
        for label, target in self.tracks.items():
            path = self.out_dir / f"{label}.wav"
            source = known.get(target)
            cmd = build_command(
                self.cmd_template, target, source.pulse_name if source else "", self.rate, path
            )
            self.procs[label] = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self.paths[label] = path

    def failed(self) -> dict[str, str]:
        dead = {}
        for label, proc in self.procs.items():
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
                dead[label] = err or f"exited with code {proc.returncode}"
        return dead

    def silent(self) -> list[str]:
        """Tracks whose file has grown by nothing beyond its header."""
        quiet = []
        for label, path in self.paths.items():
            if not path.exists() or path.stat().st_size <= max(data_offset(path), 44):
                quiet.append(label)
        return quiet

    def stop(self) -> float:
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in self.procs.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        return time.time() - self.started_at
