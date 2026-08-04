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

import errno
import fcntl
import json
import os
import re
import threading
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

RAW = "raw_output"
META = f"{RAW}/meeting.json"
LOCK = f"{RAW}/meeting.lock"
# Where these lived before the raw_output layout.
LEGACY_META = "meeting.json"
LEGACY_LOCK = "meeting.lock"


def meta_path(path: Path) -> Path:
    """Current location, falling back to the old one so a meeting recorded
    before the layout change is still recognised."""
    current = path / META
    if current.exists():
        return current
    legacy = path / LEGACY_META
    return legacy if legacy.exists() else current


def is_meeting(path: Path) -> bool:
    return (path / META).exists() or (path / LEGACY_META).exists()

_process_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def safe_label(name: str, fallback: str) -> str:
    """A speaker name that is also usable as a filename."""
    cleaned = re.sub(r"[^\w \-.']", "", (name or "").strip(), flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or fallback


# Letters NFKD does not decompose, which would otherwise vanish from a slug.
LIGATURES = {
    "æ": "ae", "œ": "oe", "ß": "ss", "ø": "o", "å": "aa",
    "đ": "d", "ð": "d", "ł": "l", "þ": "th", "ı": "i",
}


def slugify(title: str) -> str:
    # Decompose first, or an accented letter is dropped rather than folded:
    # "Chloe" with an acute e would otherwise slug to "chlo".
    lowered = "".join(LIGATURES.get(ch, ch) for ch in title.lower())
    folded = unicodedata.normalize("NFKD", lowered)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    slug = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    return slug or "meeting"


def write_atomic(path: Path, text: str) -> bool:
    """Write only when the bytes would actually change. Returns True if written.

    Compared as bytes, never read_text: universal newlines would translate the
    CRLF that RFC 5545 requires, so .ics files would look changed every run.
    Skipping identical writes is what makes "re-running touches nothing"
    literally true, mtimes included.
    """
    payload = text.encode("utf-8")
    if path.exists() and path.read_bytes() == payload:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(payload)
    os.replace(tmp, path)
    return True


def new_meeting(root: Path, title: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    path = root / f"{stamp}_{slugify(title)}"
    suffix = 2
    while path.exists():
        path = root / f"{stamp}_{slugify(title)}-{suffix}"
        suffix += 1
    (path / "audio").mkdir(parents=True)
    (path / RAW).mkdir(parents=True)
    write_meta(
        path,
        {
            "title": title,
            "created": datetime.now().isoformat(timespec="seconds"),
            "duration": 0.0,
            "tracks": {},
            "notes": [],
            "segments": [],
            "artifacts": {},
            "calendar_files": [],
            "state": "recording",
            "error": "",
        },
    )
    return path


def read_meta(path: Path) -> dict:
    return json.loads(meta_path(path).read_text(encoding="utf-8"))


def write_meta(path: Path, meta: dict) -> None:
    write_atomic(path / META, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    # Written in the new place, so the old copy is now stale.
    legacy = path / LEGACY_META
    if legacy.exists():
        legacy.unlink()


def update_meta(path: Path, **changes) -> dict:
    meta = read_meta(path)
    meta.update(changes)
    write_meta(path, meta)
    return meta


def list_meetings(root: Path) -> list[dict]:
    if not root.exists():
        return []
    out = []
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir() or not is_meeting(child):
            continue
        try:
            meta = read_meta(child)
        except (OSError, json.JSONDecodeError):
            continue
        meta["id"] = child.name
        meta["path"] = str(child)
        out.append(meta)
    return out


class Busy(RuntimeError):
    pass


@contextmanager
def exclusive(path: Path):
    """Serialise post-processing of one meeting across threads and processes.

    Raises Busy instead of blocking, so a second trigger returns immediately
    rather than queueing a duplicate run.
    """
    key = str(path.resolve())
    with _registry_lock:
        local = _process_locks.setdefault(key, threading.Lock())
    if not local.acquire(blocking=False):
        raise Busy(f"{path.name} is already being processed")

    handle = None
    try:
        lock = path / LOCK
        lock.parent.mkdir(parents=True, exist_ok=True)
        (path / LEGACY_LOCK).unlink(missing_ok=True)
        # Append mode: opening must not truncate, or the lock file's mtime
        # churns on every run and the meeting folder never looks unchanged.
        handle = lock.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise Busy(f"{path.name} is locked by another process") from exc
            raise
        yield
    finally:
        if handle is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            except OSError:
                pass
            handle.close()
        local.release()


def is_locked(path: Path) -> bool:
    lock = path / LOCK
    if not lock.exists():
        return False
    try:
        with lock.open("r+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
        return False
    except OSError:
        return True


def recover(root: Path) -> list[str]:
    """Reset meetings left mid-run by a crash. Returns the ids reset."""
    reset = []
    for meta in list_meetings(root):
        if meta.get("state") not in ("recording", "transcribing", "summarizing"):
            continue
        path = Path(meta["path"])
        if is_locked(path):
            continue
        update_meta(path, state="pending", error="interrupted, reset on startup")
        reset.append(meta["id"])
    return reset
