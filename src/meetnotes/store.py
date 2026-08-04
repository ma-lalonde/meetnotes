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
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

META = "meeting.json"
LOCK = "meeting.lock"

_process_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
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
    return json.loads((path / META).read_text(encoding="utf-8"))


def write_meta(path: Path, meta: dict) -> None:
    write_atomic(path / META, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


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
        if not child.is_dir() or not (child / META).exists():
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
        # Append mode: opening must not truncate, or the lock file's mtime
        # churns on every run and the meeting folder never looks unchanged.
        handle = (path / LOCK).open("a")
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
