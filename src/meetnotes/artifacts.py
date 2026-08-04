import hashlib
import json
from datetime import datetime
from pathlib import Path

from .store import write_atomic

MISSING = "missing"
FRESH = "fresh"
STALE = "stale"
EDITED = "hand-edited"


def sha(*parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if not isinstance(part, (str, bytes)):
            part = json.dumps(part, sort_keys=True, ensure_ascii=False, default=str)
        digest.update(part.encode("utf-8") if isinstance(part, str) else part)
        digest.update(b"\x00")
    return "sha256:" + digest.hexdigest()


def file_hash(path: Path) -> str:
    return sha(path.read_bytes())


def status(root: Path, meta: dict, name: str, fingerprint: str) -> str:
    path = root / name
    record = meta.get("artifacts", {}).get(name)
    if not path.exists() or not record:
        return MISSING
    if file_hash(path) != record.get("output_hash"):
        return EDITED
    return FRESH if record.get("fingerprint") == fingerprint else STALE


def ensure(root: Path, meta: dict, name: str, fingerprint: str, render, force: bool = False) -> str:
    """Generate an artifact only when its inputs changed.

    Never overwrites a hand-edited file unless force is set. Returns the
    resulting status: 'written', 'skipped', or 'hand-edited'.
    """
    state = status(root, meta, name, fingerprint)
    if not force:
        if state == FRESH:
            return "skipped"
        if state == EDITED:
            return EDITED

    text = render()
    path = root / name
    write_atomic(path, text)
    meta.setdefault("artifacts", {})[name] = {
        "fingerprint": fingerprint,
        "output_hash": file_hash(path),
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    return "written"


def replace_set(root: Path, meta: dict, key: str, files: dict[str, str]) -> None:
    """Rewrite a whole directory-shaped artifact, deleting what we wrote before.

    Prevents orphans when an item disappears from a regenerated set.
    """
    for stale in meta.get(key, []):
        target = root / stale
        if target.exists() and stale not in files:
            target.unlink()
    for name, text in files.items():
        write_atomic(root / name, text)
    meta[key] = sorted(files)
