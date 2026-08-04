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

"""Speech recognition in a child process, so its VRAM can actually come back.

CTranslate2 allocates GPU memory through a caching allocator (cuda_malloc_async
by default on CUDA 11.2 and later). The cache is deliberately never returned to
the driver while the process lives, so deleting the model, collecting, and
waiting all leave the memory held. Process exit is the only release.

That matters here because the language model has to fit on the same card right
after transcription finishes. Running the recogniser in a child process and
letting it exit is what frees the card; nothing inside the process can.
"""

import json
import signal
import sys
from pathlib import Path

from . import asr
from .config import Config


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_file(request: dict) -> int:
    segments = asr.transcribe_file(
        Path(request["path"]),
        Config.from_dict(request["config"]),
        request["plan"],
        request.get("extra_terms") or None,
    )
    _emit({"segments": segments})
    return 0


def run_live(request: dict) -> int:
    # The parent closes stdin to ask for a clean stop, and the engine's own
    # final flush then emits whatever was still buffered.
    track = asr.LiveTrack(
        Path(request["path"]),
        request["speaker"],
        Config.from_dict(request["config"]),
        request["plan"],
        lambda segment: _emit({"segment": segment}),
    )
    signal.signal(signal.SIGTERM, lambda *_: track.stop())
    track.start()
    try:
        sys.stdin.read()
    except (OSError, ValueError):
        pass
    track.stop()
    track.join(timeout=120)
    if track.error:
        _emit({"error": track.error})
        return 1
    return 0


def main() -> int:
    request = json.loads(sys.stdin.readline())
    try:
        if request["mode"] == "file":
            return run_file(request)
        return run_live(request)
    except Exception as exc:  # the parent has no other way to see this
        _emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
