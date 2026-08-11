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

"""The child process exists so CTranslate2's VRAM can be reclaimed on exit.

Nothing here loads a real model. What is tested is the boundary: that settings
survive the trip out and segments survive the trip back, that a stop is never
lost, and that a crashed child is reported rather than read as silence.
"""

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from meetnotes import asr
from meetnotes.config import Config


def fake_worker(tmp_path: Path, body: str) -> list[str]:
    """A stand-in child process, so the boundary is exercised for real."""
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import json, sys\n"
        "request = json.loads(sys.stdin.readline())\n"
        + textwrap.dedent(body)
    )
    # -u because a pipe is block-buffered by default; the real worker flushes
    # after every line instead.
    return [sys.executable, "-u", str(script)]


def test_config_survives_the_trip_to_the_worker():
    cfg = Config()
    cfg.asr.vocabulary = ["Godspeed You! Black Emperor", "Chloé"]
    cfg.asr.language_mode = "primary"
    rebuilt = Config.from_dict(json.loads(json.dumps(cfg.as_dict())))
    assert rebuilt.asr.vocabulary == ["Godspeed You! Black Emperor", "Chloé"]
    assert rebuilt.asr.language_mode == "primary"


def test_a_file_worker_returns_its_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "_worker_command", lambda: fake_worker(tmp_path, """
        print(json.dumps({"segments": [
            {"start": 0.0, "end": 1.0, "text": request["config"]["asr"]["language"]}
        ]}))
        """))
    segments = asr.transcribe_file_isolated(tmp_path / "a.wav", Config(), {"device": "cuda"})
    assert segments == [{"start": 0.0, "end": 1.0, "text": "fr"}]


def test_a_crashed_file_worker_is_an_error_not_an_empty_transcript(tmp_path, monkeypatch):
    # Reading a dead worker as "no speech" would produce a confident, empty
    # summary instead of a failure.
    monkeypatch.setattr(asr, "_worker_command", lambda: fake_worker(tmp_path, """
        sys.stderr.write("CUDA driver version is insufficient\\n")
        sys.exit(3)
        """))
    with pytest.raises(asr.WorkerError) as caught:
        asr.transcribe_file_isolated(tmp_path / "a.wav", Config(), {"device": "cuda"})
    assert "CUDA driver" in str(caught.value)


def test_a_worker_that_reports_an_error_is_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "_worker_command", lambda: fake_worker(tmp_path, """
        print(json.dumps({"error": "OSError: model not found"}))
        sys.exit(1)
        """))
    with pytest.raises(asr.WorkerError) as caught:
        asr.transcribe_file_isolated(tmp_path / "a.wav", Config(), {"device": "cuda"})
    assert "model not found" in str(caught.value)


def test_isolation_is_for_gpu_work_only():
    cfg = Config()
    assert asr.isolated(cfg, {"device": "cuda"})
    assert not asr.isolated(cfg, {"device": "cpu"})
    cfg.asr.isolate_gpu = False
    assert not asr.isolated(cfg, {"device": "cuda"})


def test_live_track_stays_in_process_on_cpu():
    made = asr.live_track(Path("a.wav"), "Me", Config(), {"device": "cpu"}, lambda s: None)
    assert isinstance(made, asr.LiveTrack)


def test_live_work_is_isolated_on_gpu():
    made = asr.live_track(Path("a.wav"), "Me", Config(), {"device": "cuda"}, lambda s: None)
    assert isinstance(made, asr.LiveWorker)


def test_a_live_worker_streams_segments_until_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "_worker_command", lambda: fake_worker(tmp_path, """
        print(json.dumps({"segment": {"speaker": request["speaker"], "text": "one"}}))
        sys.stdin.read()
        print(json.dumps({"segment": {"speaker": request["speaker"], "text": "tail"}}))
        """))
    got = []
    worker = asr.LiveWorker(tmp_path / "a.wav", "Me", Config(), {"device": "cuda"}, got.append)
    worker.start()
    for _ in range(200):
        if got:
            break
        time.sleep(0.02)
    assert got == [{"speaker": "Me", "text": "one"}]

    # Closing stdin is the clean stop, and the tail still arrives.
    worker.stop()
    worker.join(timeout=20)
    assert got[-1] == {"speaker": "Me", "text": "tail"}
    assert not worker.error


def test_a_stop_during_startup_is_not_lost(tmp_path, monkeypatch):
    # stop() can land before the child exists. Losing it there would leave the
    # recogniser running for the life of the application.
    monkeypatch.setattr(asr, "_worker_command", lambda: fake_worker(tmp_path, """
        sys.stdin.read()
        print(json.dumps({"segment": {"text": "flushed"}}))
        """))
    got = []
    worker = asr.LiveWorker(tmp_path / "a.wav", "Me", Config(), {"device": "cuda"}, got.append)
    worker.stop()
    worker.start()
    worker.join(timeout=20)
    assert not worker.is_alive()
    assert got == [{"text": "flushed"}]


def test_a_live_worker_that_dies_reports_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "_worker_command", lambda: fake_worker(tmp_path, """
        sys.stderr.write("no CUDA-capable device is detected\\n")
        sys.exit(4)
        """))
    worker = asr.LiveWorker(tmp_path / "a.wav", "Me", Config(), {"device": "cuda"}, lambda s: None)
    worker.start()
    worker.join(timeout=20)
    assert "no CUDA-capable device" in worker.error


def test_the_worker_loads_cuda_itself(monkeypatch):
    # The parent preloads through hardware.plan, but the plan arrives here
    # already resolved, so nothing in the child triggered it and CTranslate2
    # could not find libcublas. The loader reads LD_LIBRARY_PATH at process
    # start, so it cannot be inherited either.
    from meetnotes import hardware, worker

    called = []
    monkeypatch.setattr(hardware, "preload_cuda", lambda: called.append(True) or True)
    worker.prepare({"device": "cuda"})
    assert called == [True]


def test_the_worker_does_not_touch_cuda_on_cpu(monkeypatch):
    from meetnotes import hardware, worker

    monkeypatch.setattr(hardware, "preload_cuda", lambda: pytest.fail("not on cpu"))
    worker.prepare({"device": "cpu"})


def test_a_missing_cuda_library_is_named_not_left_to_ctranslate2(monkeypatch):
    from meetnotes import hardware, worker

    monkeypatch.setattr(hardware, "preload_cuda", lambda: False)
    with pytest.raises(RuntimeError) as caught:
        worker.prepare({"device": "cuda"})
    assert "meetnotes gpu --install" in str(caught.value)


def test_the_worker_reports_a_cuda_failure_over_the_boundary(tmp_path):
    # The parent has no other way to see it: a child that dies without a
    # message reads as a transcript with no speech.
    done = subprocess.run(
        [sys.executable, "-m", "meetnotes.worker"],
        input=json.dumps({
            "mode": "file", "path": str(tmp_path / "a.wav"),
            "config": {}, "plan": {"device": "cuda"},
        }) + "\n",
        capture_output=True, text=True, timeout=120,
    )
    payload = json.loads(done.stdout.strip().splitlines()[-1])
    assert done.returncode == 1
    assert "error" in payload


def test_the_real_worker_module_starts_and_reports_a_bad_request():
    # Proves the module is importable and runnable as a child process, which a
    # monkeypatched command can never show.
    done = subprocess.run(
        [sys.executable, "-m", "meetnotes.worker"],
        input='{"mode": "file", "path": "/nonexistent.wav", "config": {}, "plan": {}}\n',
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 1
    assert "error" in json.loads(done.stdout.strip().splitlines()[-1])
