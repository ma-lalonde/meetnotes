import struct

import pytest

from meetnotes import audio

SOURCES = [
    audio.Source("alsa_input.pci-0000_00_1f.3.analog-stereo", "Built-in Microphone", "mic", "41"),
    audio.Source("bluez_input.AC_12_34.headset", "Sony WH-1000XM4", "mic", "57"),
    audio.Source("alsa_output.pci-0000_00_1f.3.analog-stereo", "Built-in Speakers", "system", "39"),
]


@pytest.fixture(autouse=True)
def sources(monkeypatch):
    monkeypatch.setattr(audio, "list_sources", lambda: SOURCES)
    monkeypatch.setattr(audio, "_default_node_names", lambda: {})


def test_resolve_by_node_name_returns_target():
    assert audio.resolve(SOURCES[1].name, "mic") == "57"


def test_resolve_by_serial():
    assert audio.resolve("57", "mic") == "57"


def test_resolve_by_description_substring():
    assert audio.resolve("sony", "mic") == "57"


def test_resolve_is_case_insensitive():
    assert audio.resolve("XM4", "mic") == "57"


def test_resolve_rejects_ambiguous_match():
    with pytest.raises(audio.Ambiguous):
        audio.resolve("input", "mic")


def test_resolve_rejects_unknown():
    with pytest.raises(ValueError):
        audio.resolve("webcam", "mic")


def test_system_kind_is_a_sink_not_a_monitor_name():
    # PipeWire has no "<sink>.monitor" node; capturing a sink yields its monitor.
    system = audio.resolve("speakers", "system")
    assert system == "39"
    assert not any(s.name.endswith(".monitor") for s in SOURCES)


def test_defaults_follow_session_defaults(monkeypatch):
    monkeypatch.setattr(
        audio, "_default_node_names",
        lambda: {"source": SOURCES[1].name, "sink": SOURCES[2].name},
    )
    assert audio.default_sources() == ("57", "39")


def test_defaults_fall_back_to_first_of_each_kind():
    assert audio.default_sources() == ("41", "39")


def _wav(tmp_path, extra_chunk=b"", payload=b"\x01\x02"):
    body = b"WAVE"
    body += b"fmt " + struct.pack("<I", 16) + struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    body += extra_chunk
    body += b"data" + struct.pack("<I", len(payload)) + payload
    path = tmp_path / "t.wav"
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + body)
    return path


def test_data_offset_canonical_header(tmp_path):
    assert audio.data_offset(_wav(tmp_path)) == 44


def test_data_offset_survives_extra_chunks(tmp_path):
    extra = b"LIST" + struct.pack("<I", 4) + b"INFO"
    assert audio.data_offset(_wav(tmp_path, extra)) == 56


def test_data_offset_zero_for_raw(tmp_path):
    raw = tmp_path / "raw.pcm"
    raw.write_bytes(b"\x00" * 32)
    assert audio.data_offset(raw) == 0


def test_rms_of_half_amplitude_sine():
    import numpy as np

    tone = np.sin(np.linspace(0, 400 * np.pi, 16000)).astype(np.float32) * 0.5
    assert audio.rms_dbfs(tone) == pytest.approx(-9.03, abs=0.1)


def test_rms_of_silence_is_floor():
    import numpy as np

    assert audio.rms_dbfs(np.zeros(1000, dtype=np.float32)) == -120.0


def test_build_command_substitutes_pulse_name():
    cmd = audio.build_command(
        audio.RECORD_BACKENDS["parec"], "39", "sink.monitor", 16000, "/tmp/x.wav"
    )
    assert "--device=sink.monitor" in cmd
    assert "39" not in cmd


def test_build_command_falls_back_to_target_without_pulse_name():
    cmd = audio.build_command(audio.RECORD_BACKENDS["parec"], "39", "", 16000, "/tmp/x.wav")
    assert "--device=39" in cmd
