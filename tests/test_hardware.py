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

import pytest

from meetnotes import hardware
from meetnotes.config import Config

GPU = [{"name": "NVIDIA GeForce RTX 4060 Laptop GPU", "vram_mb": 8188}]


def state(monkeypatch, gpus, dirs, usable, cublas=None):
    monkeypatch.setattr(hardware, "nvidia", lambda: gpus)
    monkeypatch.setattr(hardware, "cuda_lib_dirs", lambda: dirs)
    monkeypatch.setattr(hardware, "cuda_runtime_ok", lambda: usable)
    # Defaults to matching usable, since a working GPU implies cuBLAS loaded.
    monkeypatch.setattr(
        hardware, "cublas_ok", lambda: usable if cublas is None else cublas
    )
    return hardware.cuda_state()


def test_working_gpu_without_wheels_needs_no_install(monkeypatch):
    # System CUDA libraries already serve ctranslate2, so offering a 1.4 GB
    # download would be wrong.
    result = state(monkeypatch, GPU, [], True)
    assert result["usable"] is True
    assert result["installable"] is False
    assert "system CUDA libraries" in result["detail"]


def test_working_gpu_with_wheels_needs_no_install(monkeypatch):
    result = state(monkeypatch, GPU, ["/x/nvidia/cublas/lib"], True)
    assert result["installable"] is False
    assert "pip wheels" in result["detail"]


def test_gpu_without_usable_libraries_is_installable(monkeypatch):
    result = state(monkeypatch, GPU, [], False)
    assert result["installable"] is True


def test_a_visible_gpu_without_cublas_is_installable(monkeypatch):
    # The reported case: the driver is fine and the card is visible, so the
    # device count check passed, but ctranslate2 cannot run without cuBLAS.
    # Keying "installable" off that check said there was nothing to install.
    result = state(monkeypatch, GPU, [], usable=False, cublas=False)
    assert result["installable"] is True
    assert "cuBLAS is missing" in result["detail"]


def test_half_installed_wheels_are_still_installable(monkeypatch):
    # A directory exists but cuBLAS will not load. Re-running the install is
    # exactly the right move, and "nothing to install" was not.
    result = state(monkeypatch, GPU, ["/x/nvidia/cudnn/lib"], usable=False, cublas=False)
    assert result["installable"] is True


def test_libraries_that_load_but_do_not_work_point_at_the_driver(monkeypatch):
    # cuBLAS loads, so downloading it again changes nothing.
    result = state(monkeypatch, GPU, ["/x/nvidia/cudnn/lib"], usable=False, cublas=True)
    assert result["installable"] is False
    assert "driver" in result["detail"]


def test_no_gpu_is_never_installable(monkeypatch):
    result = state(monkeypatch, [], [], False)
    assert result["installable"] is False
    assert "No NVIDIA GPU" in result["detail"]


def test_a_visible_card_without_cublas_is_not_usable(monkeypatch):
    # get_cuda_device_count only talks to the driver, so on its own it reported
    # a working GPU right up until a model tried to load.
    import types

    monkeypatch.setattr(hardware, "_cuda_ok", None, raising=False)
    monkeypatch.setattr(hardware, "preload_cuda", lambda: False)
    monkeypatch.setattr(hardware, "cublas_ok", lambda: False)
    monkeypatch.setitem(
        __import__("sys").modules, "ctranslate2",
        types.SimpleNamespace(get_cuda_device_count=lambda: 1),
    )
    assert hardware.cuda_runtime_ok(refresh=True) is False


def test_gpu_profile_selects_turbo_on_cuda(monkeypatch):
    monkeypatch.setattr(hardware, "cuda_runtime_ok", lambda: True)
    plan = hardware.plan(Config())
    assert plan["profile"] == "gpu"
    assert plan["live_model"] == "large-v3-turbo"
    assert plan["final_model"] == "large-v3-turbo"
    assert (plan["device"], plan["compute_type"]) == ("cuda", "float16")


def test_cpu_profile_when_cuda_absent(monkeypatch):
    monkeypatch.setattr(hardware, "cuda_runtime_ok", lambda: False)
    plan = hardware.plan(Config())
    # base, not small: only the two smallest reliably beat speech on a CPU, and
    # a live pass that falls behind never catches up.
    assert (plan["profile"], plan["live_model"], plan["device"]) == ("cpu", "base", "cpu")


def test_cuda_request_downgrades_when_runtime_missing(monkeypatch):
    monkeypatch.setattr(hardware, "cuda_runtime_ok", lambda: False)
    cfg = Config()
    cfg.asr.device = "cuda"  # explicit device wins over the profile
    plan = hardware.plan(cfg)
    assert (plan["device"], plan["compute_type"]) == ("cpu", "int8")


def test_final_pass_off_skips_the_final_model(monkeypatch):
    monkeypatch.setattr(hardware, "cuda_runtime_ok", lambda: True)
    cfg = Config()
    cfg.asr.final_pass = False
    assert hardware.plan(cfg)["final_model"] == ""
