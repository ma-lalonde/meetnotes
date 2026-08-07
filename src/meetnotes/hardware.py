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

import ctypes
import glob
import importlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

PROFILES = {
    "gpu": {
        "live_model": "large-v3-turbo",
        "final_model": "large-v3-turbo",
        "device": "cuda",
        "compute_type": "float16",
    },
    "cpu": {
        # base, not small: on a processor only the two smallest reliably
        # transcribe faster than speech arrives, and falling behind on the live
        # pass compounds without limit. The final pass is where accuracy is
        # bought back, at a latency nobody is waiting on.
        "live_model": "base",
        "final_model": "large-v3-turbo",
        "device": "cpu",
        "compute_type": "int8",
    },
}


def nvidia() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        total = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        used = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        gpus.append(
            {"name": parts[0], "vram_mb": total, "used_mb": used, "free_mb": max(total - used, 0)}
        )
    return gpus


def gpu_processes() -> list[dict]:
    """Who is holding VRAM right now, largest first.

    Free memory alone does not say whether the speech model, the language
    model, or something unrelated is squatting on the card, and that is the
    only question worth asking when a load fails.
    """
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        procs.append({"pid": parts[0], "used_mb": int(parts[1]), "name": parts[2]})
    return sorted(procs, key=lambda p: p["used_mb"], reverse=True)


CUDA_LIB_MODULES = ("nvidia.cublas.lib", "nvidia.cudnn.lib")
_preloaded = False


def cuda_lib_dirs() -> list[str]:
    """Directories of the pip-installed CUDA libraries, in dependency order."""
    dirs = []
    for module in CUDA_LIB_MODULES:
        try:
            found = importlib.import_module(module)
        except ImportError:
            continue
        if found.__file__:
            dirs.append(os.path.dirname(found.__file__))
    return dirs


def preload_cuda() -> bool:
    """Load cuBLAS and cuDNN into this process before ctranslate2 needs them.

    faster-whisper's own instructions say to export LD_LIBRARY_PATH, but the
    dynamic loader reads that at process start, so setting it from Python is
    too late. Loading the libraries explicitly with RTLD_GLOBAL achieves the
    same thing without asking anyone to wrap the launcher in a shell script.
    """
    global _preloaded
    if _preloaded:
        return True
    dirs = cuda_lib_dirs()
    if not dirs:
        return False
    loaded = False
    # Two passes: cuDNN links against cuBLAS, and glob order is not dependency
    # order, so anything that fails first time gets another chance.
    for _ in range(2):
        for directory in dirs:
            for lib in sorted(glob.glob(os.path.join(directory, "lib*.so*"))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    loaded = True
                except OSError:
                    continue
    _preloaded = loaded
    return loaded


_cuda_ok: bool | None = None


def cuda_runtime_ok(refresh: bool = False) -> bool:
    """Cached: the first call initializes CUDA, which on a hybrid-graphics
    laptop can take seconds while the discrete GPU wakes up. Doing that more
    than once, and never on the GUI thread, is the difference between a slow
    start and an apparent freeze.
    """
    global _cuda_ok
    if _cuda_ok is not None and not refresh:
        return _cuda_ok
    try:
        import ctranslate2
    except ImportError:
        _cuda_ok = False
        return False
    preload_cuda()
    try:
        _cuda_ok = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        _cuda_ok = False
    return _cuda_ok


def project_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def cuda_diagnostics() -> list[tuple[str, str]]:
    """Each check that decides GPU availability, so a 'no' is explainable."""
    rows = [("nvidia-smi on PATH", str(bool(shutil.which("nvidia-smi"))))]
    gpus = nvidia()
    rows.append(("nvidia-smi reports", ", ".join(g["name"] for g in gpus) or "no GPU"))

    try:
        import ctranslate2

        rows.append(("ctranslate2", getattr(ctranslate2, "__version__", "installed")))
    except ImportError:
        rows.append(("ctranslate2", "NOT INSTALLED"))
        return rows

    dirs = cuda_lib_dirs()
    rows.append(("cuBLAS/cuDNN wheels", ", ".join(dirs) if dirs else "not installed"))
    rows.append(("preload", "ok" if preload_cuda() else "nothing to load"))
    try:
        rows.append(("ctranslate2 CUDA devices", str(ctranslate2.get_cuda_device_count())))
    except Exception as exc:
        rows.append(("ctranslate2 CUDA devices", f"error: {exc}"))
    try:
        rows.append(("ctranslate2 compute types", str(sorted(
            ctranslate2.get_supported_compute_types("cuda")
        ))))
    except Exception as exc:
        rows.append(("ctranslate2 compute types", f"error: {exc}"))
    return rows


def cuda_state() -> dict:
    gpus = nvidia()
    dirs = cuda_lib_dirs()
    usable = cuda_runtime_ok()
    if usable:
        source = "pip wheels" if dirs else "system CUDA libraries"
        detail = f"GPU acceleration active, using the {source}"
    elif not gpus:
        detail = "No NVIDIA GPU detected, running on CPU"
    elif not dirs:
        detail = "GPU found but no usable CUDA libraries (the wheels are about 1.4 GB)"
    else:
        detail = "CUDA libraries installed but unusable; check the driver version"
    return {
        "gpus": gpus,
        "libs_installed": bool(dirs),
        "usable": usable,
        # Only worth downloading when the GPU exists and nothing already works.
        # A working system CUDA install makes the wheels redundant.
        "installable": bool(gpus) and not usable and not dirs,
        "detail": detail,
    }


def install_cuda(log=None) -> bool:
    """Install the CUDA extra into this project's environment."""
    root = project_root()
    if root is None:
        if log:
            log("cannot locate the project to sync")
        return False
    if not shutil.which("uv"):
        if log:
            log("uv is not on PATH")
        return False

    process = subprocess.Popen(
        ["uv", "sync", "--extra", "cuda"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        if log:
            log(line.rstrip())
    process.wait()
    if process.returncode != 0:
        return False
    if log:
        log("installed. Restart meetnotes to use the GPU.")
    return True


def detect_profile() -> str:
    return "gpu" if cuda_runtime_ok() else "cpu"


def plan(cfg) -> dict:
    """Resolve the ASR profile into concrete models and compute settings.

    Explicit config values always win over the profile defaults.
    """
    name = cfg.asr.profile if cfg.asr.profile in PROFILES else detect_profile()
    base = dict(PROFILES[name])
    base["profile"] = name
    if cfg.asr.live_model:
        base["live_model"] = cfg.asr.live_model
    if cfg.asr.final_model:
        base["final_model"] = cfg.asr.final_model
    if not cfg.asr.final_pass:
        base["final_model"] = ""
    if cfg.asr.device != "auto":
        base["device"] = cfg.asr.device
    if cfg.asr.compute_type != "auto":
        base["compute_type"] = cfg.asr.compute_type
    if base["device"] == "cuda" and not cuda_runtime_ok():
        base["device"] = "cpu"
        base["compute_type"] = "int8"
    return base


def provenance(cfg, plan_: dict | None = None) -> dict:
    """What actually ran, recorded per meeting so results are explainable."""
    resolved = plan_ or plan(cfg)
    gpus = nvidia()
    return {
        "profile": resolved["profile"],
        "live_model": resolved["live_model"],
        "final_model": resolved["final_model"] or "(skipped)",
        "device": resolved["device"],
        "compute_type": resolved["compute_type"],
        "language": cfg.asr.language or "auto",
        "gpu": gpus[0]["name"] if gpus else "none",
        "llm_model": cfg.llm.model or "(none)",
    }


def venv_health() -> dict:
    """Detect the setup that makes uv rebuild the environment on every launch.

    uv validates ./.venv by resolving .venv/bin/python. Filesystems without
    symlink support (exFAT, NTFS, so most VeraCrypt containers) cannot hold
    that link, so uv deletes and reinstalls everything each time.
    """
    import sys

    prefix = Path(sys.prefix).resolve()
    root = project_root()
    inside = bool(root) and str(prefix).startswith(str(root.resolve()))

    symlinks = True
    if root:
        probe = root / ".meetnotes-symlink-probe"
        try:
            probe.symlink_to(root / "pyproject.toml")
            symlinks = probe.is_symlink()
        except (OSError, NotImplementedError):
            symlinks = False
        finally:
            try:
                probe.unlink()
            except OSError:
                pass

    return {
        "venv": str(prefix),
        "inside_project": inside,
        "symlinks_supported": symlinks,
        "rebuilds_every_launch": inside and not symlinks,
    }


def tray_available() -> bool:
    try:
        from PySide6.QtWidgets import QSystemTrayIcon
    except ImportError:
        return False
    try:
        return QSystemTrayIcon.isSystemTrayAvailable()
    except Exception:
        return False


def report(cfg=None) -> dict:
    from .config import Config

    cfg = cfg or Config()
    resolved = plan(cfg)
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "gpus": nvidia(),
        "cuda_runtime": cuda_runtime_ok(),
        "profile": resolved["profile"],
        "live_model": resolved["live_model"],
        "final_model": resolved["final_model"] or "(skipped)",
        "device": resolved["device"],
        "compute_type": resolved["compute_type"],
        "pactl": bool(shutil.which("pactl")),
        "pw_record": bool(shutil.which("pw-record")),
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP", ""),
        "session_type": os.environ.get("XDG_SESSION_TYPE", ""),
    }
