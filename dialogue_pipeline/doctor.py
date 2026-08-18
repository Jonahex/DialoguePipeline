from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from typing import Any

from .util import default_model_cache_root


def run_doctor() -> tuple[list[dict[str, Any]], bool]:
    checks: list[dict[str, Any]] = []
    ok = True

    checks.append(
        {
            "check": "python",
            "status": "ok" if sys.version_info >= (3, 11) else "error",
            "detail": sys.version.split()[0],
        }
    )
    if sys.version_info < (3, 11):
        ok = False

    for package in (
        "numpy",
        "openpyxl",
        "rapidfuzz",
        "faster_whisper",
        "sounddevice",
    ):
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            checks.append(
                {"check": f"python:{package}", "status": "ok", "detail": version}
            )
        except Exception as error:
            ok = False
            checks.append(
                {
                    "check": f"python:{package}",
                    "status": "error",
                    "detail": str(error),
                }
            )

    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        if path:
            checks.append(
                {"check": f"exe:{executable}", "status": "ok", "detail": path}
            )
        else:
            ok = False
            checks.append(
                {
                    "check": f"exe:{executable}",
                    "status": "error",
                    "detail": "not found on PATH",
                }
            )

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        checks.append(
            {
                "check": "gpu:nvidia",
                "status": "ok" if result.returncode == 0 else "warning",
                "detail": result.stdout.strip() or result.stderr.strip(),
            }
        )
    else:
        checks.append(
            {
                "check": "gpu:nvidia",
                "status": "warning",
                "detail": "nvidia-smi unavailable; CPU transcription will still work",
            }
        )

    model_cache = default_model_cache_root()
    cached_models = (
        sorted(
            path.name.removeprefix("models--").replace("--", "/")
            for path in model_cache.glob("models--*")
            if path.is_dir()
        )
        if model_cache.is_dir()
        else []
    )
    cache_detail = str(model_cache)
    if cached_models:
        cache_detail += " (" + ", ".join(cached_models) + ")"
    checks.append(
        {
            "check": "cache:models",
            "status": "ok",
            "detail": cache_detail,
        }
    )
    return checks, ok
