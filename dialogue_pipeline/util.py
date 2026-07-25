from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(data: Any) -> str:
    encoded = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def slugify(value: str, *, fallback: str = "item") -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_value).strip("_").lower()
    return slug or fallback


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("’", "'").replace("‘", "'")
    value = value.lower()
    value = re.sub(r"[^\w']+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def word_count(value: str) -> int:
    normalized = normalize_text(value)
    return len(normalized.split()) if normalized else 0


def is_nonverbal_script(value: str) -> bool:
    return bool(re.fullmatch(r"\s*\([^)]*\)\s*", value or ""))


def relpath_for_config(path: Path, project_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), project_dir.resolve())).as_posix()


def resolve_project_path(project_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def default_model_cache_root() -> Path:
    configured = os.environ.get("DIALOGUE_VA_MODEL_CACHE")
    if configured:
        return Path(os.path.expandvars(configured)).expanduser().resolve()

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        cache_base = (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
        return cache_base / "DialogueVAPipeline" / "models"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_base = (
        Path(xdg_cache_home).expanduser()
        if xdg_cache_home
        else Path.home() / ".cache"
    )
    return cache_base / "dialogue-va-pipeline" / "models"


def resolve_model_cache_root(
    project_dir: Path,
    transcription_settings: dict[str, Any],
) -> Path:
    configured = transcription_settings.get("model_cache")
    if not configured:
        return default_model_cache_root()
    expanded = Path(os.path.expandvars(str(configured))).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_dir / expanded).resolve()


def project_file_from_arg(project: str | Path) -> Path:
    path = Path(project).resolve()
    if path.is_dir():
        path = path / "project.json"
    if not path.is_file():
        raise FileNotFoundError(f"Project configuration not found: {path}")
    return path


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]
