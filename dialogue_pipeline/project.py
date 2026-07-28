from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from .alignment_settings import default_alignment_config
from .audio import probe_audio
from .cancellation import check_processing_cancelled
from .util import (
    read_json,
    relpath_for_config,
    resolve_project_path,
    slugify,
    write_json,
)
from .workbook_io import parse_workbook


PROJECT_STRUCTURE_KEYS = {
    "schema_version",
    "workbook",
    "audio_dir",
    "source_lines",
    "audio_inventory",
    "sessions",
}


def _name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _find_named_sheet(stem: str, sheet_names: list[str]) -> str | None:
    stem_key = _name_key(stem)
    matches = [
        name
        for name in sheet_names
        if len(_name_key(name)) >= 5 and _name_key(name) in stem_key
    ]
    if not matches:
        return None
    return max(matches, key=lambda name: len(_name_key(name)))


def infer_sessions(
    audio_files: list[Path], source_data: dict[str, Any], project_dir: Path
) -> list[dict[str, Any]]:
    sheet_names = [sheet["name"] for sheet in source_data["sheets"]]
    sheet_counts = {
        sheet["name"]: int(sheet["line_count"]) for sheet in source_data["sheets"]
    }
    generic_sheets = [
        name for name in sheet_names if sheet_counts.get(name, 0) >= 80
    ]
    speaking_sheet = next(
        (
            name
            for name in generic_sheets
            if "combat" not in next(
                sheet["voice_header"].lower()
                for sheet in source_data["sheets"]
                if sheet["name"] == name
            )
        ),
        generic_sheets[0] if generic_sheets else None,
    )
    combat_sheet = next(
        (
            name
            for name in generic_sheets
            if "combat"
            in next(
                sheet["voice_header"].lower()
                for sheet in source_data["sheets"]
                if sheet["name"] == name
            )
        ),
        generic_sheets[1] if len(generic_sheets) > 1 else None,
    )

    named_sheets = {
        match
        for audio_file in audio_files
        if (match := _find_named_sheet(audio_file.stem, sheet_names))
    }
    sessions = []

    for audio_file in sorted(audio_files, key=lambda path: path.name.lower()):
        stem_lower = audio_file.stem.lower()
        named_sheet = _find_named_sheet(audio_file.stem, sheet_names)
        excel_rows: list[int] = []
        row_match = re.search(r"row[_ -]?(\d+)", stem_lower)
        if row_match:
            excel_rows = [int(row_match.group(1))]

        needs_review = False
        if named_sheet:
            sheets = [named_sheet]
        elif "combatgenerics" in stem_lower or "speakinggenerics" in stem_lower:
            if stem_lower.find("combatgenerics") < stem_lower.find("speakinggenerics"):
                sheets = [name for name in [combat_sheet, speaking_sheet] if name]
            else:
                sheets = [name for name in [speaking_sheet, combat_sheet] if name]
            needs_review = not bool(sheets)
        elif "misctabs" in stem_lower:
            sheets = [
                name
                for name in sheet_names
                if name not in set(generic_sheets) | named_sheets
            ]
        else:
            sheets = sheet_names.copy()
            needs_review = True

        session_id = slugify(audio_file.stem)
        sessions.append(
            {
                "id": session_id,
                "enabled": True,
                "audio": relpath_for_config(audio_file, project_dir),
                "sheets": sheets,
                "excel_rows": excel_rows,
                "line_ids": [],
                "pass": _infer_pass(stem_lower),
                "needs_mapping_review": needs_review,
            }
        )
    return sessions


def _infer_pass(stem_lower: str) -> str:
    labels = []
    for token in ("pickup", "shouting", "speaking", "louder", "quieter", "combat"):
        if token in stem_lower:
            labels.append(token)
    return "+".join(labels) or "main"


def _merge_project_settings(
    target: dict[str, Any],
    overrides: dict[str, Any],
) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_project_settings(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def default_project_settings() -> dict[str, Any]:
    """Return every editable setting written to a newly created project."""
    return {
        "language": "en",
        "transcription": {
            "model": "large-v3",
            "model_cache": None,
            "device": "auto",
            "compute_type": "auto",
            "beam_size": 5,
            "batch_size": "auto",
            "batch_size_max": 32,
            "condition_on_previous_text": False,
            "vad_filter": True,
            "vad_min_silence_ms": 500,
        },
        "segment_transcription": {
            "enabled": True,
            "model": None,
            "device": None,
            "compute_type": None,
            "beam_size": 5,
            "batch_size": "auto",
            "batch_size_max": 32,
            "prompt_fallback_enabled": True,
            "prompt_fallback_max_segment_seconds": 6.0,
            "prompt_fallback_max_script_words": 8,
            "prompt_fallback_top_k": 8,
            "prompt_fallback_max_characters": 800,
            "prompt_fallback_trigger_probability": 0.55,
            "prompt_fallback_trigger_ordered_score": 70.0,
            "candidate_verification_enabled": True,
            "candidate_verification_min_match_score": 45.0,
        },
        "segmentation": {
            "silence_noise_db": -45.0,
            "silence_detection_min_seconds": 0.35,
            "split_gap_seconds": 0.35,
            "minimum_segment_seconds": 0.15,
            "pre_padding_seconds": 0.15,
            "post_padding_seconds": 0.25,
            "fade_ms": 5.0,
            "word_split_enabled": True,
            "word_split_gap_seconds": 0.3,
            "word_split_min_region_seconds": 1.5,
            "word_split_max_boundaries": 2,
        },
        "alignment": default_alignment_config(),
        "export": {
            "extension": ".wav",
            "sample_rate": 48000,
            "channels": 1,
            "bits_per_sample": 16,
            "allow_segment_reuse": False,
        },
    }


def editable_project_settings(project: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a complete editable settings view without structural project data."""
    if project is None:
        return default_project_settings()

    defaults = default_project_settings()
    result: dict[str, Any] = {}
    for key, default_value in defaults.items():
        if key not in project:
            result[key] = copy.deepcopy(default_value)
        elif (
            key != "alignment"
            and isinstance(default_value, dict)
            and isinstance(project[key], dict)
        ):
            merged = copy.deepcopy(default_value)
            _merge_project_settings(merged, project[key])
            result[key] = merged
        else:
            # Preserve the exact grouped/legacy alignment representation so
            # the dialog never creates conflicting duplicate keys.
            result[key] = copy.deepcopy(project[key])

    for key, value in project.items():
        if key not in PROJECT_STRUCTURE_KEYS and key not in result:
            result[key] = copy.deepcopy(value)
    return result


def apply_project_settings(
    project: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    """Replace editable groups while preserving paths, sessions, and metadata."""
    for key, value in settings.items():
        if key in PROJECT_STRUCTURE_KEYS:
            continue
        project[key] = copy.deepcopy(value)


def create_project(
    *,
    workbook_path: Path,
    audio_dir: Path,
    project_dir: Path,
    force: bool = False,
    project_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workbook_path = workbook_path.resolve()
    audio_dir = audio_dir.resolve()
    project_dir = project_dir.resolve()
    project_file = project_dir / "project.json"
    if project_file.exists() and not force:
        raise FileExistsError(
            f"Project already exists: {project_file}. Use --force to regenerate it."
        )
    if not workbook_path.is_file():
        raise FileNotFoundError(workbook_path)
    if not audio_dir.is_dir():
        raise NotADirectoryError(audio_dir)

    check_processing_cancelled()
    project_dir.mkdir(parents=True, exist_ok=True)
    source_data = parse_workbook(workbook_path)
    check_processing_cancelled()
    audio_files = sorted(audio_dir.glob("*.wav"), key=lambda path: path.name.lower())
    if not audio_files:
        raise ValueError(f"No WAV files found in {audio_dir}")

    inventory = []
    for index, audio_file in enumerate(audio_files, start=1):
        check_processing_cancelled()
        print(f"[inventory {index}/{len(audio_files)}] {audio_file.name}", flush=True)
        inventory.append(probe_audio(audio_file, include_hash=True))

    check_processing_cancelled()
    write_json(project_dir / "source_lines.json", source_data)
    write_json(project_dir / "audio_inventory.json", {"files": inventory})

    project = {
        "schema_version": 1,
        "workbook": relpath_for_config(workbook_path, project_dir),
        "audio_dir": relpath_for_config(audio_dir, project_dir),
        "source_lines": "source_lines.json",
        "audio_inventory": "audio_inventory.json",
        **default_project_settings(),
        "sessions": infer_sessions(audio_files, source_data, project_dir),
    }
    if project_settings:
        _merge_project_settings(project, project_settings)
    write_json(project_file, project)
    return project


def load_project(project_file: Path) -> tuple[Path, dict[str, Any]]:
    project_file = project_file.resolve()
    project_dir = project_file.parent
    project = read_json(project_file)
    return project_dir, project


def load_source_data(project_dir: Path, project: dict[str, Any]) -> dict[str, Any]:
    path = resolve_project_path(project_dir, project["source_lines"])
    return read_json(path)


def inventory_by_path(
    project_dir: Path, project: dict[str, Any]
) -> dict[Path, dict[str, Any]]:
    inventory_path = resolve_project_path(project_dir, project["audio_inventory"])
    inventory = read_json(inventory_path)
    return {
        Path(item["path"]).resolve(): item
        for item in inventory.get("files", [])
    }
