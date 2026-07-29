from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from .alignment_settings import (
    complete_alignment_config,
    default_alignment_config,
)
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
    "settings_version",
    "workbook",
    "audio_dir",
    "source_lines",
    "audio_inventory",
    "sessions",
}
PROJECT_SETTINGS_VERSION = 3


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


def _is_combat_sheet(sheet: dict[str, Any]) -> bool:
    return "combat" in (
        f"{sheet.get('name', '')} {sheet.get('voice_header', '')}".lower()
    )


def _generic_sheet_pair(
    source_data: dict[str, Any],
) -> tuple[str | None, str | None]:
    sheets = list(source_data["sheets"])
    generic_sheets = [
        sheet for sheet in sheets if int(sheet.get("line_count", 0)) >= 80
    ]
    speaking = next(
        (sheet["name"] for sheet in generic_sheets if not _is_combat_sheet(sheet)),
        generic_sheets[0]["name"] if generic_sheets else None,
    )
    combat = next(
        (sheet["name"] for sheet in generic_sheets if _is_combat_sheet(sheet)),
        generic_sheets[1]["name"] if len(generic_sheets) > 1 else None,
    )
    return speaking, combat


def _bandit_sheet_pair(
    source_data: dict[str, Any],
) -> tuple[str | None, str | None]:
    bandit_sheets = [
        sheet
        for sheet in source_data["sheets"]
        if "bandit" in str(sheet.get("voice_header") or "").lower()
    ]
    speaking = next(
        (sheet["name"] for sheet in bandit_sheets if not _is_combat_sheet(sheet)),
        None,
    )
    combat = next(
        (sheet["name"] for sheet in bandit_sheets if _is_combat_sheet(sheet)),
        None,
    )
    return speaking, combat


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
    speaking_sheet, combat_sheet = _generic_sheet_pair(source_data)
    bandit_speaking_sheet, bandit_combat_sheet = _bandit_sheet_pair(source_data)

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
        elif "banditgeneric" in stem_lower:
            sheets = [bandit_speaking_sheet] if bandit_speaking_sheet else []
            needs_review = not bool(sheets)
        elif "banditcombat" in stem_lower:
            sheets = [bandit_combat_sheet] if bandit_combat_sheet else []
            needs_review = not bool(sheets)
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
        elif "generic" in stem_lower:
            sheets = [speaking_sheet] if speaking_sheet else []
            needs_review = not bool(sheets)
        elif "combat" in stem_lower:
            sheets = [combat_sheet] if combat_sheet else []
            needs_review = not bool(sheets)
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
            "batch_size_max": 4,
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
            "batch_size_max": 4,
            "prompt_fallback_enabled": True,
            "prompt_fallback_max_segment_seconds": 6.0,
            "prompt_fallback_max_script_words": 8,
            "prompt_fallback_top_k": 8,
            "prompt_fallback_max_characters": 800,
            "prompt_fallback_trigger_probability": 0.55,
            "prompt_fallback_trigger_ordered_score": 70.0,
            "candidate_verification_enabled": True,
            "candidate_verification_min_match_score": 45.0,
            "silence_rejection_enabled": True,
            "silence_rejection_max_rms_dbfs": -40.0,
            "silence_rejection_min_no_speech_probability": 0.80,
        },
        "segmentation": {
            "silence_noise_db": -40.0,
            "silence_detection_min_seconds": 0.20,
            "split_gap_seconds": 0.20,
            "minimum_segment_seconds": 0.15,
            "pre_padding_seconds": 0.15,
            "post_padding_seconds": 0.25,
            "fade_ms": 5.0,
            "word_split_enabled": True,
            "word_split_gap_seconds": 0.3,
            "word_split_min_region_seconds": 1.5,
            "word_split_max_boundaries": 2,
            "word_split_max_segment_seconds": 8.0,
            "word_split_snap_enabled": True,
            "word_split_snap_search_seconds": 0.20,
            "word_split_snap_window_seconds": 0.02,
            "word_split_snap_max_rms_dbfs": -42.0,
            "voice_boundary_detection_enabled": True,
            "voice_boundary_vad_threshold": 0.50,
            "voice_boundary_breath_vad_threshold": 0.70,
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

    project = migrate_project_config(project)
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
    project["settings_version"] = PROJECT_SETTINGS_VERSION


def migrate_project_config(project: dict[str, Any]) -> dict[str, Any]:
    """Return a complete current representation for a grouped-settings project."""

    migrated = copy.deepcopy(project)
    version = int(migrated.get("settings_version", 1))
    if version < 2:
        segmentation = migrated.get("segmentation")
        if isinstance(segmentation, dict):
            # Upgrade only values written by the previous defaults; explicit
            # project tuning remains untouched.
            if segmentation.get("silence_noise_db") == -45.0:
                segmentation["silence_noise_db"] = -40.0
            if segmentation.get("silence_detection_min_seconds") == 0.35:
                segmentation["silence_detection_min_seconds"] = 0.20
            if segmentation.get("split_gap_seconds") == 0.35:
                segmentation["split_gap_seconds"] = 0.20
    if version < 3:
        segmentation = migrated.get("segmentation")
        if not isinstance(segmentation, dict):
            segmentation = {}
            migrated["segmentation"] = segmentation
        configured_alignment = migrated.get("alignment")
        audio_boundaries = {}
        if isinstance(configured_alignment, dict):
            recovery = configured_alignment.get("recovery")
            if isinstance(recovery, dict):
                candidate = recovery.get("audio_boundaries")
                if isinstance(candidate, dict):
                    audio_boundaries = candidate
        segmentation.setdefault("voice_boundary_detection_enabled", True)
        segmentation.setdefault(
            "word_split_snap_enabled",
            bool(audio_boundaries.get("snap_word_gaps", True)),
        )
        segmentation.setdefault(
            "word_split_snap_search_seconds",
            float(audio_boundaries.get("snap_search_seconds", 0.20)),
        )
        segmentation.setdefault(
            "word_split_snap_window_seconds",
            float(audio_boundaries.get("snap_window_seconds", 0.02)),
        )
        segmentation.setdefault(
            "word_split_snap_max_rms_dbfs",
            float(audio_boundaries.get("snap_maximum_rms_dbfs", -42.0)),
        )
        segmentation.setdefault(
            "voice_boundary_vad_threshold",
            float(audio_boundaries.get("vad_threshold", 0.50)),
        )
        segmentation.setdefault(
            "voice_boundary_breath_vad_threshold",
            float(audio_boundaries.get("breath_vad_threshold", 0.70)),
        )
    configured = migrated.get("alignment")
    if configured is not None:
        is_grouped_v1 = bool(
            version < 2
            and isinstance(configured, dict)
            and "span_search" in configured
        )
        if is_grouped_v1:
            span_search = configured.get("span_search") or {}
            recovery = configured.get("recovery") or {}
            # Version 1 accidentally changed both historical defaults from
            # eight to ten. Restore values written by those incomplete
            # grouped defaults.
            if span_search.get("max_segments") == 10:
                span_search["max_segments"] = 8
            if recovery.get("max_segments") == 10:
                recovery["max_segments"] = 8
        migrated["alignment"] = complete_alignment_config(configured)
    migrated["settings_version"] = PROJECT_SETTINGS_VERSION
    return migrated


def repair_review_session_mappings(
    *,
    project_dir: Path,
    project: dict[str, Any],
) -> int:
    """Repair only mappings explicitly marked as inferred and uncertain."""

    if not project.get("source_lines") or not project.get("sessions"):
        return 0
    source_path = resolve_project_path(project_dir, project["source_lines"])
    if not source_path.is_file():
        return 0
    source_data = load_source_data(project_dir, project)
    audio_files = [
        resolve_project_path(project_dir, session["audio"])
        for session in project.get("sessions", [])
    ]
    inferred_by_id = {
        session["id"]: session
        for session in infer_sessions(audio_files, source_data, project_dir)
    }
    repaired = 0
    for session in project.get("sessions", []):
        if not session.get("needs_mapping_review"):
            continue
        inferred = inferred_by_id.get(session["id"])
        if not inferred or inferred.get("needs_mapping_review"):
            continue
        session["sheets"] = list(inferred["sheets"])
        session["needs_mapping_review"] = False
        repaired += 1
    return repaired


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
    existing_project: dict[str, Any] | None = None
    if project_file.exists() and not force:
        raise FileExistsError(
            f"Project already exists: {project_file}. Use --force to regenerate it."
        )
    if project_file.exists():
        existing_project = migrate_project_config(read_json(project_file))
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
        "settings_version": PROJECT_SETTINGS_VERSION,
        "workbook": relpath_for_config(workbook_path, project_dir),
        "audio_dir": relpath_for_config(audio_dir, project_dir),
        "source_lines": "source_lines.json",
        "audio_inventory": "audio_inventory.json",
        **default_project_settings(),
        "sessions": infer_sessions(audio_files, source_data, project_dir),
    }
    if existing_project is not None:
        backup_path = project_dir / "project.before-force.json"
        if not backup_path.exists():
            write_json(backup_path, existing_project)
        apply_project_settings(
            project,
            editable_project_settings(existing_project),
        )
        existing_sessions = {
            session["id"]: session
            for session in existing_project.get("sessions", [])
        }
        for session in project["sessions"]:
            previous = existing_sessions.get(session["id"])
            if not previous:
                continue
            if (
                previous.get("needs_mapping_review")
                and not session.get("needs_mapping_review")
            ):
                session["enabled"] = previous.get("enabled", True)
                continue
            project_session_defaults = copy.deepcopy(session)
            project_session_defaults.update(copy.deepcopy(previous))
            project_session_defaults["audio"] = session["audio"]
            session.clear()
            session.update(project_session_defaults)
    if project_settings:
        _merge_project_settings(project, project_settings)
    write_json(project_file, project)
    return project


def load_project(project_file: Path) -> tuple[Path, dict[str, Any]]:
    project_file = project_file.resolve()
    project_dir = project_file.parent
    project = migrate_project_config(read_json(project_file))
    repair_review_session_mappings(project_dir=project_dir, project=project)
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
