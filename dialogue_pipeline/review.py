from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import is_vocalization_script, read_json, resolve_project_path, write_json


REVIEW_FILE_NAME = "line_review.json"
REVIEW_SCHEMA_VERSION = 1
LINE_TYPES = {"normal", "nonverbal"}
LINE_STATUSES = {"AUTO_OK", "REVIEW", "MISSING", "MANUALLY_REVIEWED"}


def _review_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": int(candidate.get("rank", 0)),
        "segment_id": str(candidate["segment_id"]),
        "segment_file": str(candidate["segment_file"]),
        "transcript": str(candidate.get("transcript") or ""),
        "score": float(candidate.get("match_score", 0.0)),
        "match_score": float(candidate.get("match_score", 0.0)),
        "selection_score": float(candidate.get("selection_score", 0.0)),
        "reliable": bool(candidate.get("reliable", False)),
        "reliability_reason": str(candidate.get("reliability_reason") or ""),
        "technical_score": float(candidate.get("technical_score", 0.0)),
        "confidence_margin": float(candidate.get("confidence_margin", 0.0)),
        "source_audio": str(candidate.get("source_audio") or ""),
        "start_seconds": float(candidate.get("start_seconds", 0.0)),
        "end_seconds": float(candidate.get("end_seconds", 0.0)),
        "duration_seconds": float(candidate.get("duration_seconds", 0.0)),
    }


def _unmatched_candidate(segment: dict[str, Any]) -> dict[str, Any]:
    technical_score = segment.get("technical_score")
    if technical_score is None:
        technical_score = 100.0 * float(segment.get("asr_confidence") or 0.0)
    return {
        "segment_id": str(segment["segment_id"]),
        "segment_file": str(segment["segment_file"]),
        "transcript": str(segment.get("transcript") or ""),
        "score": float(technical_score),
        "technical_score": float(technical_score),
        "asr_confidence": segment.get("asr_confidence"),
        "reason": str(segment.get("reason") or ""),
        "technical_flags": str(segment.get("technical_flags") or ""),
        "source_audio": str(segment.get("source_wav") or ""),
        "start_seconds": float(segment.get("start_seconds", 0.0)),
        "end_seconds": float(segment.get("end_seconds", 0.0)),
        "duration_seconds": float(segment.get("duration_seconds", 0.0)),
    }


def build_line_review(
    *,
    source_lines: list[dict[str, Any]],
    candidates_by_line: dict[str, list[dict[str, Any]]],
    unmatched_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    review_lines = []
    for source_line in source_lines:
        line_type = (
            "nonverbal"
            if is_vocalization_script(str(source_line["line"]))
            else "normal"
        )
        candidates = sorted(
            candidates_by_line.get(str(source_line["line_id"]), []),
            key=lambda candidate: float(candidate.get("selection_score", 0.0)),
            reverse=True,
        )
        review_candidates = [_review_candidate(candidate) for candidate in candidates]
        reliable_best = next(
            (candidate for candidate in review_candidates if candidate["reliable"]),
            None,
        )

        if line_type == "nonverbal":
            status = "REVIEW"
            selected_segment_id = None
            suggested_segment_id = None
            review_candidates = []
        else:
            status = (
                "AUTO_OK"
                if reliable_best
                else ("REVIEW" if review_candidates else "MISSING")
            )
            selected_segment_id = (
                reliable_best["segment_id"] if reliable_best else None
            )
            suggested_segment_id = (
                review_candidates[0]["segment_id"] if review_candidates else None
            )

        review_lines.append(
            {
                "line_id": str(source_line["line_id"]),
                "sheet": str(source_line["sheet"]),
                "excel_row": int(source_line["excel_row"]),
                "line_text": str(source_line["line"]),
                "target_filename": str(source_line["target_filename"]),
                "type": line_type,
                "status": status,
                "suggested_segment_id": suggested_segment_id,
                "selected_segment_id": selected_segment_id,
                "candidates": review_candidates,
            }
        )

    audible_unmatched = [
        _unmatched_candidate(segment)
        for segment in unmatched_segments
        if bool(
            segment.get(
                "audible",
                "VERY_QUIET"
                not in str(segment.get("technical_flags") or "").split(","),
            )
        )
    ]
    audible_unmatched.sort(
        key=lambda segment: (
            -float(segment.get("score", 0.0)),
            str(segment["segment_id"]),
        )
    )
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "lines": review_lines,
        "unmatched_segments": audible_unmatched,
    }


def validate_line_review(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Review data must be a JSON object")
    if data.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported line_review.json schema version: "
            f"{data.get('schema_version')!r}"
        )
    if not isinstance(data.get("lines"), list):
        raise ValueError("Review data has no lines list")
    if not isinstance(data.get("unmatched_segments"), list):
        raise ValueError("Review data has no unmatched_segments list")

    seen_line_ids: set[str] = set()
    for index, line in enumerate(data["lines"], start=1):
        if not isinstance(line, dict):
            raise ValueError(f"Review line {index} must be an object")
        missing = {
            key
            for key in (
                "line_id",
                "sheet",
                "line_text",
                "target_filename",
                "type",
                "status",
                "selected_segment_id",
                "candidates",
            )
            if key not in line
        }
        if missing:
            raise ValueError(
                f"Review line {index} is missing: {', '.join(sorted(missing))}"
            )
        line_id = str(line["line_id"])
        if not line_id or line_id in seen_line_ids:
            raise ValueError(f"Invalid or duplicate review line ID: {line_id!r}")
        seen_line_ids.add(line_id)
        if line["type"] not in LINE_TYPES:
            raise ValueError(
                f"Invalid type for {line_id}: {line['type']!r}"
            )
        if line["status"] not in LINE_STATUSES:
            raise ValueError(
                f"Invalid status for {line_id}: {line['status']!r}"
            )
        if not isinstance(line["candidates"], list):
            raise ValueError(f"Candidates for {line_id} must be a list")
    return data


def load_line_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return validate_line_review(read_json(path))


def save_line_review(path: Path, data: dict[str, Any]) -> None:
    validate_line_review(data)
    write_json(path, data)


def segment_file_for_id(
    *,
    project_dir: Path,
    review_data: dict[str, Any],
    segment_id: str,
) -> Path:
    for line in review_data["lines"]:
        for candidate in line["candidates"]:
            if candidate["segment_id"] == segment_id:
                return resolve_project_path(project_dir, candidate["segment_file"])
    for segment in review_data["unmatched_segments"]:
        if segment["segment_id"] == segment_id:
            return resolve_project_path(project_dir, segment["segment_file"])
    raise KeyError(f"Segment is not present in review data: {segment_id}")
