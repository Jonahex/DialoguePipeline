from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .util import is_vocalization_script, read_json, resolve_project_path, write_json


REVIEW_FILE_NAME = "line_review.json"
REVIEW_SCHEMA_VERSION = 1
LINE_TYPES = {"normal", "nonverbal"}
LINE_STATUSES = {"AUTO_OK", "REVIEW", "MISSING", "MANUALLY_REVIEWED"}
REVIEW_CANDIDATE_SCORE_GAP = 12.0
REVIEW_CANDIDATE_MAX_SCORE_DROP = 15.0
STRUCTURALLY_INCOMPLETE_REASONS = {
    "LOW_MATCH_SCORE",
    "MISSING_SENTENCE",
    "SEGMENT_BETTER_MATCH_ELSEWHERE",
    "SENTENCE_ORDER_MISMATCH",
    "SHORT_LINE_LOW_SCORE",
}


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float]:
    return (
        float(candidate.get("match_score", 0.0)),
        float(candidate.get("selection_score", 0.0)),
    )


def _is_dominated_span(
    candidate: dict[str, Any],
    better_candidates: list[dict[str, Any]],
) -> bool:
    base_indices = set(candidate.get("base_indices") or [])
    if not base_indices:
        return False
    for better in better_candidates:
        if candidate.get("session_id") != better.get("session_id"):
            continue
        better_indices = set(better.get("base_indices") or [])
        if (
            base_indices < better_indices
            and float(better.get("match_score", 0.0))
            >= float(candidate.get("match_score", 0.0))
            and (
                bool(better.get("reliable", False))
                or not int(better.get("missing_clause_count", 0))
            )
        ):
            return True
    return False


def prune_line_candidates(
    candidates: list[dict[str, Any]],
    *,
    score_gap: float = REVIEW_CANDIDATE_SCORE_GAP,
    max_score_drop: float = REVIEW_CANDIDATE_MAX_SCORE_DROP,
) -> list[dict[str, Any]]:
    primary = sorted(
        (
            candidate
            for candidate in candidates
            if bool(candidate.get("is_primary_match", True))
        ),
        key=_candidate_sort_key,
        reverse=True,
    )
    unique = []
    seen_segment_ids: set[str] = set()
    for candidate in primary:
        segment_id = str(candidate["segment_id"])
        if segment_id in seen_segment_ids:
            continue
        seen_segment_ids.add(segment_id)
        if not _is_dominated_span(candidate, unique):
            unique.append(candidate)
    if not unique:
        return []

    scores = [float(candidate.get("match_score", 0.0)) for candidate in unique]
    cutoff = scores[0] - max_score_drop
    gaps = [
        (scores[index] - scores[index + 1], index)
        for index in range(len(scores) - 1)
    ]
    if gaps:
        largest_gap, gap_index = max(gaps)
        if largest_gap >= score_gap:
            cutoff = max(
                cutoff,
                (scores[gap_index] + scores[gap_index + 1]) / 2.0,
            )

    retained = [
        candidate
        for candidate in unique
        if float(candidate.get("match_score", 0.0)) >= cutoff
    ]
    if not any(candidate.get("reliable", False) for candidate in retained):
        reliable_best = next(
            (candidate for candidate in unique if candidate.get("reliable", False)),
            None,
        )
        if reliable_best is not None:
            retained.append(reliable_best)

    if any(candidate.get("reliable", False) for candidate in retained):
        retained = [
            candidate
            for candidate in retained
            if (
                candidate.get("reliable", False)
                or str(candidate.get("reliability_reason") or "")
                not in STRUCTURALLY_INCOMPLETE_REASONS
            )
        ]
    return sorted(
        retained,
        key=_candidate_sort_key,
        reverse=True,
    )


def _review_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": int(candidate.get("rank", 0)),
        "segment_id": str(candidate["segment_id"]),
        "segment_file": str(candidate["segment_file"]),
        "session_id": str(candidate.get("session_id") or ""),
        "base_indices": [
            int(base_index) for base_index in candidate.get("base_indices") or []
        ],
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


def _base_segment_key(segment: dict[str, Any]) -> tuple[str, int] | None:
    if segment.get("session_id") is not None and segment.get("base_index") is not None:
        return str(segment["session_id"]), int(segment["base_index"])
    match = re.fullmatch(r"(.+)__s(\d+)", str(segment.get("segment_id") or ""))
    if not match:
        return None
    return match.group(1), int(match.group(2)) - 1


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
    retained_by_line = {
        str(source_line["line_id"]): (
            []
            if is_vocalization_script(str(source_line["line"]))
            else prune_line_candidates(
                candidates_by_line.get(str(source_line["line_id"]), [])
            )
        )
        for source_line in source_lines
    }
    verbal_candidate_segment_ids: set[str] = set()
    verbal_candidate_base_segments: set[tuple[str, int]] = set()
    for candidates in retained_by_line.values():
        for candidate in candidates:
            verbal_candidate_segment_ids.add(str(candidate["segment_id"]))
            session_id = str(candidate.get("session_id") or "")
            verbal_candidate_base_segments.update(
                (session_id, int(base_index))
                for base_index in candidate.get("base_indices") or []
            )

    review_lines = []
    for source_line in source_lines:
        line_type = (
            "nonverbal"
            if is_vocalization_script(str(source_line["line"]))
            else "normal"
        )
        candidates = retained_by_line[str(source_line["line_id"])]
        review_candidates = [_review_candidate(candidate) for candidate in candidates]
        for rank, candidate in enumerate(review_candidates, start=1):
            candidate["rank"] = rank
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
                "context": str(source_line.get("context") or ""),
                "line_text": str(source_line["line"]),
                "acting_note": str(source_line.get("acting_note") or ""),
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
        and str(segment["segment_id"]) not in verbal_candidate_segment_ids
        and _base_segment_key(segment) not in verbal_candidate_base_segments
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
        line.setdefault("context", "")
        line.setdefault("acting_note", "")
        missing = {
            key
            for key in (
                "line_id",
                "sheet",
                "context",
                "line_text",
                "acting_note",
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


def preserve_manual_selections(
    new_data: dict[str, Any],
    previous_data: dict[str, Any],
) -> dict[str, Any]:
    validate_line_review(new_data)
    validate_line_review(previous_data)
    previous_unmatched = {
        segment["segment_id"]: segment
        for segment in previous_data["unmatched_segments"]
    }
    new_unmatched_ids = {
        segment["segment_id"] for segment in new_data["unmatched_segments"]
    }
    new_lines = {line["line_id"]: line for line in new_data["lines"]}

    for previous_line in previous_data["lines"]:
        if previous_line["status"] != "MANUALLY_REVIEWED":
            continue
        selected_id = previous_line.get("selected_segment_id")
        new_line = new_lines.get(previous_line["line_id"])
        if not selected_id or new_line is None:
            continue
        new_line["selected_segment_id"] = selected_id
        new_line["status"] = "MANUALLY_REVIEWED"

        previous_candidates = {
            candidate["segment_id"]: candidate
            for candidate in previous_line["candidates"]
        }
        selected_candidate = previous_candidates.get(selected_id)
        if (
            new_line["type"] == "normal"
            and selected_candidate is not None
            and selected_id
            not in {
                candidate["segment_id"]
                for candidate in new_line["candidates"]
            }
        ):
            new_line["candidates"].append(selected_candidate)
            new_line["candidates"].sort(
                key=lambda candidate: float(candidate.get("score", 0.0)),
                reverse=True,
            )
            for rank, candidate in enumerate(new_line["candidates"], start=1):
                candidate["rank"] = rank
        if new_line["type"] == "normal" and selected_candidate is not None:
            selected_base_segments = {
                (
                    str(selected_candidate.get("session_id") or ""),
                    int(base_index),
                )
                for base_index in selected_candidate.get("base_indices") or []
            }
            new_data["unmatched_segments"] = [
                segment
                for segment in new_data["unmatched_segments"]
                if segment["segment_id"] != selected_id
                and _base_segment_key(segment) not in selected_base_segments
            ]
            new_unmatched_ids = {
                segment["segment_id"]
                for segment in new_data["unmatched_segments"]
            }
        if (
            (
                new_line["type"] == "nonverbal"
                or not new_line["candidates"]
            )
            and selected_id not in new_unmatched_ids
            and selected_id in previous_unmatched
        ):
            new_data["unmatched_segments"].append(previous_unmatched[selected_id])
            new_unmatched_ids.add(selected_id)
    return validate_line_review(new_data)


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
