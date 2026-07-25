from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.chart import PieChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.series import DataPoint
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from rapidfuzz import fuzz

from .project import load_source_data
from .segmentation import materialize_derived_segment
from .util import (
    is_nonverbal_script,
    normalize_text,
    read_json,
    resolve_project_path,
    word_count,
    write_json,
)
from .workbook_io import lines_for_session


@dataclass
class TraceNode:
    previous: "TraceNode | None"
    action: dict[str, Any] | None


@dataclass
class StateRecord:
    score: float
    trace: TraceNode


def text_similarity(expected: str, observed: str) -> float:
    expected_normalized = normalize_text(expected)
    observed_normalized = normalize_text(observed)
    if not expected_normalized or not observed_normalized:
        return 0.0
    expected_tokens = expected_normalized.split()
    observed_tokens = observed_normalized.split()
    expected_counts = Counter(expected_tokens)
    observed_counts = Counter(observed_tokens)
    covered = sum(
        min(count, observed_counts.get(token, 0))
        for token, count in expected_counts.items()
    )
    coverage = 100.0 * covered / max(1, len(expected_tokens))
    ratio = fuzz.ratio(expected_normalized, observed_normalized)
    weighted = fuzz.WRatio(expected_normalized, observed_normalized)
    partial = fuzz.partial_ratio(expected_normalized, observed_normalized)

    if len(expected_tokens) <= 3:
        score = 0.55 * ratio + 0.25 * weighted + 0.20 * coverage
    else:
        score = 0.40 * ratio + 0.30 * weighted + 0.15 * partial + 0.15 * coverage

    length_ratio = len(observed_tokens) / max(1, len(expected_tokens))
    if length_ratio < 0.45:
        score *= 0.72
    elif length_ratio < 0.70:
        score *= 0.88
    if length_ratio > 2.2:
        score *= 0.85
    return max(0.0, min(100.0, score))


def _technical_score(segment: dict[str, Any]) -> float:
    metrics = segment.get("metrics") or {}
    score = 100.0
    if int(metrics.get("clipping_samples") or 0) > 0:
        score -= 35.0
    peak = metrics.get("peak_dbfs")
    if peak is None or not math.isfinite(float(peak)):
        score -= 40.0
    elif float(peak) > -0.05:
        score -= 10.0
    rms = metrics.get("rms_dbfs")
    if rms is None or not math.isfinite(float(rms)):
        score -= 30.0
    elif float(rms) < -45.0:
        score -= 20.0
    probability = segment.get("asr_probability")
    if probability is not None:
        score += max(-20.0, min(0.0, (float(probability) - 0.75) * 40.0))
    return max(0.0, min(100.0, score))


def _valid_span(
    segments: list[dict[str, Any]],
    start_index: int,
    count: int,
    *,
    max_merge_gap_seconds: float,
    max_span_seconds: float,
) -> bool:
    selected = segments[start_index : start_index + count]
    if len(selected) != count:
        return False
    if selected[-1]["end_seconds"] - selected[0]["start_seconds"] > max_span_seconds:
        return False
    for left, right in zip(selected, selected[1:]):
        gap = max(0.0, right["start_seconds"] - left["end_seconds"])
        if gap > max_merge_gap_seconds:
            return False
    return True


def _span_preview(
    segments: list[dict[str, Any]], start_index: int, count: int
) -> dict[str, Any]:
    selected = segments[start_index : start_index + count]
    probabilities = [
        segment["asr_probability"]
        for segment in selected
        if segment.get("asr_probability") is not None
    ]
    return {
        "start_index": start_index,
        "count": count,
        "transcript": " ".join(
            segment.get("transcript", "").strip()
            for segment in selected
            if segment.get("transcript", "").strip()
        ),
        "start_seconds": selected[0]["start_seconds"],
        "end_seconds": selected[-1]["end_seconds"],
        "asr_probability": (
            sum(probabilities) / len(probabilities) if probabilities else None
        ),
    }


def _update_state(
    states: list[dict[int, StateRecord]],
    segment_index: int,
    line_index: int,
    candidate: StateRecord,
) -> None:
    existing = states[segment_index].get(line_index)
    if existing is None or candidate.score > existing.score:
        states[segment_index][line_index] = candidate


def sequence_align(
    segments: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    segment_count = len(segments)
    line_count = len(lines)
    states: list[dict[int, StateRecord]] = [
        {} for _ in range(segment_count + 1)
    ]
    root = TraceNode(previous=None, action=None)
    states[0][0] = StateRecord(score=0.0, trace=root)

    max_merge = int(settings.get("max_merge_segments", 3))
    max_gap = float(settings.get("max_merge_gap_seconds", 2.5))
    max_span = float(settings.get("max_span_seconds", 35.0))
    lookahead = int(settings.get("lookahead_lines", 8))
    minimum_path_score = float(settings.get("path_min_score", 35.0))
    noise_penalty = float(settings.get("noise_penalty", 2.2))
    skip_penalty = float(settings.get("skip_line_penalty", 1.8))
    repeat_penalty = float(settings.get("repeat_take_penalty", 0.8))

    for segment_index in range(segment_count):
        for next_line, state in list(states[segment_index].items()):
            noise_node = TraceNode(
                previous=state.trace,
                action={
                    "type": "unmatched",
                    "start_index": segment_index,
                    "count": 1,
                },
            )
            _update_state(
                states,
                segment_index + 1,
                next_line,
                StateRecord(state.score - noise_penalty, noise_node),
            )
            if next_line >= line_count:
                continue

            for count in range(1, max_merge + 1):
                if not _valid_span(
                    segments,
                    segment_index,
                    count,
                    max_merge_gap_seconds=max_gap,
                    max_span_seconds=max_span,
                ):
                    break
                span = _span_preview(segments, segment_index, count)
                last_target = min(line_count, next_line + lookahead + 1)
                for target_line in range(next_line, last_target):
                    match_score = text_similarity(
                        lines[target_line]["line"], span["transcript"]
                    )
                    if match_score < minimum_path_score:
                        continue
                    contribution = (match_score - 50.0) / 5.0
                    base_score = (
                        state.score
                        + contribution
                        - (target_line - next_line) * skip_penalty
                    )
                    action = {
                        "type": "assigned",
                        "start_index": segment_index,
                        "count": count,
                        "line_index": target_line,
                        "match_score": match_score,
                        "transcript": span["transcript"],
                    }
                    advance_node = TraceNode(previous=state.trace, action=action)
                    _update_state(
                        states,
                        segment_index + count,
                        target_line + 1,
                        StateRecord(base_score, advance_node),
                    )
                    repeat_action = dict(action)
                    repeat_action["repeat_transition"] = True
                    repeat_node = TraceNode(
                        previous=state.trace, action=repeat_action
                    )
                    _update_state(
                        states,
                        segment_index + count,
                        target_line,
                        StateRecord(base_score - repeat_penalty, repeat_node),
                    )

    if not states[segment_count]:
        return []
    _, best_state = max(
        states[segment_count].items(),
        key=lambda item: item[1].score - (line_count - item[0]) * skip_penalty,
    )
    actions = []
    node: TraceNode | None = best_state.trace
    while node and node.action is not None:
        actions.append(node.action)
        node = node.previous
    actions.reverse()
    return [action for action in actions if action["type"] == "assigned"]


def _candidate_reliability(
    *,
    line: dict[str, Any],
    match_score: float,
    margin: float,
    settings: dict[str, Any],
) -> tuple[bool, str]:
    if is_nonverbal_script(line["line"]):
        return False, "NONVERBAL_SCRIPT"
    if word_count(line["line"]) <= 3:
        minimum_score = float(settings.get("short_line_min_score", 88.0))
        minimum_margin = float(settings.get("short_line_min_margin", 15.0))
        if match_score < minimum_score:
            return False, "SHORT_LINE_LOW_SCORE"
        if margin < minimum_margin:
            return False, "SHORT_LINE_AMBIGUOUS"
        return True, ""
    if match_score < float(settings.get("reliable_min_score", 72.0)):
        return False, "LOW_MATCH_SCORE"
    if margin < float(settings.get("reliable_min_margin", 8.0)):
        return False, "AMBIGUOUS_MATCH"
    return True, ""


def align_project(
    *,
    project_dir: Path,
    project: dict[str, Any],
    session_filter: set[str] | None = None,
) -> dict[str, Path]:
    manifest_path = project_dir / "segments_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing segment manifest: {manifest_path}. Run segment first."
        )
    manifest = read_json(manifest_path)
    source_data = load_source_data(project_dir, project)
    line_by_id = {line["line_id"]: line for line in source_data["lines"]}
    settings = dict(project.get("alignment") or {})
    manifest_session_by_id = {
        entry["session_id"]: entry for entry in manifest["sessions"]
    }

    candidates: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    alignment_sessions = []

    selected_sessions = [
        item
        for item in project["sessions"]
        if item.get("enabled", True)
        and (not session_filter or item["id"] in session_filter)
    ]
    if not selected_sessions:
        raise ValueError("No enabled sessions matched the requested filter.")
    for session_index, session in enumerate(selected_sessions, start=1):
        session_entry = manifest_session_by_id.get(session["id"])
        if not session_entry:
            raise KeyError(f"Segments missing for session: {session['id']}")
        base_segments = session_entry.get("segments", [])
        session_lines = lines_for_session(source_data, session)
        if not session_lines:
            continue
        print(
            f"[align {session_index}] {session['id']}: "
            f"{len(base_segments)} segments → {len(session_lines)} lines",
            flush=True,
        )
        actions = sequence_align(base_segments, session_lines, settings)
        reliable_coverage: set[int] = set()
        assignment_by_base: dict[int, list[dict[str, Any]]] = defaultdict(list)
        serialized_actions = []

        for action in actions:
            line = session_lines[action["line_index"]]
            segment = materialize_derived_segment(
                project_dir=project_dir,
                project=project,
                session_entry=session_entry,
                base_segments=base_segments,
                start_index=action["start_index"],
                count=action["count"],
            )
            all_scores = sorted(
                (
                    text_similarity(other_line["line"], segment["transcript"]),
                    other_line["line_id"],
                )
                for other_line in session_lines
                if other_line["line_id"] != line["line_id"]
            )
            second_score = all_scores[-1][0] if all_scores else 0.0
            margin = action["match_score"] - second_score
            reliable, reason = _candidate_reliability(
                line=line,
                match_score=action["match_score"],
                margin=margin,
                settings=settings,
            )
            technical_score = _technical_score(segment)
            selection_score = (
                action["match_score"]
                + 0.10 * technical_score
                + 5.0 * float(segment.get("asr_probability") or 0.0)
            )
            candidate = {
                "line_id": line["line_id"],
                "session_id": session["id"],
                "segment_id": segment["segment_id"],
                "segment_file": segment["file"],
                "transcript": segment.get("transcript", ""),
                "match_score": action["match_score"],
                "confidence_margin": margin,
                "technical_score": technical_score,
                "selection_score": selection_score,
                "reliable": reliable,
                "reliability_reason": reason,
                "source_audio": segment["source_audio"],
                "start_seconds": segment["start_seconds"],
                "end_seconds": segment["end_seconds"],
                "duration_seconds": segment["metrics"]["duration_seconds"],
                "asr_probability": segment.get("asr_probability"),
                "base_indices": segment["base_indices"],
            }
            if action["match_score"] >= float(
                settings.get("candidate_min_score", 45.0)
            ):
                candidates.append(candidate)
            for base_index in segment["base_indices"]:
                assignment_by_base[base_index].append(candidate)
                if reliable:
                    reliable_coverage.add(base_index)
            serialized_actions.append(candidate)

        for base_index, segment in enumerate(base_segments):
            if base_index in reliable_coverage:
                continue
            transcript = segment.get("transcript", "")
            suggestions = sorted(
                (
                    text_similarity(line["line"], transcript),
                    line["line_id"],
                    line["line"],
                )
                for line in session_lines
            )
            best = suggestions[-1] if suggestions else (0.0, "", "")
            second = suggestions[-2] if len(suggestions) > 1 else (0.0, "", "")
            related = assignment_by_base.get(base_index, [])
            if not transcript.strip():
                reason = "TRANSCRIPTION_FAILED_OR_NONVERBAL"
            elif related:
                reason = "AMBIGUOUS_OR_LOW_CONFIDENCE"
            else:
                reason = "NO_RELIABLE_MATCH"
            unmatched_rows.append(
                {
                    "segment_id": segment["segment_id"],
                    "segment_file": segment["file"],
                    "source_wav": segment["source_audio"],
                    "start_seconds": segment["start_seconds"],
                    "end_seconds": segment["end_seconds"],
                    "duration_seconds": segment["metrics"]["duration_seconds"],
                    "transcript": transcript,
                    "asr_confidence": segment.get("asr_probability"),
                    "reason": reason,
                    "suggested_line_1": best[1],
                    "suggested_line_1_text": best[2],
                    "suggested_line_1_score": best[0],
                    "suggested_line_2": second[1],
                    "suggested_line_2_text": second[2],
                    "suggested_line_2_score": second[0],
                    "technical_flags": _technical_flags(segment),
                }
            )

        alignment_sessions.append(
            {
                "session_id": session["id"],
                "script_line_count": len(session_lines),
                "base_segment_count": len(base_segments),
                "assignments": serialized_actions,
                "reliable_base_segment_count": len(reliable_coverage),
            }
        )

    by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_line[candidate["line_id"]].append(candidate)
    for line_candidates in by_line.values():
        line_candidates.sort(
            key=lambda candidate: candidate["selection_score"], reverse=True
        )
        for rank, candidate in enumerate(line_candidates, start=1):
            candidate["rank"] = rank

    alignment_payload = {
        "schema_version": 1,
        "sessions": alignment_sessions,
        "candidate_count": len(candidates),
        "unmatched_count": len(unmatched_rows),
        "unmatched_segments": unmatched_rows,
        "candidates": sorted(
            candidates,
            key=lambda candidate: (
                line_by_id[candidate["line_id"]]["sheet_index"],
                line_by_id[candidate["line_id"]]["excel_row"],
                candidate["rank"],
            ),
        ),
    }
    alignment_path = project_dir / "alignment.json"
    write_json(alignment_path, alignment_payload)
    write_json(manifest_path, manifest)

    review_path = project_dir / "A_line_review.xlsx"
    _write_review_workbook(
        review_path=review_path,
        project_dir=project_dir,
        source_lines=source_data["lines"],
        candidates_by_line=by_line,
        unmatched_rows=unmatched_rows,
    )
    (project_dir / "B_unmatched_segments.tsv").unlink(missing_ok=True)
    return {
        "review": review_path,
        "alignment": alignment_path,
        "manifest": manifest_path,
    }


def _technical_flags(segment: dict[str, Any]) -> str:
    flags = []
    metrics = segment.get("metrics") or {}
    if int(metrics.get("clipping_samples") or 0) > 0:
        flags.append("CLIPPING")
    if float(metrics.get("rms_dbfs") or -999.0) < -45.0:
        flags.append("VERY_QUIET")
    if not segment.get("transcript", "").strip():
        flags.append("NO_TRANSCRIPT")
    return ",".join(flags)


def _write_review_workbook(
    *,
    review_path: Path,
    project_dir: Path,
    source_lines: list[dict[str, Any]],
    candidates_by_line: dict[str, list[dict[str, Any]]],
    unmatched_rows: list[dict[str, Any]],
) -> None:
    workbook = Workbook()
    lines_sheet = workbook.active
    lines_sheet.title = "Lines"
    candidates_sheet = workbook.create_sheet("Candidates")
    unmatched_sheet = workbook.create_sheet("Unmatched Segments")
    instructions_sheet = workbook.create_sheet("Instructions")

    line_headers = [
        "Line ID",
        "Sheet",
        "Excel Row",
        "Quest",
        "Context",
        "Line to Speak",
        "Acting Note",
        "Facial Emotion",
        "Target Filename",
        "Candidate Count",
        "Candidate 1",
        "Candidate 2",
        "Candidate 3",
        "Suggested Best Segment",
        "Selected Segment",
        "Status",
        "Candidate Summary",
        "User Notes",
    ]
    lines_sheet.append(line_headers)
    for line in source_lines:
        line_candidates = candidates_by_line.get(line["line_id"], [])
        best = line_candidates[0] if line_candidates else None
        reliable_best = next(
            (candidate for candidate in line_candidates if candidate["reliable"]),
            None,
        )
        status = (
            "AUTO_OK"
            if reliable_best
            else ("REVIEW" if line_candidates else "MISSING")
        )
        suggested = best["segment_id"] if best else ""
        selected = reliable_best["segment_id"] if reliable_best else ""
        top_candidates = line_candidates[:3]
        summary = "\n".join(
            (
                f"#{candidate['rank']} {candidate['segment_id']} | "
                f"match={candidate['match_score']:.1f} | "
                f"margin={candidate['confidence_margin']:.1f} | "
                f"{candidate['transcript']}"
            )
            for candidate in line_candidates
        )
        lines_sheet.append(
            [
                line["line_id"],
                line["sheet"],
                line["excel_row"],
                line["quest"],
                line["context"],
                line["line"],
                line["acting_note"],
                line["emotion"],
                line["target_filename"],
                len(line_candidates),
                *[
                    (
                        top_candidates[index]["segment_id"]
                        if index < len(top_candidates)
                        else ""
                    )
                    for index in range(3)
                ],
                suggested,
                selected,
                status,
                summary,
                "",
            ]
        )
        output_row = lines_sheet.max_row
        link_targets = [
            (11, top_candidates[0] if len(top_candidates) > 0 else None),
            (12, top_candidates[1] if len(top_candidates) > 1 else None),
            (13, top_candidates[2] if len(top_candidates) > 2 else None),
            (14, best),
            (15, reliable_best),
        ]
        for column, candidate in link_targets:
            if candidate:
                _set_segment_hyperlink(
                    lines_sheet.cell(output_row, column),
                    project_dir,
                    candidate["segment_file"],
                )

    candidate_headers = [
        "Line ID",
        "Rank",
        "Segment ID",
        "Segment File",
        "Transcript",
        "Match Score",
        "Confidence Margin",
        "Technical Score",
        "Reliable",
        "Reliability Reason",
        "Source WAV",
        "Start Seconds",
        "End Seconds",
        "Duration Seconds",
        "ASR Confidence",
    ]
    candidates_sheet.append(candidate_headers)
    line_order = {
        line["line_id"]: index for index, line in enumerate(source_lines)
    }
    all_candidates = sorted(
        (
            candidate
            for line_candidates in candidates_by_line.values()
            for candidate in line_candidates
        ),
        key=lambda candidate: (
            line_order.get(candidate["line_id"], 10**9),
            candidate["rank"],
        ),
    )
    for candidate in all_candidates:
        candidates_sheet.append(
            [
                candidate["line_id"],
                candidate["rank"],
                candidate["segment_id"],
                candidate["segment_file"],
                candidate["transcript"],
                candidate["match_score"],
                candidate["confidence_margin"],
                candidate["technical_score"],
                candidate["reliable"],
                candidate["reliability_reason"],
                candidate["source_audio"],
                candidate["start_seconds"],
                candidate["end_seconds"],
                candidate["duration_seconds"],
                candidate["asr_probability"],
            ]
        )
        file_cell = candidates_sheet.cell(candidates_sheet.max_row, 4)
        _set_segment_hyperlink(
            file_cell,
            project_dir,
            candidate["segment_file"],
        )

    unmatched_headers = [
        "Segment ID",
        "Segment File",
        "Source WAV",
        "Start Seconds",
        "End Seconds",
        "Duration Seconds",
        "Transcript",
        "ASR Confidence",
        "Reason",
        "Suggested Line 1",
        "Suggested Line 1 Text",
        "Suggested Line 1 Score",
        "Suggested Line 2",
        "Suggested Line 2 Text",
        "Suggested Line 2 Score",
        "Technical Flags",
    ]
    unmatched_sheet.append(unmatched_headers)
    for unmatched in unmatched_rows:
        unmatched_sheet.append(
            [
                unmatched["segment_id"],
                unmatched["segment_file"],
                unmatched["source_wav"],
                unmatched["start_seconds"],
                unmatched["end_seconds"],
                unmatched["duration_seconds"],
                unmatched["transcript"],
                unmatched["asr_confidence"],
                unmatched["reason"],
                unmatched["suggested_line_1"],
                unmatched["suggested_line_1_text"],
                unmatched["suggested_line_1_score"],
                unmatched["suggested_line_2"],
                unmatched["suggested_line_2_text"],
                unmatched["suggested_line_2_score"],
                unmatched["technical_flags"],
            ]
        )
        row = unmatched_sheet.max_row
        _set_segment_hyperlink(
            unmatched_sheet.cell(row, 1),
            project_dir,
            unmatched["segment_file"],
        )
        _set_segment_hyperlink(
            unmatched_sheet.cell(row, 2),
            project_dir,
            unmatched["segment_file"],
        )

    instructions = [
        ("Purpose", "Choose one temporary segment for every line that should be exported."),
        (
            "Editable field",
            "Click a linked candidate to audition it. Copy the whole Candidate 1, "
            "Candidate 2, or Candidate 3 cell into Lines!Selected Segment so both "
            "the Segment ID and its file link follow the selection. Segment IDs "
            "from the Candidates or Unmatched Segments sheet are also accepted.",
        ),
        (
            "Unmatched audio",
            "The Unmatched Segments sheet contains audio that could not be mapped "
            "reliably. Its Segment ID and Segment File cells are linked to the WAV.",
        ),
        (
            "Status",
            "AUTO_OK was filled automatically; REVIEW and MISSING require attention. "
            "Set Status to SKIP to intentionally omit a line.",
        ),
        (
            "Finalization",
            "The finalizer reads Selected Segment only. Suggested Best is informational.",
        ),
    ]
    instructions_sheet.append(["Topic", "Instruction"])
    for row in instructions:
        instructions_sheet.append(list(row))

    _style_workbook(
        lines_sheet,
        candidates_sheet,
        unmatched_sheet,
        instructions_sheet,
    )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(review_path)


def _set_segment_hyperlink(cell, project_dir: Path, segment_file: str) -> None:
    segment_path = resolve_project_path(project_dir, segment_file)
    try:
        cell.hyperlink = segment_path.as_uri()
        cell.style = "Hyperlink"
    except ValueError:
        pass


def _style_workbook(
    lines_sheet,
    candidates_sheet,
    unmatched_sheet,
    instructions_sheet,
) -> None:
    navy = "1F4E78"
    pale_blue = "D9EAF7"
    yellow = "FFF2CC"
    green = "E2F0D9"
    red = "FCE4D6"
    gray = "E7E6E6"
    white = "FFFFFF"
    thin_gray = Side(style="thin", color="D9E2F3")
    header_fill = PatternFill("solid", fgColor=navy)
    header_font = Font(color=white, bold=True)

    for sheet in (
        lines_sheet,
        candidates_sheet,
        unmatched_sheet,
        instructions_sheet,
    ):
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.sheet_view.showGridLines = False
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = Border(bottom=thin_gray)
        sheet.row_dimensions[1].height = 28

    selected_column = 15
    status_column = 16
    candidate_summary_column = 17
    notes_column = 18
    lines_data_last_row = lines_sheet.max_row
    for row in range(2, lines_sheet.max_row + 1):
        lines_sheet.cell(row, selected_column).fill = PatternFill(
            "solid", fgColor=yellow
        )
        lines_sheet.cell(row, notes_column).fill = PatternFill(
            "solid", fgColor=yellow
        )
        lines_sheet.cell(row, candidate_summary_column).alignment = Alignment(
            wrap_text=True, vertical="top"
        )
        for column in (4, 5, 6, 7, 8, 11, 12, 13, 14, 15):
            lines_sheet.cell(row, column).alignment = Alignment(
                wrap_text=True, vertical="top"
            )
    lines_sheet.conditional_formatting.add(
        f"P2:P{lines_data_last_row}",
        FormulaRule(formula=['P2="AUTO_OK"'], fill=PatternFill("solid", fgColor=green)),
    )
    lines_sheet.conditional_formatting.add(
        f"P2:P{lines_data_last_row}",
        FormulaRule(formula=['P2="REVIEW"'], fill=PatternFill("solid", fgColor=yellow)),
    )
    lines_sheet.conditional_formatting.add(
        f"P2:P{lines_data_last_row}",
        FormulaRule(formula=['P2="MISSING"'], fill=PatternFill("solid", fgColor=red)),
    )
    lines_sheet.conditional_formatting.add(
        f"P2:P{lines_data_last_row}",
        FormulaRule(formula=['P2="SKIP"'], fill=PatternFill("solid", fgColor=gray)),
    )
    lines_widths = {
        1: 34,
        2: 24,
        3: 10,
        4: 25,
        5: 55,
        6: 55,
        7: 34,
        8: 18,
        9: 42,
        10: 14,
        11: 42,
        12: 42,
        13: 42,
        14: 42,
        15: 42,
        status_column: 12,
        candidate_summary_column: 85,
        notes_column: 35,
    }
    for column, width in lines_widths.items():
        lines_sheet.column_dimensions[get_column_letter(column)].width = width

    for row in range(2, candidates_sheet.max_row + 1):
        for column in (3, 4, 5, 10, 11):
            candidates_sheet.cell(row, column).alignment = Alignment(
                wrap_text=True, vertical="top"
            )
        candidates_sheet.row_dimensions[row].height = 54
        for column in range(6, 9):
            candidates_sheet.cell(row, column).number_format = "0.0"
        for column in range(12, 16):
            candidates_sheet.cell(row, column).number_format = "0.000"
    candidate_widths = [34, 8, 38, 62, 70, 12, 16, 14, 10, 26, 58, 14, 14, 14, 14]
    for column, width in enumerate(candidate_widths, start=1):
        candidates_sheet.column_dimensions[get_column_letter(column)].width = width

    for row in range(2, unmatched_sheet.max_row + 1):
        for column in (1, 2, 3, 7, 9, 10, 11, 13, 14, 16):
            unmatched_sheet.cell(row, column).alignment = Alignment(
                wrap_text=True, vertical="top"
            )
        unmatched_sheet.row_dimensions[row].height = 54
        for column in range(4, 9):
            unmatched_sheet.cell(row, column).number_format = "0.000"
        for column in (12, 15):
            unmatched_sheet.cell(row, column).number_format = "0.0"
    unmatched_widths = [
        42,
        62,
        58,
        14,
        14,
        16,
        70,
        16,
        26,
        34,
        65,
        18,
        34,
        65,
        18,
        24,
    ]
    for column, width in enumerate(unmatched_widths, start=1):
        unmatched_sheet.column_dimensions[get_column_letter(column)].width = width

    instructions_sheet.column_dimensions["A"].width = 22
    instructions_sheet.column_dimensions["B"].width = 100
    for row in range(2, instructions_sheet.max_row + 1):
        instructions_sheet.cell(row, 1).fill = PatternFill(
            "solid", fgColor=pale_blue
        )
        instructions_sheet.cell(row, 1).font = Font(bold=True)
        instructions_sheet.cell(row, 2).alignment = Alignment(
            wrap_text=True, vertical="top"
        )

    _add_status_summary_and_chart(
        lines_sheet,
        data_last_row=lines_data_last_row,
        status_column=status_column,
        header_fill=header_fill,
        header_font=header_font,
        status_fills={
            "AUTO_OK": green,
            "REVIEW": yellow,
            "MISSING": red,
            "SKIP": gray,
        },
        border=thin_gray,
    )


def _add_status_summary_and_chart(
    lines_sheet,
    *,
    data_last_row: int,
    status_column: int,
    header_fill: PatternFill,
    header_font: Font,
    status_fills: dict[str, str],
    border: Side,
) -> None:
    title_row = data_last_row + 3
    header_row = title_row + 1
    first_status_row = header_row + 1
    status_column_letter = get_column_letter(status_column)

    lines_sheet.merge_cells(
        start_row=title_row,
        start_column=1,
        end_row=title_row,
        end_column=2,
    )
    title_cell = lines_sheet.cell(title_row, 1, "Line Status Summary")
    title_cell.fill = header_fill
    title_cell.font = header_font
    title_cell.alignment = Alignment(horizontal="center", vertical="center")
    lines_sheet.row_dimensions[title_row].height = 24

    lines_sheet.cell(header_row, 1, "Status")
    lines_sheet.cell(header_row, 2, "Line Count")
    for cell in lines_sheet[header_row][0:2]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(bottom=border)

    statuses = ["AUTO_OK", "REVIEW", "MISSING", "SKIP"]
    for index, status in enumerate(statuses):
        row = first_status_row + index
        lines_sheet.cell(row, 1, status)
        lines_sheet.cell(
            row,
            2,
            (
                f'=COUNTIF(${status_column_letter}$2:'
                f'${status_column_letter}${data_last_row},A{row})'
            ),
        )
        lines_sheet.cell(row, 1).fill = PatternFill(
            "solid", fgColor=status_fills[status]
        )
        lines_sheet.cell(row, 2).number_format = "#,##0"

    chart = PieChart()
    chart.title = "Review Status Distribution"
    chart.height = 7.0
    chart.width = 11.5
    chart.legend.position = "r"
    data = Reference(
        lines_sheet,
        min_col=2,
        min_row=header_row,
        max_row=first_status_row + len(statuses) - 1,
    )
    labels = Reference(
        lines_sheet,
        min_col=1,
        min_row=first_status_row,
        max_row=first_status_row + len(statuses) - 1,
    )
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(labels)
    chart.dataLabels = DataLabelList()
    chart.dataLabels.showPercent = True
    chart.dataLabels.showVal = True
    chart.series[0].data_points = [
        DataPoint(
            idx=index,
            spPr=GraphicalProperties(solidFill=status_fills[status]),
        )
        for index, status in enumerate(statuses)
    ]
    lines_sheet.add_chart(chart, f"D{title_row}")
