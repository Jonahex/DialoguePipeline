from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from .project import load_source_data
from .review import (
    REVIEW_FILE_NAME,
    build_line_review,
    load_line_review,
    preserve_manual_selections,
)
from .segmentation import materialize_derived_segment
from .transcription import (
    transcribe_candidate_spans,
    transcribe_segments_project,
)
from .util import (
    is_nonverbal_script,
    is_vocalization_script,
    normalize_text,
    read_json,
    word_count,
    write_json,
)
from .workbook_io import lines_for_session


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
    token_sort = fuzz.token_sort_ratio(expected_normalized, observed_normalized)
    token_set = fuzz.token_set_ratio(expected_normalized, observed_normalized)
    compact_ratio = fuzz.ratio(
        expected_normalized.replace(" ", "").replace("'", ""),
        observed_normalized.replace(" ", "").replace("'", ""),
    )

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

    order_insensitive = max(
        0.50 * token_sort + 0.30 * token_set + 0.20 * compact_ratio,
        0.60 * token_set + 0.40 * compact_ratio,
    )
    if compact_ratio < 98.0 and length_ratio < 0.45:
        order_insensitive *= 0.80
    elif compact_ratio < 98.0 and length_ratio < 0.70:
        order_insensitive *= 0.97
    if length_ratio > 2.2:
        order_insensitive *= 0.92
    score = max(score, order_insensitive)
    return max(0.0, min(100.0, score))


def transcript_fidelity(
    expected: str,
    observed: str,
    *,
    token_min_similarity: float = 78.0,
) -> dict[str, float | int]:
    """Measure ordered agreement separately from tolerant candidate retrieval."""

    expected_normalized = normalize_text(expected)
    observed_normalized = normalize_text(observed)
    expected_tokens = expected_normalized.split()
    observed_tokens = observed_normalized.split()
    if not expected_tokens or not observed_tokens:
        return {
            "ordered_similarity": 0.0,
            "token_coverage": 0.0,
            "token_precision": 0.0,
            "extra_word_count": len(observed_tokens),
        }

    possible_matches = sorted(
        (
            (
                fuzz.ratio(expected_token, observed_token),
                expected_index,
                observed_index,
            )
            for expected_index, expected_token in enumerate(expected_tokens)
            for observed_index, observed_token in enumerate(observed_tokens)
        ),
        reverse=True,
    )
    matched_expected: set[int] = set()
    matched_observed: set[int] = set()
    for score, expected_index, observed_index in possible_matches:
        if score < token_min_similarity:
            break
        if (
            expected_index in matched_expected
            or observed_index in matched_observed
        ):
            continue
        matched_expected.add(expected_index)
        matched_observed.add(observed_index)

    matched_count = len(matched_expected)
    return {
        "ordered_similarity": fuzz.ratio(
            expected_normalized,
            observed_normalized,
        ),
        "token_coverage": matched_count / len(expected_tokens),
        "token_precision": matched_count / len(observed_tokens),
        "extra_word_count": len(observed_tokens) - matched_count,
    }


def script_clauses(value: str) -> list[str]:
    return [
        clause
        for clause in (part.strip() for part in re.split(r"[.!?]+", value or ""))
        if normalize_text(clause)
    ]


def sentence_fidelity(
    expected: str,
    observed: str,
    *,
    token_min_similarity: float = 78.0,
    minimum_clause_score: float = 55.0,
    short_clause_max_words: int = 4,
    short_clause_min_token_coverage: float = 1.0,
) -> dict[str, Any]:
    clauses = script_clauses(expected)
    observed_normalized = normalize_text(observed)
    clause_scores = []
    clause_token_coverages = []
    clause_word_counts = []
    clause_positions = []
    for clause in clauses:
        clause_fidelity = transcript_fidelity(
            clause,
            observed,
            token_min_similarity=token_min_similarity,
        )
        partial_score = (
            fuzz.partial_ratio(normalize_text(clause), observed_normalized)
            if observed_normalized
            else 0.0
        )
        clause_scores.append(
            0.50 * partial_score
            + 50.0 * float(clause_fidelity["token_coverage"])
        )
        clause_token_coverages.append(
            float(clause_fidelity["token_coverage"])
        )
        clause_word_counts.append(word_count(clause))
        alignment = (
            fuzz.partial_ratio_alignment(
                normalize_text(clause),
                observed_normalized,
            )
            if observed_normalized
            else None
        )
        clause_positions.append(
            int(alignment.dest_start) if alignment is not None else -1
        )
    minimum_score = min(clause_scores) if clause_scores else 0.0
    missing_clauses = [
        score < minimum_clause_score
        or (
            clause_words <= short_clause_max_words
            and token_coverage < short_clause_min_token_coverage
        )
        for score, token_coverage, clause_words in zip(
            clause_scores,
            clause_token_coverages,
            clause_word_counts,
        )
    ]
    return {
        "clause_count": len(clauses),
        "clause_scores": clause_scores,
        "clause_token_coverages": clause_token_coverages,
        "clause_word_counts": clause_word_counts,
        "clause_positions": clause_positions,
        "clauses_in_order": all(
            left <= right
            for left, right in zip(
                clause_positions,
                clause_positions[1:],
            )
        ),
        "minimum_clause_score": minimum_score,
        "missing_clause_count": sum(missing_clauses),
    }


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


def _span_preview_with_transcription(
    segments: list[dict[str, Any]],
    start_index: int,
    count: int,
    *,
    transcription: dict[str, Any] | None,
    settings: dict[str, Any],
    word_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    span = _span_preview(segments, start_index, count)
    if not transcription:
        return span

    minimum_overlap_seconds = float(
        settings.get("span_word_min_overlap_seconds", 0.20)
    )
    minimum_overlap_fraction = float(
        settings.get("span_word_min_overlap_fraction", 0.20)
    )
    start_seconds = float(span["start_seconds"])
    end_seconds = float(span["end_seconds"])
    words = []
    seen = set()
    index = word_index or _transcription_word_index(transcription)
    indexed_words = index["words"]
    starts = index["starts"]
    maximum_duration = float(index["maximum_duration"])
    lower = bisect_left(starts, start_seconds - maximum_duration)
    upper = bisect_right(starts, end_seconds)
    for word_start, word_end, word in indexed_words[lower:upper]:
        word_duration = max(0.001, word_end - word_start)
        overlap = min(end_seconds, word_end) - max(
            start_seconds,
            word_start,
        )
        midpoint = (word_start + word_end) / 2.0
        begins_before_span = (
            word_start < start_seconds < word_end
            and overlap >= minimum_overlap_seconds
            and overlap / word_duration >= minimum_overlap_fraction
        )
        if not (
            start_seconds <= midpoint <= end_seconds
            or begins_before_span
        ):
            continue
        key = (
            round(word_start, 4),
            round(word_end, 4),
            normalize_text(str(word.get("word") or "")),
        )
        if key in seen:
            continue
        seen.add(key)
        words.append((word_start, word_end, word))
    if not words:
        return span

    words.sort(key=lambda item: (item[0], item[1]))
    span["transcript"] = "".join(
        str(word.get("word") or "") for _, _, word in words
    ).strip()
    probabilities = [
        float(word["probability"])
        for _, _, word in words
        if word.get("probability") is not None
    ]
    span["asr_probability"] = (
        sum(probabilities) / len(probabilities) if probabilities else None
    )
    span["transcript_source"] = "session_word_span"
    return span


def _transcription_word_index(
    transcription: dict[str, Any] | None,
) -> dict[str, Any]:
    words = []
    maximum_duration = 0.001
    for transcript_segment in (transcription or {}).get("segments", []):
        for word in transcript_segment.get("words") or []:
            word_start = float(
                word.get("start", transcript_segment["start"])
            )
            word_end = float(
                word.get("end", transcript_segment["end"])
            )
            maximum_duration = max(
                maximum_duration,
                word_end - word_start,
            )
            words.append((word_start, word_end, word))
    words.sort(key=lambda item: (item[0], item[1]))
    return {
        "words": words,
        "starts": [item[0] for item in words],
        "maximum_duration": maximum_duration,
    }


def _text_matchable_lines(
    lines: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    source_indexes = [
        index
        for index, line in enumerate(lines)
        if not is_vocalization_script(line["line"])
    ]
    return [lines[index] for index in source_indexes], source_indexes


def _restore_source_line_indexes(
    actions: list[dict[str, Any]],
    source_indexes: list[int],
) -> list[dict[str, Any]]:
    for action in actions:
        action["line_index"] = source_indexes[int(action["line_index"])]
        for match in action.get("top_matches") or []:
            match["line_index"] = source_indexes[int(match["line_index"])]
    return actions


def _duration_plausibility(
    line: dict[str, Any],
    span: dict[str, Any],
) -> float:
    expected_words = max(1, word_count(line["line"]))
    expected_seconds = max(0.35, expected_words / 2.7)
    observed_seconds = max(
        0.05,
        float(span["end_seconds"]) - float(span["start_seconds"]),
    )
    duration_ratio = observed_seconds / expected_seconds
    distance = abs(math.log(max(0.05, duration_ratio)))
    return 100.0 * math.exp(-distance / 1.25)


def _order_hint(
    *,
    segment_index: int,
    segment_count: int,
    line_index: int,
    line_count: int,
) -> float:
    segment_position = segment_index / max(1, segment_count - 1)
    line_position = line_index / max(1, line_count - 1)
    return max(0.0, 1.0 - abs(segment_position - line_position))


def order_independent_align(
    segments: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Match non-overlapping audio spans without requiring script order.

    Each selected span has one primary line plus a small set of alternative
    line matches. Multiple separate spans may select the same line, which is
    how repeated takes are represented.
    """

    lines, source_indexes = _text_matchable_lines(lines)
    if not segments or not lines:
        return []

    segment_count = len(segments)
    line_count = len(lines)
    max_merge = int(settings.get("max_merge_segments", 3))
    max_gap = float(settings.get("max_merge_gap_seconds", 2.5))
    max_span = float(settings.get("max_span_seconds", 35.0))
    minimum_score = float(settings.get("candidate_min_score", 45.0))
    top_k = max(1, int(settings.get("candidate_top_k", 8)))
    noise_penalty = float(settings.get("noise_penalty", 2.2))
    merge_penalty = float(settings.get("merge_span_penalty", 0.2))
    duration_weight = float(settings.get("duration_hint_weight", 1.0))
    order_weight = float(settings.get("order_hint_weight", 0.0))

    proposals_by_start: list[list[dict[str, Any]]] = [
        [] for _ in range(segment_count)
    ]
    for segment_index in range(segment_count):
        for count in range(1, max_merge + 1):
            if not _valid_span(
                segments,
                segment_index,
                count,
                max_merge_gap_seconds=max_gap,
                max_span_seconds=max_span,
            ):
                break
            if (
                count > 1
                and bool(settings.get("merge_require_text_boundaries", True))
                and (
                    not str(
                        segments[segment_index].get("transcript") or ""
                    ).strip()
                    or not str(
                        segments[segment_index + count - 1].get(
                            "transcript"
                        )
                        or ""
                    ).strip()
                )
            ):
                continue
            span = _span_preview(segments, segment_index, count)
            line_matches = []
            for line_index, line in enumerate(lines):
                match_score = text_similarity(line["line"], span["transcript"])
                duration_score = _duration_plausibility(line, span)
                order_score = _order_hint(
                    segment_index=segment_index,
                    segment_count=segment_count,
                    line_index=line_index,
                    line_count=line_count,
                )
                ranking_score = (
                    match_score
                    + duration_weight * ((duration_score - 50.0) / 50.0)
                    + order_weight * order_score
                )
                line_matches.append(
                    {
                        "line_index": line_index,
                        "match_score": match_score,
                        "ranking_score": ranking_score,
                        "duration_plausibility": duration_score,
                        "order_hint": order_score,
                    }
                )
            line_matches.sort(
                key=lambda match: (
                    match["ranking_score"],
                    match["match_score"],
                    -match["line_index"],
                ),
                reverse=True,
            )
            pure_scores = sorted(
                (
                    match["match_score"],
                    int(match["line_index"]),
                )
                for match in line_matches
            )
            best_pure = pure_scores[-1] if pure_scores else (0.0, -1)
            second_pure = (
                pure_scores[-2] if len(pure_scores) > 1 else (0.0, -1)
            )
            for match in line_matches:
                other_score = (
                    second_pure[0]
                    if int(match["line_index"]) == best_pure[1]
                    else best_pure[0]
                )
                match["confidence_margin"] = (
                    float(match["match_score"]) - float(other_score)
                )
            primary = line_matches[0]
            if primary["match_score"] < minimum_score:
                continue
            proposal = {
                "type": "assigned",
                "start_index": segment_index,
                "count": count,
                "line_index": primary["line_index"],
                "match_score": primary["match_score"],
                "transcript": span["transcript"],
                "duration_plausibility": primary["duration_plausibility"],
                "order_hint": primary["order_hint"],
                "top_matches": [
                    match
                    for match in line_matches[:top_k]
                    if match["match_score"] >= minimum_score
                ],
            }
            proposal["path_utility"] = (
                (primary["ranking_score"] - 50.0) / 5.0
                - merge_penalty * (count - 1)
            )
            proposals_by_start[segment_index].append(proposal)

    best_scores = [0.0] * (segment_count + 1)
    choices: list[dict[str, Any] | None] = [None] * segment_count
    for segment_index in range(segment_count - 1, -1, -1):
        best_score = best_scores[segment_index + 1] - noise_penalty
        best_choice = None
        for proposal in proposals_by_start[segment_index]:
            next_index = segment_index + int(proposal["count"])
            proposal_score = float(proposal["path_utility"]) + best_scores[next_index]
            if proposal_score > best_score:
                best_score = proposal_score
                best_choice = proposal
        best_scores[segment_index] = best_score
        choices[segment_index] = best_choice

    selected = []
    segment_index = 0
    while segment_index < segment_count:
        choice = choices[segment_index]
        if choice is None:
            segment_index += 1
            continue
        selected.append(choice)
        segment_index += int(choice["count"])
    return _restore_source_line_indexes(selected, source_indexes)


def _apply_duplicate_line_policy(
    actions: list[dict[str, Any]],
    *,
    lines: list[dict[str, Any]],
    base_segments: list[dict[str, Any]],
    settings: dict[str, Any],
) -> None:
    policy = str(settings.get("duplicate_line_policy", "review")).lower()
    if policy not in {"review", "weak_order", "reuse"}:
        raise ValueError(
            "alignment.duplicate_line_policy must be 'review', "
            f"'weak_order', or 'reuse', got {policy!r}"
        )
    if policy != "weak_order":
        return

    duplicate_indexes: dict[str, list[int]] = defaultdict(list)
    for line_index, line in enumerate(lines):
        duplicate_indexes[normalize_text(line["line"])].append(line_index)
    duplicate_indexes = {
        text: indexes
        for text, indexes in duplicate_indexes.items()
        if text and len(indexes) > 1
    }
    if not duplicate_indexes:
        return

    maximum_gap = float(settings.get("take_group_gap_seconds", 12.0))
    for normalized, indexes in duplicate_indexes.items():
        matching_actions = []
        for action in actions:
            primary_index = int(action["line_index"])
            if normalize_text(lines[primary_index]["line"]) != normalized:
                continue
            matching_actions.append(action)
        matching_actions.sort(key=lambda action: int(action["start_index"]))
        if not matching_actions:
            continue

        clusters: list[list[dict[str, Any]]] = []
        previous_end = None
        for action in matching_actions:
            start_index = int(action["start_index"])
            count = int(action["count"])
            start_seconds = float(base_segments[start_index]["start_seconds"])
            end_seconds = float(
                base_segments[start_index + count - 1]["end_seconds"]
            )
            if (
                not clusters
                or previous_end is None
                or start_seconds - previous_end > maximum_gap
            ):
                clusters.append([action])
            else:
                clusters[-1].append(action)
            previous_end = end_seconds
        if len(clusters) < len(indexes):
            continue

        for cluster_index, cluster in enumerate(clusters):
            if len(clusters) == 1:
                duplicate_index = indexes[0]
            else:
                duplicate_position = round(
                    cluster_index
                    * (len(indexes) - 1)
                    / (len(clusters) - 1)
                )
                duplicate_index = indexes[duplicate_position]
            for action in cluster:
                action["line_index"] = duplicate_index
                action["duplicate_resolution"] = "weak_order"
                action["duplicate_resolved"] = True
                matches = list(action.get("top_matches") or [])
                selected = next(
                    (
                        match
                        for match in matches
                        if int(match["line_index"]) == duplicate_index
                    ),
                    None,
                )
                if selected is None:
                    selected = {
                        "line_index": duplicate_index,
                        "match_score": action["match_score"],
                        "ranking_score": action["match_score"],
                        "duration_plausibility": action.get(
                            "duration_plausibility", 0.0
                        ),
                        "order_hint": action.get("order_hint", 0.0),
                        "confidence_margin": 0.0,
                    }
                action["top_matches"] = [
                    selected,
                    *[
                        match
                        for match in matches
                        if int(match["line_index"]) != duplicate_index
                    ],
                ]


def _multisentence_fragment_join_actions(
    actions: list[dict[str, Any]],
    *,
    lines: list[dict[str, Any]],
    base_segments: list[dict[str, Any]],
    settings: dict[str, Any],
    transcription: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not bool(settings.get("fragment_join_enabled", True)):
        return []
    ordered_actions = sorted(
        actions,
        key=lambda action: (
            int(action["start_index"]),
            int(action["count"]),
        ),
    )
    maximum_segments = max(
        2,
        int(
            settings.get(
                "fragment_join_max_segments",
                settings.get(
                    "fragment_join_max_actions",
                    max(6, int(settings.get("max_merge_segments", 3))),
                ),
            )
        ),
    )
    maximum_actions_per_line = max(
        1,
        int(settings.get("fragment_join_max_actions", 3)),
    )
    maximum_gap = float(settings.get("max_merge_gap_seconds", 2.5))
    maximum_span = float(settings.get("max_span_seconds", 35.0))
    minimum_match_gain = float(
        settings.get("fragment_join_min_match_gain", 1.0)
    )
    minimum_clause_gain = float(
        settings.get("fragment_join_min_clause_gain", 20.0)
    )
    minimum_coverage_gain = float(
        settings.get("fragment_join_min_token_coverage_gain", 0.15)
    )
    minimum_token_coverage = float(
        settings.get("fragment_join_min_token_coverage", 0.85)
    )
    minimum_token_precision = float(
        settings.get("fragment_join_min_token_precision", 0.83)
    )
    neighbor_radius = max(
        0,
        int(settings.get("fragment_join_neighbor_radius", 1)),
    )
    minimum_ordered_score = float(
        settings.get("fragment_join_min_ordered_score", 70.0)
    )
    require_text_boundaries = bool(
        settings.get("fragment_join_require_text_boundaries", True)
    )
    only_incomplete_lines = bool(
        settings.get("fragment_join_only_incomplete_lines", True)
    )
    token_min_similarity = float(
        settings.get("fidelity_token_min_similarity", 78.0)
    )
    minimum_clause_score = float(
        settings.get("reliable_min_clause_score", 55.0)
    )
    existing = {
        (
            int(action["line_index"]),
            int(action["start_index"]),
            int(action["count"]),
        )
        for action in actions
    }
    evaluated: set[tuple[int, int, int]] = set()
    preview_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
    word_index = _transcription_word_index(transcription)
    joined = []

    def score_preview(
        line_text: str,
        preview: dict[str, Any],
    ) -> dict[str, Any]:
        transcript = str(preview.get("transcript") or "")
        fidelity = transcript_fidelity(
            line_text,
            transcript,
            token_min_similarity=token_min_similarity,
        )
        sentence = sentence_fidelity(
            line_text,
            transcript,
            token_min_similarity=token_min_similarity,
            minimum_clause_score=minimum_clause_score,
            short_clause_max_words=int(
                settings.get("reliable_short_clause_max_words", 4)
            ),
            short_clause_min_token_coverage=float(
                settings.get(
                    "reliable_short_clause_min_token_coverage",
                    1.0,
                )
            ),
        )
        return {
            "preview": preview,
            "match": text_similarity(line_text, transcript),
            "fidelity": fidelity,
            "sentence": sentence,
        }

    def preview_quality(scored: dict[str, Any]) -> tuple[Any, ...]:
        sentence = scored["sentence"]
        fidelity = scored["fidelity"]
        return (
            int(sentence["missing_clause_count"]) == 0,
            bool(sentence["clauses_in_order"]),
            float(fidelity["token_coverage"]),
            float(sentence["minimum_clause_score"]),
            float(scored["match"]),
            float(fidelity["token_precision"]),
            float(fidelity["ordered_similarity"]),
        )

    complete_line_indexes = set()
    if only_incomplete_lines:
        minimum_complete_match = float(
            settings.get("fragment_join_complete_min_match_score", 72.0)
        )
        minimum_complete_ordered = float(
            settings.get(
                "fragment_join_complete_min_ordered_score",
                70.0,
            )
        )
        minimum_length_ratio = float(
            settings.get("fragment_join_complete_min_length_ratio", 0.75)
        )
        maximum_length_ratio = float(
            settings.get("fragment_join_complete_max_length_ratio", 1.35)
        )
        for action in ordered_actions:
            line_index = int(action["line_index"])
            if line_index in complete_line_indexes:
                continue
            line_text = lines[line_index]["line"]
            scored = score_preview(
                line_text,
                {"transcript": str(action.get("transcript") or "")},
            )
            expected_words = max(1, word_count(line_text))
            observed_text = str(action.get("transcript") or "")
            observed_words = word_count(observed_text)
            length_ratio = observed_words / expected_words
            expected_counts = Counter(normalize_text(line_text).split())
            observed_counts = Counter(normalize_text(observed_text).split())
            repeated_excess = any(
                count >= 2 and count > expected_counts.get(token, 0)
                for token, count in observed_counts.items()
            )
            if (
                float(scored["match"]) >= minimum_complete_match
                and float(scored["fidelity"]["ordered_similarity"])
                >= minimum_complete_ordered
                and int(scored["sentence"]["missing_clause_count"]) == 0
                and bool(scored["sentence"]["clauses_in_order"])
                and minimum_length_ratio
                <= length_ratio
                <= maximum_length_ratio
                and not repeated_excess
            ):
                complete_line_indexes.add(line_index)

    for seed_action in ordered_actions:
        line_index = int(seed_action["line_index"])
        line = lines[line_index]
        if (
            is_vocalization_script(line["line"])
            or line_index in complete_line_indexes
        ):
            continue
        seed_start = int(seed_action["start_index"])
        seed_end = seed_start + int(seed_action["count"]) - 1
        first_start = max(0, seed_end - maximum_segments + 1)
        # A resolver-selected fragment can itself be a multi-segment span. A
        # complete subspan may begin inside it and continue to the right, so
        # search every window that overlaps the seed rather than only windows
        # that fully contain it.
        last_start = min(len(base_segments) - 2, seed_end)
        for start_index in range(first_start, last_start + 1):
            for count in range(2, maximum_segments + 1):
                end_index = start_index + count - 1
                if end_index >= len(base_segments):
                    break
                if (
                    end_index < seed_start - neighbor_radius
                    or start_index > seed_end + neighbor_radius
                ):
                    continue
                key = (line_index, start_index, count)
                if key in existing or key in evaluated:
                    continue
                evaluated.add(key)
                if not _valid_span(
                    base_segments,
                    start_index,
                    count,
                    max_merge_gap_seconds=maximum_gap,
                    max_span_seconds=maximum_span,
                ):
                    continue

                if require_text_boundaries and (
                    not str(
                        base_segments[start_index].get("transcript") or ""
                    ).strip()
                    or not str(
                        base_segments[end_index].get("transcript") or ""
                    ).strip()
                ):
                    continue

                span_key = (start_index, count)
                previews = preview_cache.get(span_key)
                if previews is None:
                    base_span = _span_preview(
                        base_segments,
                        start_index,
                        count,
                    )
                    base_span["transcript_source"] = "segment_asr_span"
                    previews = [base_span]
                    if transcription:
                        timestamp_span = _span_preview_with_transcription(
                            base_segments,
                            start_index,
                            count,
                            transcription=transcription,
                            settings=settings,
                            word_index=word_index,
                        )
                        if (
                            normalize_text(timestamp_span["transcript"])
                            != normalize_text(base_span["transcript"])
                        ):
                            previews.append(timestamp_span)
                    preview_cache[span_key] = previews
                scored_previews = [
                    score_preview(line["line"], preview)
                    for preview in previews
                    if str(preview.get("transcript") or "").strip()
                ]
                if not scored_previews:
                    continue
                combined = max(scored_previews, key=preview_quality)
                span = combined["preview"]
                combined_match = float(combined["match"])
                combined_fidelity = combined["fidelity"]
                combined_sentence = combined["sentence"]
                contained_actions = [
                    action
                    for action in ordered_actions
                    if int(action["line_index"]) == line_index
                    and int(action["start_index"]) >= start_index
                    and (
                        int(action["start_index"])
                        + int(action["count"])
                        - 1
                    )
                    <= end_index
                ]
                comparison_actions = contained_actions or [seed_action]
                fragment_match = max(
                    float(action["match_score"])
                    for action in comparison_actions
                )
                fragment_metrics = [
                    score_preview(
                        line["line"],
                        {
                            "transcript": str(
                                action.get("transcript") or ""
                            )
                        },
                    )
                    for action in comparison_actions
                ]
                fragment_clause_score = max(
                    float(metric["sentence"]["minimum_clause_score"])
                    for metric in fragment_metrics
                )
                fragment_missing_clauses = min(
                    int(metric["sentence"]["missing_clause_count"])
                    for metric in fragment_metrics
                )
                fragment_token_coverage = max(
                    float(metric["fidelity"]["token_coverage"])
                    for metric in fragment_metrics
                )
                match_gain = combined_match - fragment_match
                coverage_gain = (
                    float(combined_fidelity["token_coverage"])
                    - fragment_token_coverage
                )
                missing_clause_gain = (
                    fragment_missing_clauses
                    - int(combined_sentence["missing_clause_count"])
                )
                if (
                    match_gain < minimum_match_gain
                    and coverage_gain < minimum_coverage_gain
                    and missing_clause_gain <= 0
                ):
                    continue
                clause_gain = (
                    float(combined_sentence["minimum_clause_score"])
                    - fragment_clause_score
                )
                if (
                    len(script_clauses(line["line"])) >= 2
                    and clause_gain < minimum_clause_gain
                    and missing_clause_gain <= 0
                    and coverage_gain < minimum_coverage_gain
                ):
                    continue
                if (
                    combined_sentence["clause_count"] >= 2
                    and combined_sentence["missing_clause_count"] > 0
                ):
                    continue
                if not bool(combined_sentence["clauses_in_order"]):
                    continue
                if (
                    float(combined_fidelity["token_coverage"])
                    < minimum_token_coverage
                ):
                    continue
                if (
                    float(combined_fidelity["ordered_similarity"])
                    < minimum_ordered_score
                ):
                    continue
                if (
                    float(combined_fidelity["token_precision"])
                    < minimum_token_precision
                ):
                    continue

                # Do not generate a join whose apparent improvement comes only
                # from repeating the same text across adjacent takes.
                if (
                    int(combined_fidelity["extra_word_count"]) > 0
                    and float(combined_fidelity["token_precision"])
                    < float(
                        settings.get(
                            "reliable_min_token_precision",
                            0.70,
                        )
                    )
                ):
                    continue

                other_score = max(
                    (
                        text_similarity(
                            other_line["line"],
                            span["transcript"],
                        )
                        for other_index, other_line in enumerate(lines)
                        if other_index != line_index
                        and not is_vocalization_script(other_line["line"])
                    ),
                    default=0.0,
                )
                duration_score = _duration_plausibility(line, span)
                relevant_hints = contained_actions or [seed_action]
                order_score = sum(
                    float(action.get("order_hint", 0.0))
                    for action in relevant_hints
                ) / len(relevant_hints)
                joined.append(
                    {
                        "type": "assigned",
                        "start_index": start_index,
                        "count": count,
                        "line_index": line_index,
                        "match_score": combined_match,
                        "transcript": span["transcript"],
                        "transcript_source": span.get(
                            "transcript_source",
                            "base_segments",
                        ),
                        "duration_plausibility": duration_score,
                        "order_hint": order_score,
                        "top_matches": [
                            {
                                "line_index": line_index,
                                "match_score": combined_match,
                                "ranking_score": combined_match,
                                "duration_plausibility": duration_score,
                                "order_hint": order_score,
                                "confidence_margin": (
                                    combined_match - other_score
                                ),
                            }
                        ],
                        "fragment_join": True,
                        "fragment_source_count": count,
                    }
                )
                existing.add(key)
                break

    joined_by_line: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for action in joined:
        joined_by_line[int(action["line_index"])].append(action)
    selected_joins = []
    for line_index, line_actions in joined_by_line.items():
        line_text = lines[line_index]["line"]

        def action_quality(action: dict[str, Any]) -> tuple[Any, ...]:
            scored = score_preview(
                line_text,
                {"transcript": str(action.get("transcript") or "")},
            )
            return (
                int(scored["sentence"]["missing_clause_count"]) == 0,
                bool(scored["sentence"]["clauses_in_order"]),
                float(scored["fidelity"]["token_coverage"]),
                float(scored["match"]),
                float(scored["fidelity"]["token_precision"]),
                float(scored["fidelity"]["ordered_similarity"]),
                -int(action["count"]),
            )

        line_actions.sort(key=action_quality, reverse=True)
        selected_joins.extend(line_actions[:maximum_actions_per_line])
    selected_joins.sort(
        key=lambda action: (
            int(action["start_index"]),
            int(action["count"]),
            int(action["line_index"]),
        )
    )
    return selected_joins


def _expand_alignment_actions(
    actions: list[dict[str, Any]],
    *,
    base_segments: list[dict[str, Any]],
    session_id: str,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    take_group_gap = float(settings.get("take_group_gap_seconds", 12.0))
    group_number = 0
    previous_primary: dict[str, Any] | None = None
    expanded = []

    for action in actions:
        primary_line_index = int(action["line_index"])
        start_index = int(action["start_index"])
        count = int(action["count"])
        start_seconds = float(base_segments[start_index]["start_seconds"])
        end_seconds = float(
            base_segments[start_index + count - 1]["end_seconds"]
        )
        if (
            previous_primary is not None
            and int(previous_primary["line_index"]) == primary_line_index
            and start_seconds - float(previous_primary["end_seconds"])
            <= take_group_gap
        ):
            take_group_id = str(previous_primary["take_group_id"])
        else:
            group_number += 1
            take_group_id = f"{session_id}__g{group_number:05d}"
        previous_primary = {
            "line_index": primary_line_index,
            "end_seconds": end_seconds,
            "take_group_id": take_group_id,
        }

        matches = action.get("top_matches") or [
            {
                "line_index": primary_line_index,
                "match_score": action["match_score"],
                "duration_plausibility": action.get(
                    "duration_plausibility", 0.0
                ),
                "order_hint": action.get("order_hint", 0.0),
            }
        ]
        for match_rank, match in enumerate(matches, start=1):
            expanded.append(
                {
                    "type": "assigned",
                    "start_index": start_index,
                    "count": count,
                    "line_index": int(match["line_index"]),
                    "primary_line_index": primary_line_index,
                    "match_score": float(match["match_score"]),
                    "transcript": action["transcript"],
                    "duration_plausibility": float(
                        match.get("duration_plausibility", 0.0)
                    ),
                    "order_hint": float(match.get("order_hint", 0.0)),
                    "confidence_margin": float(
                        match.get("confidence_margin", 0.0)
                    ),
                    "segment_match_rank": match_rank,
                    "is_primary_match": match_rank == 1,
                    "take_group_id": (
                        take_group_id if match_rank == 1 else None
                    ),
                    "duplicate_resolution": action.get(
                        "duplicate_resolution"
                    ),
                    "duplicate_resolved": bool(
                        action.get("duplicate_resolved", False)
                        and match_rank == 1
                    ),
                    "fragment_join": bool(
                        action.get("fragment_join", False)
                        and match_rank == 1
                    ),
                    "fragment_source_count": (
                        int(action.get("fragment_source_count", 0))
                        if match_rank == 1
                        else 0
                    ),
                }
            )
    return expanded



def _candidate_reliability(
    *,
    line: dict[str, Any],
    match_score: float,
    margin: float,
    settings: dict[str, Any],
    observed: str | None = None,
    duplicate_text: bool = False,
    unsafe_untranscribed_merge: bool = False,
    duration_plausibility: float | None = None,
) -> tuple[bool, str]:
    if is_nonverbal_script(line["line"]):
        return False, "NONVERBAL_SCRIPT"
    if unsafe_untranscribed_merge:
        return False, "MERGED_UNTRANSCRIBED_AUDIO"
    if (
        duration_plausibility is not None
        and duration_plausibility
        < float(settings.get("reliable_min_duration_plausibility", 25.0))
    ):
        return False, "POSSIBLE_REPEATED_TAKES"
    if (
        observed
        and not duplicate_text
        and normalize_text(line["line"]) == normalize_text(observed)
    ):
        return True, ""
    fidelity = transcript_fidelity(
        line["line"],
        observed or "",
        token_min_similarity=float(
            settings.get("fidelity_token_min_similarity", 78.0)
        ),
    )
    sentence = sentence_fidelity(
        line["line"],
        observed or "",
        token_min_similarity=float(
            settings.get("fidelity_token_min_similarity", 78.0)
        ),
        minimum_clause_score=float(
            settings.get("reliable_min_clause_score", 55.0)
        ),
        short_clause_max_words=int(
            settings.get("reliable_short_clause_max_words", 4)
        ),
        short_clause_min_token_coverage=float(
            settings.get(
                "reliable_short_clause_min_token_coverage",
                1.0,
            )
        ),
    )
    is_short_line = word_count(line["line"]) <= 3
    if is_short_line:
        minimum_score = float(settings.get("short_line_min_score", 88.0))
        minimum_margin = float(settings.get("short_line_min_margin", 15.0))
        if match_score < minimum_score:
            return False, "SHORT_LINE_LOW_SCORE"
        if margin < minimum_margin:
            return False, "SHORT_LINE_AMBIGUOUS"
        if (
            sentence["clause_count"] >= 2
            and sentence["missing_clause_count"] > 0
        ):
            return False, "MISSING_SENTENCE"
        if fidelity["token_coverage"] < float(
            settings.get("short_line_min_token_coverage", 1.0)
        ):
            return False, "SHORT_LINE_INCOMPLETE_TRANSCRIPT"
        if fidelity["token_precision"] < float(
            settings.get("short_line_min_token_precision", 1.0)
        ):
            return False, "SHORT_LINE_EXTRA_WORDS"
        if fidelity["ordered_similarity"] < float(
            settings.get("short_line_min_ordered_score", 70.0)
        ):
            return False, "SHORT_LINE_ORDER_MISMATCH"
        return True, ""
    if match_score < float(settings.get("reliable_min_score", 72.0)):
        return False, "LOW_MATCH_SCORE"
    if margin < float(settings.get("reliable_min_margin", 8.0)):
        return False, "AMBIGUOUS_MATCH"
    if (
        sentence["clause_count"] >= 2
        and sentence["missing_clause_count"] > 0
    ):
        return False, "MISSING_SENTENCE"
    if fidelity["token_coverage"] < float(
        settings.get("reliable_min_token_coverage", 0.60)
    ):
        return False, "INCOMPLETE_TRANSCRIPT"
    if fidelity["token_precision"] < float(
        settings.get("reliable_min_token_precision", 0.70)
    ):
        return False, "EXCESS_TRANSCRIPT_WORDS"
    if (
        sentence["clause_count"] >= 2
        and not bool(sentence["clauses_in_order"])
    ):
        return False, "SENTENCE_ORDER_MISMATCH"
    if fidelity["ordered_similarity"] < float(
        settings.get("reliable_min_ordered_score", 55.0)
    ):
        return False, "LOW_ORDERED_SIMILARITY"
    return True, ""


def _has_unsafe_untranscribed_merge(
    *,
    action: dict[str, Any],
    base_segments: list[dict[str, Any]],
    settings: dict[str, Any],
) -> bool:
    if not bool(settings.get("auto_reject_untranscribed_merge", True)):
        return False
    start_index = int(action["start_index"])
    count = int(action["count"])
    if count <= 1:
        return False
    minimum_seconds = float(
        settings.get("untranscribed_merge_min_seconds", 0.5)
    )
    minimum_rms = float(
        settings.get("untranscribed_merge_min_rms_dbfs", -45.0)
    )
    for segment in base_segments[start_index : start_index + count]:
        if str(segment.get("transcript") or "").strip():
            continue
        duration = float(
            (segment.get("metrics") or {}).get("duration_seconds")
            or (
                float(segment.get("end_seconds", 0.0))
                - float(segment.get("start_seconds", 0.0))
            )
        )
        rms = (segment.get("metrics") or {}).get("rms_dbfs")
        if (
            duration >= minimum_seconds
            and (rms is None or float(rms) >= minimum_rms)
        ):
            return True
    return False


def align_project(
    *,
    project_dir: Path,
    project: dict[str, Any],
    session_filter: set[str] | None = None,
    segment_model_override: str | None = None,
    segment_device_override: str | None = None,
) -> dict[str, Path]:
    manifest_path = project_dir / "segments_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing segment manifest: {manifest_path}. Run segment first."
        )
    segment_asr_settings = dict(project.get("segment_transcription") or {})
    segment_asr_runtime: dict[str, Any] = {}
    if bool(segment_asr_settings.get("enabled", False)):
        transcribe_segments_project(
            project_dir=project_dir,
            project=project,
            session_filter=session_filter,
            model_override=segment_model_override,
            device_override=segment_device_override,
            runtime=segment_asr_runtime,
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
        text_matchable_session_lines = [
            line
            for line in session_lines
            if not is_vocalization_script(line["line"])
        ]
        normalized_line_counts = Counter(
            normalize_text(line["line"]) for line in session_lines
        )
        print(
            f"[align {session_index}] {session['id']}: "
            f"{len(base_segments)} segments → {len(session_lines)} lines",
            flush=True,
        )
        primary_actions = order_independent_align(
            base_segments,
            session_lines,
            settings,
        )
        _apply_duplicate_line_policy(
            primary_actions,
            lines=session_lines,
            base_segments=base_segments,
            settings=settings,
        )
        transcription_path = (
            project_dir / "transcripts" / f"{session['id']}.json"
        )
        session_transcription = (
            read_json(transcription_path)
            if transcription_path.is_file()
            else None
        )
        fragment_join_actions = _multisentence_fragment_join_actions(
            primary_actions,
            lines=session_lines,
            base_segments=base_segments,
            settings=settings,
            transcription=session_transcription,
        )
        candidate_actions = sorted(
            [*primary_actions, *fragment_join_actions],
            key=lambda action: (
                int(action["start_index"]),
                int(action["count"]),
            ),
        )
        actions = _expand_alignment_actions(
            candidate_actions,
            base_segments=base_segments,
            session_id=session["id"],
            settings=settings,
        )
        verification_enabled = bool(
            segment_asr_settings.get("enabled", False)
            and segment_asr_settings.get(
                "candidate_verification_enabled",
                True,
            )
        )
        verification_min_score = float(
            segment_asr_settings.get(
                "candidate_verification_min_match_score",
                settings.get("candidate_min_score", 45.0),
            )
        )
        materialized_by_span: dict[
            tuple[int, int],
            dict[str, Any],
        ] = {}
        verification_segments: dict[str, dict[str, Any]] = {}
        for action in actions:
            span_key = (
                int(action["start_index"]),
                int(action["count"]),
            )
            segment = materialized_by_span.get(span_key)
            if segment is None:
                segment = materialize_derived_segment(
                    project_dir=project_dir,
                    project=project,
                    session_entry=session_entry,
                    base_segments=base_segments,
                    start_index=span_key[0],
                    count=span_key[1],
                )
                materialized_by_span[span_key] = segment
            line = session_lines[int(action["line_index"])]
            if (
                verification_enabled
                and not is_vocalization_script(line["line"])
                and float(action["match_score"]) >= verification_min_score
            ):
                verification_segments[str(segment["segment_id"])] = segment
        exact_asr_by_segment = transcribe_candidate_spans(
            project_dir=project_dir,
            project=project,
            segments=list(verification_segments.values()),
            runtime=segment_asr_runtime,
            model_override=segment_model_override,
            device_override=segment_device_override,
        )
        normalized_session_lines = [
            normalize_text(line["line"]) for line in session_lines
        ]
        exact_scores_by_segment: dict[str, list[float]] = {}
        for segment_id, exact_asr in exact_asr_by_segment.items():
            exact_text = str(exact_asr.get("transcript") or "").strip()
            if exact_text and not exact_asr.get("error"):
                exact_scores_by_segment[segment_id] = [
                    text_similarity(line["line"], exact_text)
                    for line in session_lines
                ]
        reliable_coverage: set[int] = set()
        assignment_by_base: dict[int, list[dict[str, Any]]] = defaultdict(list)
        serialized_actions = []

        for action in actions:
            line = session_lines[action["line_index"]]
            segment = materialized_by_span[
                (
                    int(action["start_index"]),
                    int(action["count"]),
                )
            ]
            preliminary_transcript = str(
                action.get("transcript")
                or segment.get("transcript")
                or ""
            )
            observed_transcript = preliminary_transcript
            match_score = float(action["match_score"])
            exact_span_asr_verified = False
            exact_span_asr_error = ""
            transcript_source = segment.get(
                "transcript_source",
                action.get("transcript_source", "session_asr"),
            )
            verification_required = bool(
                verification_enabled
                and not is_vocalization_script(line["line"])
                and match_score >= verification_min_score
            )
            if verification_required:
                exact_asr = exact_asr_by_segment.get(
                    str(segment["segment_id"])
                )
                if exact_asr and not exact_asr.get("error"):
                    exact_text = str(
                        exact_asr.get("transcript") or ""
                    ).strip()
                    if exact_text:
                        observed_transcript = exact_text
                        exact_scores = exact_scores_by_segment[
                            str(segment["segment_id"])
                        ]
                        match_score = exact_scores[
                            int(action["line_index"])
                        ]
                        exact_span_asr_verified = True
                        transcript_source = "exact_span_asr"
                        segment["transcript"] = observed_transcript
                        segment["word_count"] = int(
                            exact_asr.get("word_count")
                            if exact_asr.get("word_count") is not None
                            else word_count(observed_transcript)
                        )
                        segment["words"] = list(
                            exact_asr.get("words") or []
                        )
                        segment["asr_probability"] = exact_asr.get(
                            "asr_probability"
                        )
                        segment["transcript_source"] = transcript_source
                    else:
                        exact_span_asr_error = "EMPTY_TRANSCRIPT"
                else:
                    exact_span_asr_error = str(
                        (exact_asr or {}).get(
                            "error",
                            "EXACT_SPAN_ASR_MISSING",
                        )
                    )
                    print(
                        f"[candidate ASR] {segment['segment_id']} failed: "
                        f"{exact_span_asr_error}",
                        flush=True,
                    )

            if exact_span_asr_verified:
                exact_scores = exact_scores_by_segment[
                    str(segment["segment_id"])
                ]
                second_score = max(
                    (
                        score
                        for other_index, score in enumerate(exact_scores)
                        if other_index != int(action["line_index"])
                    ),
                    default=0.0,
                )
                margin = match_score - second_score
            elif "confidence_margin" in action:
                margin = float(action["confidence_margin"])
            else:
                all_scores = sorted(
                    (
                        text_similarity(
                            other_line["line"],
                            observed_transcript,
                        ),
                        other_line["line_id"],
                    )
                    for other_line in session_lines
                    if other_line["line_id"] != line["line_id"]
                )
                second_score = all_scores[-1][0] if all_scores else 0.0
                margin = match_score - second_score
            duplicate_policy = str(
                settings.get("duplicate_line_policy", "review")
            ).lower()
            primary_line = session_lines[
                int(action.get("primary_line_index", action["line_index"]))
            ]
            reusable_duplicate = bool(
                duplicate_policy == "reuse"
                and normalize_text(primary_line["line"])
                == normalize_text(line["line"])
            )
            fidelity = transcript_fidelity(
                line["line"],
                observed_transcript,
                token_min_similarity=float(
                    settings.get("fidelity_token_min_similarity", 78.0)
                ),
            )
            sentence = sentence_fidelity(
                line["line"],
                observed_transcript,
                token_min_similarity=float(
                    settings.get("fidelity_token_min_similarity", 78.0)
                ),
                minimum_clause_score=float(
                    settings.get("reliable_min_clause_score", 55.0)
                ),
                short_clause_max_words=int(
                    settings.get("reliable_short_clause_max_words", 4)
                ),
                short_clause_min_token_coverage=float(
                    settings.get(
                        "reliable_short_clause_min_token_coverage",
                        1.0,
                    )
                ),
            )
            unsafe_untranscribed_merge = _has_unsafe_untranscribed_merge(
                action=action,
                base_segments=base_segments,
                settings=settings,
            )
            reliable, reason = _candidate_reliability(
                line=line,
                match_score=match_score,
                margin=margin,
                settings=settings,
                observed=observed_transcript,
                duplicate_text=(
                    normalized_line_counts[normalize_text(line["line"])] > 1
                    and not action.get("duplicate_resolved", False)
                    and not reusable_duplicate
                ),
                unsafe_untranscribed_merge=unsafe_untranscribed_merge,
                duration_plausibility=(
                    float(action["duration_plausibility"])
                    if action.get("duration_plausibility") is not None
                    else None
                ),
            )
            if verification_required and not exact_span_asr_verified:
                reliable = False
                reason = "EXACT_SPAN_ASR_FAILED"
            candidate_is_primary = bool(
                action.get("is_primary_match", True)
            )
            if exact_span_asr_verified:
                exact_scores = exact_scores_by_segment[
                    str(segment["segment_id"])
                ]
                line_index = int(action["line_index"])
                best_other_score = max(
                    (
                        score
                        for other_index, score in enumerate(exact_scores)
                        if (
                            other_index != line_index
                            and normalized_session_lines[other_index]
                            != normalized_session_lines[line_index]
                        )
                    ),
                    default=0.0,
                )
                if best_other_score > match_score + 0.01:
                    candidate_is_primary = False
            if (
                not candidate_is_primary
                and not reusable_duplicate
            ):
                reliable = False
                reason = "SEGMENT_BETTER_MATCH_ELSEWHERE"
            if action.get("forced_review_reason"):
                reliable = False
                reason = str(action["forced_review_reason"])
            technical_score = _technical_score(segment)
            selection_score = (
                match_score
                + 0.10 * technical_score
                + 5.0 * float(segment.get("asr_probability") or 0.0)
                + float(
                    settings.get("clause_completeness_weight", 5.0)
                )
                * (
                    float(sentence["minimum_clause_score"]) / 100.0
                    if sentence["clause_count"] >= 2
                    else 1.0
                )
                + (
                    float(settings.get("primary_match_bonus", 2.0))
                    if candidate_is_primary
                    else 0.0
                )
            )
            candidate = {
                "line_id": line["line_id"],
                "session_id": session["id"],
                "segment_id": segment["segment_id"],
                "segment_file": segment["file"],
                "transcript": observed_transcript,
                "match_score": match_score,
                "ordered_similarity": fidelity["ordered_similarity"],
                "token_coverage": fidelity["token_coverage"],
                "token_precision": fidelity["token_precision"],
                "extra_word_count": fidelity["extra_word_count"],
                "clause_count": sentence["clause_count"],
                "minimum_clause_score": sentence["minimum_clause_score"],
                "missing_clause_count": sentence["missing_clause_count"],
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
                "is_primary_match": candidate_is_primary,
                "segment_match_rank": int(
                    action.get("segment_match_rank", 1)
                ),
                "take_group_id": action.get("take_group_id"),
                "duration_plausibility": float(
                    action.get("duration_plausibility", 0.0)
                ),
                "order_hint": float(action.get("order_hint", 0.0)),
                "duplicate_resolution": action.get("duplicate_resolution"),
                "transcript_source": transcript_source,
                "exact_span_asr_verified": exact_span_asr_verified,
                "exact_span_asr_error": exact_span_asr_error,
                "unsafe_untranscribed_merge": unsafe_untranscribed_merge,
                "fragment_join": bool(action.get("fragment_join", False)),
                "fragment_source_count": int(
                    action.get("fragment_source_count", 0)
                ),
            }
            if (
                match_score
                >= float(settings.get("candidate_min_score", 45.0))
                or action.get("force_candidate", False)
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
                for line in text_matchable_session_lines
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
                    "session_id": session["id"],
                    "base_index": base_index,
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
                    "technical_score": _technical_score(segment),
                    "audible": (
                        float(segment["metrics"]["duration_seconds"]) >= 0.15
                        and float(
                            segment["metrics"].get("rms_dbfs")
                            if segment["metrics"].get("rms_dbfs") is not None
                            else -999.0
                        )
                        >= float(
                            settings.get(
                                "untranscribed_merge_min_rms_dbfs",
                                -45.0,
                            )
                        )
                    ),
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

    review_path = project_dir / REVIEW_FILE_NAME
    review_data = build_line_review(
        source_lines=source_data["lines"],
        candidates_by_line=by_line,
        unmatched_segments=unmatched_rows,
    )
    if review_path.is_file():
        review_data = preserve_manual_selections(
            review_data,
            load_line_review(review_path),
        )
    write_json(review_path, review_data)
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
