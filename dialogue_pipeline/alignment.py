from __future__ import annotations

import heapq
import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rapidfuzz import fuzz

from .alignment_settings import AlignmentSettings
from .audio import (
    first_quiet_pcm_boundary,
    pcm_voice_bounds,
    pcm_leading_sibilant_start,
    pcm_voice_regions,
    quietest_pcm_boundary,
)
from .cancellation import check_processing_cancelled
from .project import load_source_data
from .review import (
    REVIEW_FILE_NAME,
    build_line_review,
    load_line_review,
    preserve_manual_selections,
)
from .segmentation import (
    materialize_derived_segment,
    materialize_trimmed_segment,
)
from .transcription import (
    transcribe_candidate_spans,
    transcribe_segments_project,
)
from .util import (
    is_nonverbal_script,
    is_vocalization_script,
    normalize_spoken_text,
    normalize_text,
    read_json,
    resolve_project_path,
    verbal_script_text,
    word_count,
    write_json,
)
from .workbook_io import lines_for_session


@dataclass(frozen=True)
class TextFeatures:
    normalized: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class TranscriptEvaluation:
    match_score: float
    fidelity: dict[str, float | int]
    sentence: dict[str, Any]


_PARALINGUISTIC_TOKENS = frozenset(
    {
        "ah",
        "aha",
        "ahh",
        "argh",
        "breath",
        "breathing",
        "cough",
        "coughing",
        "coughs",
        "gasp",
        "gasps",
        "giggle",
        "giggles",
        "groan",
        "groans",
        "grunt",
        "grunts",
        "ha",
        "hah",
        "haha",
        "heh",
        "hehe",
        "hiccup",
        "hiccups",
        "hm",
        "hmm",
        "hmmm",
        "huh",
        "laugh",
        "laughing",
        "laughs",
        "laughter",
        "moan",
        "moans",
        "mm",
        "mmm",
        "oh",
        "ooh",
        "pfft",
        "sigh",
        "sighing",
        "sighs",
        "sniff",
        "sniffs",
        "snort",
        "snorts",
        "sob",
        "sobs",
        "uh",
        "ugh",
        "um",
        "wheeze",
        "wheezes",
        "whimper",
        "whimpers",
        "yawn",
        "yawns",
    }
)

_HESITATION_TOKENS = frozenset(
    {"er", "err", "erm", "uh", "uhh", "um", "umm"}
)


def _script_edge_performance_cues(value: str) -> tuple[bool, bool]:
    """Return whether a spoken line starts or ends with a direction cue."""

    text = value or ""
    return (
        bool(re.match(r"^\s*\([^)]*\)", text)),
        bool(re.search(r"\([^)]*\)\s*$", text)),
    )


def _is_paralinguistic_transcript(value: str) -> bool:
    tokens = normalize_text(value).split()
    if not tokens:
        return False
    return all(
        token in _PARALINGUISTIC_TOKENS
        or bool(
            re.fullmatch(
                r"(?:ha)+|h+a+h*|h+m+|m+h+|p+f+t+|u+h+|a+h+|o+h+",
                token,
            )
        )
        for token in tokens
    )


def _text_features(
    value: str,
    *,
    remove_parenthetical_cues: bool = False,
) -> TextFeatures:
    normalized = normalize_spoken_text(
        value,
        remove_parenthetical_cues=remove_parenthetical_cues,
    )
    tokens = tuple(
        "hesitation" if token in _HESITATION_TOKENS else token
        for token in normalized.split()
    )
    normalized = " ".join(tokens)
    return TextFeatures(
        normalized=normalized,
        tokens=tokens,
    )


def _text_similarity_features(
    expected: TextFeatures,
    observed: TextFeatures,
) -> float:
    if not expected.normalized or not observed.normalized:
        return 0.0
    # Keep candidate scoring sensitive to the spoken order within a line.
    # Workbook lines are still matched in arbitrary recording order by the
    # aligner, but a suffix followed by the next take's prefix must not look
    # like a complete delivery merely because it contains the same token set.
    return float(fuzz.ratio(expected.normalized, observed.normalized))


def text_similarity(expected: str, observed: str) -> float:
    return _text_similarity_features(
        _text_features(expected, remove_parenthetical_cues=True),
        _text_features(observed),
    )


def _ordered_token_edit_operations(
    expected_tokens: list[str],
    observed_tokens: list[str],
    *,
    token_min_similarity: float,
) -> list[str]:
    """Align tokens while preserving insertions/deletions at line boundaries."""

    return [
        operation
        for operation, _, _ in _ordered_token_alignment(
            expected_tokens,
            observed_tokens,
            token_min_similarity=token_min_similarity,
        )
    ]


def _ordered_token_alignment(
    expected_tokens: list[str],
    observed_tokens: list[str],
    *,
    token_min_similarity: float,
) -> list[tuple[str, int | None, int | None]]:
    expected_count = len(expected_tokens)
    observed_count = len(observed_tokens)
    costs = [
        [0.0] * (observed_count + 1)
        for _ in range(expected_count + 1)
    ]
    operations = [
        [""] * (observed_count + 1)
        for _ in range(expected_count + 1)
    ]
    for expected_index in range(1, expected_count + 1):
        costs[expected_index][0] = float(expected_index)
        operations[expected_index][0] = "delete"
    for observed_index in range(1, observed_count + 1):
        costs[0][observed_index] = float(observed_index)
        operations[0][observed_index] = "insert"

    for expected_index in range(1, expected_count + 1):
        for observed_index in range(1, observed_count + 1):
            similarity = fuzz.ratio(
                expected_tokens[expected_index - 1],
                observed_tokens[observed_index - 1],
            )
            if similarity >= token_min_similarity:
                diagonal_operation = "match"
                diagonal_cost = (
                    costs[expected_index - 1][observed_index - 1]
                    + 0.25 * (100.0 - similarity) / 100.0
                )
                diagonal_priority = 0
            else:
                diagonal_operation = "substitute"
                diagonal_cost = (
                    costs[expected_index - 1][observed_index - 1] + 1.0
                )
                diagonal_priority = 3
            choices = [
                (diagonal_cost, diagonal_priority, diagonal_operation),
                (
                    costs[expected_index - 1][observed_index] + 1.0,
                    1,
                    "delete",
                ),
                (
                    costs[expected_index][observed_index - 1] + 1.0,
                    2,
                    "insert",
                ),
            ]
            cost, _, operation = min(choices)
            costs[expected_index][observed_index] = cost
            operations[expected_index][observed_index] = operation

    result: list[tuple[str, int | None, int | None]] = []
    expected_index = expected_count
    observed_index = observed_count
    while expected_index or observed_index:
        operation = operations[expected_index][observed_index]
        if operation in {"match", "substitute"}:
            result.append(
                (operation, expected_index - 1, observed_index - 1)
            )
            expected_index -= 1
            observed_index -= 1
        elif operation == "delete":
            result.append((operation, expected_index - 1, None))
            expected_index -= 1
        else:
            result.append((operation, None, observed_index - 1))
            observed_index -= 1
    result.reverse()
    return result


def _boundary_edit_counts(operations: list[str]) -> dict[str, int]:
    try:
        first_match = operations.index("match")
        last_match = len(operations) - 1 - operations[::-1].index("match")
    except ValueError:
        first_match = len(operations)
        last_match = -1
    leading = operations[:first_match]
    trailing = operations[last_match + 1 :]
    return {
        "leading_missing_token_count": leading.count("delete"),
        "leading_extra_token_count": leading.count("insert"),
        "leading_substitution_count": leading.count("substitute"),
        "trailing_missing_token_count": trailing.count("delete"),
        "trailing_extra_token_count": trailing.count("insert"),
        "trailing_substitution_count": trailing.count("substitute"),
    }


def transcript_fidelity(
    expected: str,
    observed: str,
    *,
    token_min_similarity: float = 78.0,
    boundary_window_tokens: int = 4,
    boundary_observed_slack_tokens: int = 2,
) -> dict[str, float | int]:
    """Measure ordered agreement separately from tolerant candidate retrieval."""

    return _transcript_fidelity_features(
        _text_features(expected, remove_parenthetical_cues=True),
        _text_features(observed),
        token_min_similarity=token_min_similarity,
        boundary_window_tokens=boundary_window_tokens,
        boundary_observed_slack_tokens=boundary_observed_slack_tokens,
    )


def _transcript_fidelity_features(
    expected: TextFeatures,
    observed: TextFeatures,
    *,
    token_min_similarity: float,
    boundary_window_tokens: int,
    boundary_observed_slack_tokens: int,
) -> dict[str, float | int]:
    expected_tokens = list(expected.tokens)
    observed_tokens = list(observed.tokens)
    if not expected_tokens or not observed_tokens:
        return {
            "ordered_similarity": 0.0,
            "token_coverage": 0.0,
            "token_precision": 0.0,
            "extra_word_count": len(observed_tokens),
            "prefix_token_coverage": 0.0,
            "suffix_token_coverage": 0.0,
            "prefix_missing_token_count": len(expected_tokens),
            "suffix_missing_token_count": len(expected_tokens),
            "leading_missing_token_count": len(expected_tokens),
            "leading_extra_token_count": len(observed_tokens),
            "leading_substitution_count": 0,
            "trailing_missing_token_count": len(expected_tokens),
            "trailing_extra_token_count": len(observed_tokens),
            "trailing_substitution_count": 0,
        }

    operations = _ordered_token_edit_operations(
        expected_tokens,
        observed_tokens,
        token_min_similarity=token_min_similarity,
    )
    matched_count = operations.count("match")
    boundary_size = min(
        len(expected_tokens),
        max(1, int(boundary_window_tokens)),
    )
    observed_boundary_size = min(
        len(observed_tokens),
        boundary_size + max(0, int(boundary_observed_slack_tokens)),
    )
    prefix_match_count = _ordered_token_edit_operations(
        expected_tokens[:boundary_size],
        observed_tokens[:observed_boundary_size],
        token_min_similarity=token_min_similarity,
    ).count("match")
    suffix_match_count = _ordered_token_edit_operations(
        expected_tokens[-boundary_size:],
        observed_tokens[-observed_boundary_size:],
        token_min_similarity=token_min_similarity,
    ).count("match")
    boundary_edits = _boundary_edit_counts(operations)
    return {
        "ordered_similarity": fuzz.ratio(
            expected.normalized,
            observed.normalized,
        ),
        "token_coverage": matched_count / len(expected_tokens),
        "token_precision": matched_count / len(observed_tokens),
        "extra_word_count": len(observed_tokens) - matched_count,
        "prefix_token_coverage": prefix_match_count / boundary_size,
        "suffix_token_coverage": suffix_match_count / boundary_size,
        "prefix_missing_token_count": boundary_size - prefix_match_count,
        "suffix_missing_token_count": boundary_size - suffix_match_count,
        **boundary_edits,
    }


def script_clauses(value: str) -> list[str]:
    clauses = [
        clause
        for clause in (
            part.strip()
            for part in re.split(r"[.!?]+", verbal_script_text(value))
        )
        if normalize_text(clause)
    ]
    hesitation_indexes = [
        index
        for index, clause in enumerate(clauses)
        if _text_features(clause).tokens == ("hesitation",)
    ]
    if not hesitation_indexes:
        return clauses

    # Ellipses normally separate script clauses, but actors also use them to
    # spell out a hesitation: "I... err... I misspoke." Treat that whole
    # performed phrase as one clause while preserving ordinary ellipsis
    # boundaries such as "Maybe over here... No, nothing."
    ranges = [
        (
            max(0, index - 1),
            min(len(clauses) - 1, index + 1),
        )
        for index in hesitation_indexes
    ]
    merged_ranges: list[list[int]] = []
    for start, end in ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1][1] = max(merged_ranges[-1][1], end)
        else:
            merged_ranges.append([start, end])
    result = []
    clause_index = 0
    for start, end in merged_ranges:
        result.extend(clauses[clause_index:start])
        result.append(" ".join(clauses[start : end + 1]))
        clause_index = end + 1
    result.extend(clauses[clause_index:])
    return result


def sentence_fidelity(
    expected: str,
    observed: str,
    *,
    token_min_similarity: float = 78.0,
    minimum_clause_score: float = 55.0,
    short_clause_max_words: int = 4,
    short_clause_min_token_coverage: float = 1.0,
) -> dict[str, Any]:
    return _sentence_fidelity_features(
        tuple(script_clauses(expected)),
        _text_features(observed),
        token_min_similarity=token_min_similarity,
        minimum_clause_score=minimum_clause_score,
        short_clause_max_words=short_clause_max_words,
        short_clause_min_token_coverage=short_clause_min_token_coverage,
    )


def _sentence_fidelity_features(
    clauses: tuple[str, ...],
    observed: TextFeatures,
    *,
    token_min_similarity: float,
    minimum_clause_score: float,
    short_clause_max_words: int,
    short_clause_min_token_coverage: float,
) -> dict[str, Any]:
    observed_normalized = observed.normalized
    clause_scores = []
    clause_token_coverages = []
    clause_word_counts = []
    clause_feature_list = [_text_features(clause) for clause in clauses]
    expected_tokens = [
        token
        for clause_features in clause_feature_list
        for token in clause_features.tokens
    ]
    observed_positions_by_expected = {
        int(expected_index): int(observed_index)
        for operation, expected_index, observed_index in (
            _ordered_token_alignment(
                expected_tokens,
                list(observed.tokens),
                token_min_similarity=token_min_similarity,
            )
        )
        if (
            operation == "match"
            and expected_index is not None
            and observed_index is not None
        )
    }
    clause_positions = []
    expected_offset = 0
    for clause, clause_features in zip(clauses, clause_feature_list):
        clause_fidelity = _transcript_fidelity_features(
            clause_features,
            observed,
            token_min_similarity=token_min_similarity,
            boundary_window_tokens=4,
            boundary_observed_slack_tokens=2,
        )
        partial_score = (
            fuzz.partial_ratio(
                clause_features.normalized,
                observed_normalized,
            )
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
        matched_positions = [
            observed_positions_by_expected[expected_index]
            for expected_index in range(
                expected_offset,
                expected_offset + len(clause_features.tokens),
            )
            if expected_index in observed_positions_by_expected
        ]
        clause_positions.append(
            min(matched_positions) if matched_positions else -1
        )
        expected_offset += len(clause_features.tokens)
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
            left >= 0 and right >= 0 and left <= right
            for left, right in zip(
                clause_positions,
                clause_positions[1:],
            )
        ),
        "minimum_clause_score": minimum_score,
        "missing_clause_count": sum(missing_clauses),
    }


@dataclass(frozen=True)
class ScriptLineFeatures:
    text: TextFeatures
    clauses: tuple[str, ...]
    word_count: int


class TranscriptEvaluator:
    """Session-local cache for normalized text and line/transcript scoring."""

    def __init__(
        self,
        lines: list[dict[str, Any]],
        settings: Mapping[str, Any] | AlignmentSettings,
    ) -> None:
        self.settings = AlignmentSettings.from_value(settings)
        self.lines = lines
        self.line_features = []
        for line in lines:
            text = _text_features(
                str(line["line"]),
                remove_parenthetical_cues=True,
            )
            self.line_features.append(
                ScriptLineFeatures(
                    text=text,
                    clauses=tuple(script_clauses(str(line["line"]))),
                    word_count=len(text.tokens),
                )
            )
        self._observed_features: dict[str, TextFeatures] = {}
        self._match_cache: OrderedDict[
            tuple[int, str],
            float,
        ] = OrderedDict()
        self._evaluation_cache: dict[
            tuple[int, str],
            TranscriptEvaluation,
        ] = {}

    def observed_features(self, observed: str) -> TextFeatures:
        cached = self._observed_features.get(observed)
        if cached is None:
            cached = _text_features(observed)
            self._observed_features[observed] = cached
        return cached

    def match(self, line_index: int, observed: str) -> float:
        observed_features = self.observed_features(observed)
        key = (line_index, observed_features.normalized)
        cached = self._match_cache.get(key)
        if cached is None:
            cached = _text_similarity_features(
                self.line_features[line_index].text,
                observed_features,
            )
            self._match_cache[key] = cached
            if len(self._match_cache) > 50_000:
                self._match_cache.popitem(last=False)
        else:
            self._match_cache.move_to_end(key)
        return cached

    def evaluate(
        self,
        line_index: int,
        observed: str,
    ) -> TranscriptEvaluation:
        observed_features = self.observed_features(observed)
        key = (line_index, observed_features.normalized)
        cached = self._evaluation_cache.get(key)
        if cached is not None:
            return cached

        line_features = self.line_features[line_index]
        fidelity = _transcript_fidelity_features(
            line_features.text,
            observed_features,
            token_min_similarity=float(
                self.settings["fidelity_token_min_similarity"]
            ),
            boundary_window_tokens=int(
                self.settings["reliable_boundary_window_tokens"]
            ),
            boundary_observed_slack_tokens=int(
                self.settings["reliable_boundary_observed_slack_tokens"]
            ),
        )
        sentence = _sentence_fidelity_features(
            line_features.clauses,
            observed_features,
            token_min_similarity=float(
                self.settings["fidelity_token_min_similarity"]
            ),
            minimum_clause_score=float(
                self.settings["reliable_min_clause_score"]
            ),
            short_clause_max_words=int(
                self.settings["reliable_short_clause_max_words"]
            ),
            short_clause_min_token_coverage=float(
                self.settings[
                    "reliable_short_clause_min_token_coverage"
                ]
            ),
        )
        cached = TranscriptEvaluation(
            match_score=self.match(line_index, observed),
            fidelity=fidelity,
            sentence=sentence,
        )
        self._evaluation_cache[key] = cached
        return cached


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


class SpanCatalog:
    """Enumerate valid contiguous spans once and cache transcript previews."""

    def __init__(
        self,
        segments: list[dict[str, Any]],
        settings: Mapping[str, Any] | AlignmentSettings,
        *,
        transcription: dict[str, Any] | None = None,
    ) -> None:
        self.segments = segments
        self.settings = AlignmentSettings.from_value(settings)
        self.transcription = transcription
        self.word_index = _transcription_word_index(transcription)
        self.maximum_segments = max(
            int(self.settings["max_merge_segments"]),
            int(self.settings["fragment_join_max_segments"]),
        )
        self._base_previews: dict[tuple[int, int], dict[str, Any]] = {}
        self._previews: dict[
            tuple[int, int],
            list[dict[str, Any]],
        ] = {}
        self._counts_by_start: list[list[int]] = [
            [] for _ in range(len(segments))
        ]
        for start_index in range(len(segments)):
            for count in range(1, self.maximum_segments + 1):
                if not _valid_span(
                    segments,
                    start_index,
                    count,
                    max_merge_gap_seconds=float(
                        self.settings["max_merge_gap_seconds"]
                    ),
                    max_span_seconds=float(
                        self.settings["max_span_seconds"]
                    ),
                ):
                    break
                self._counts_by_start[start_index].append(count)

    def is_valid(self, start_index: int, count: int) -> bool:
        return (
            0 <= start_index < len(self._counts_by_start)
            and count in self._counts_by_start[start_index]
        )

    def counts_from(
        self,
        start_index: int,
        *,
        maximum: int | None = None,
    ) -> list[int]:
        counts = self._counts_by_start[start_index]
        if maximum is None:
            return counts
        return [count for count in counts if count <= maximum]

    def base_preview(
        self,
        start_index: int,
        count: int,
    ) -> dict[str, Any]:
        key = (start_index, count)
        preview = self._base_previews.get(key)
        if preview is None:
            if not self.is_valid(start_index, count):
                raise KeyError(f"Invalid span: {key}")
            preview = _span_preview(self.segments, start_index, count)
            preview["transcript_source"] = "segment_asr_span"
            self._base_previews[key] = preview
        return preview

    def transcript_previews(
        self,
        start_index: int,
        count: int,
    ) -> list[dict[str, Any]]:
        key = (start_index, count)
        previews = self._previews.get(key)
        if previews is not None:
            return previews
        base = self.base_preview(start_index, count)
        previews = [base]
        if self.transcription:
            timestamp_span = _span_preview_with_transcription(
                self.segments,
                start_index,
                count,
                transcription=self.transcription,
                settings=dict(self.settings),
                word_index=self.word_index,
            )
            if normalize_text(timestamp_span["transcript"]) != normalize_text(
                base["transcript"]
            ):
                previews.append(timestamp_span)
        self._previews[key] = previews
        return previews

    def transcription_preview_for_bounds(
        self,
        *,
        start_seconds: float,
        end_seconds: float,
    ) -> dict[str, Any] | None:
        if not self.transcription:
            return None
        return _transcription_preview_for_bounds(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            transcription=self.transcription,
            settings=dict(self.settings),
            word_index=self.word_index,
        )


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

    timestamp_span = _transcription_preview_for_bounds(
        start_seconds=float(span["start_seconds"]),
        end_seconds=float(span["end_seconds"]),
        transcription=transcription,
        settings=settings,
        word_index=word_index,
    )
    if timestamp_span is None:
        return span
    span.update(timestamp_span)
    return span


def _transcription_preview_for_bounds(
    *,
    start_seconds: float,
    end_seconds: float,
    transcription: dict[str, Any],
    settings: Mapping[str, Any],
    word_index: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a recording-ASR preview for arbitrary sample-accurate bounds."""

    minimum_overlap_seconds = float(
        settings.get("span_word_min_overlap_seconds", 0.20)
    )
    minimum_overlap_fraction = float(
        settings.get("span_word_min_overlap_fraction", 0.20)
    )
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
        return None

    words.sort(key=lambda item: (item[0], item[1]))
    transcript = "".join(
        str(word.get("word") or "") for _, _, word in words
    ).strip()
    probabilities = [
        float(word["probability"])
        for _, _, word in words
        if word.get("probability") is not None
    ]
    return {
        "transcript": transcript,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "asr_probability": (
            sum(probabilities) / len(probabilities)
            if probabilities
            else None
        ),
        "transcript_source": "session_word_span",
    }


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
    *,
    expected_word_count: int | None = None,
) -> float:
    expected_words = max(
        1,
        (
            int(expected_word_count)
            if expected_word_count is not None
            else len(
                normalize_spoken_text(
                    line["line"],
                    remove_parenthetical_cues=True,
                ).split()
            )
        ),
    )
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
    settings: Mapping[str, Any] | AlignmentSettings,
    *,
    evaluator: TranscriptEvaluator | None = None,
    span_catalog: SpanCatalog | None = None,
) -> list[dict[str, Any]]:
    """Match non-overlapping audio spans without requiring script order.

    Each selected span has one primary line plus a small set of alternative
    line matches. Multiple separate spans may select the same line, which is
    how repeated takes are represented.
    """

    settings = AlignmentSettings.from_value(settings)
    all_lines = lines
    lines, source_indexes = _text_matchable_lines(lines)
    if not segments or not lines:
        return []
    evaluator = evaluator or TranscriptEvaluator(all_lines, settings)
    span_catalog = span_catalog or SpanCatalog(segments, settings)

    segment_count = len(segments)
    line_count = len(lines)
    max_merge = int(settings.get("max_merge_segments", 8))
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
        check_processing_cancelled()
        for count in span_catalog.counts_from(
            segment_index,
            maximum=max_merge,
        ):
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
            span = span_catalog.base_preview(segment_index, count)
            line_matches = []
            for line_index, line in enumerate(lines):
                match_score = evaluator.match(
                    source_indexes[line_index],
                    span["transcript"],
                )
                duration_score = _duration_plausibility(
                    line,
                    span,
                    expected_word_count=evaluator.line_features[
                        source_indexes[line_index]
                    ].word_count,
                )
                order_score = (
                    _order_hint(
                        segment_index=segment_index,
                        segment_count=segment_count,
                        line_index=line_index,
                        line_count=line_count,
                    )
                    if order_weight
                    else 0.0
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
            pure_scores = heapq.nlargest(
                2,
                (
                    (
                        match["match_score"],
                        int(match["line_index"]),
                    )
                    for match in line_matches
                ),
            )
            line_matches = heapq.nlargest(
                top_k,
                line_matches,
                key=lambda match: (
                    match["ranking_score"],
                    match["match_score"],
                    -match["line_index"],
                ),
            )
            best_pure = pure_scores[0] if pure_scores else (0.0, -1)
            second_pure = (
                pure_scores[1] if len(pure_scores) > 1 else (0.0, -1)
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
                "confidence_margin": primary["confidence_margin"],
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
    settings: Mapping[str, Any] | AlignmentSettings,
) -> None:
    settings = AlignmentSettings.from_value(settings)
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
        duplicate_indexes[
            normalize_spoken_text(
                line["line"],
                remove_parenthetical_cues=True,
            )
        ].append(line_index)
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
            if (
                normalize_spoken_text(
                    lines[primary_index]["line"],
                    remove_parenthetical_cues=True,
                )
                != normalized
            ):
                continue
            matching_actions.append(action)
        matching_actions.sort(key=lambda action: int(action["start_index"]))
        if not matching_actions:
            continue

        expected_word_count = len(normalized.split())
        anchor_minimum_score = float(
            settings.get(
                (
                    "short_line_min_score"
                    if expected_word_count <= 3
                    else "reliable_min_score"
                ),
                88.0 if expected_word_count <= 3 else 72.0,
            )
        )
        anchor_actions = [
            action
            for action in matching_actions
            if float(action.get("match_score", 0.0))
            >= anchor_minimum_score
        ]
        if not anchor_actions:
            continue

        clusters: list[list[dict[str, Any]]] = []
        previous_end = None
        for action in anchor_actions:
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
        assignment_groups = clusters
        if len(assignment_groups) < len(indexes):
            # Identical script rows are acoustically indistinguishable. When
            # all nearby takes form one cluster but there are still enough
            # distinct spans, weak order should distribute the spans instead
            # of leaving every later duplicate row missing.
            assignment_groups = [
                [action] for action in anchor_actions
            ]

        assigned_anchor_rows: dict[int, int] = {}

        def assign_action(
            action: dict[str, Any],
            duplicate_index: int,
        ) -> None:
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

        for cluster_index, cluster in enumerate(assignment_groups):
            if len(assignment_groups) == 1:
                duplicate_index = indexes[0]
            else:
                duplicate_position = round(
                    cluster_index
                    * (len(indexes) - 1)
                    / (len(assignment_groups) - 1)
                )
                duplicate_index = indexes[duplicate_position]
            for action in cluster:
                assign_action(action, duplicate_index)
                assigned_anchor_rows[id(action)] = duplicate_index

        weak_actions = [
            action
            for action in matching_actions
            if id(action) not in assigned_anchor_rows
        ]
        for action in weak_actions:
            action_start = float(
                base_segments[int(action["start_index"])]["start_seconds"]
            )
            nearest_anchor = min(
                anchor_actions,
                key=lambda anchor: abs(
                    float(
                        base_segments[int(anchor["start_index"])][
                            "start_seconds"
                        ]
                    )
                    - action_start
                ),
            )
            assign_action(
                action,
                assigned_anchor_rows[id(nearest_anchor)],
            )


def _segment_transcript(segment: Mapping[str, Any]) -> str:
    primary_asr = ((segment.get("segment_asr") or {}).get("primary") or {})
    return str(
        primary_asr.get("transcript")
        or segment.get("transcript")
        or ""
    ).strip()


def _segment_duration(segment: Mapping[str, Any]) -> float:
    return float(
        (segment.get("metrics") or {}).get("duration_seconds")
        or (
            float(segment.get("end_seconds", 0.0))
            - float(segment.get("start_seconds", 0.0))
        )
    )


def _is_likely_edge_vocalization_segment(
    segment: Mapping[str, Any],
    *,
    settings: Mapping[str, Any] | AlignmentSettings,
) -> bool:
    settings = AlignmentSettings.from_value(settings)
    if _segment_duration(segment) > float(
        settings.get("edge_cue_extension_max_segment_seconds", 5.0)
    ):
        return False
    transcript = _segment_transcript(segment)
    if transcript:
        return _is_paralinguistic_transcript(transcript)
    rms = (segment.get("metrics") or {}).get("rms_dbfs")
    return bool(
        rms is not None
        and float(rms)
        >= float(settings.get("untranscribed_merge_min_rms_dbfs", -45.0))
    )


def _boundary_segment_is_removable(
    segment: Mapping[str, Any],
    *,
    line_text: str,
) -> bool:
    transcript = _segment_transcript(segment)
    if not _is_paralinguistic_transcript(transcript):
        return False
    expected_tokens = set(_text_features(verbal_script_text(line_text)).tokens)
    segment_tokens = set(_text_features(transcript).tokens)
    return not expected_tokens.intersection(segment_tokens)


def _action_for_rebounded_span(
    source_action: Mapping[str, Any],
    *,
    start_index: int,
    count: int,
    lines: list[dict[str, Any]],
    settings: Mapping[str, Any] | AlignmentSettings,
    evaluator: TranscriptEvaluator,
    span_catalog: SpanCatalog,
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not span_catalog.is_valid(start_index, count):
        return None
    line_index = int(source_action["line_index"])
    previews = [
        preview
        for preview in span_catalog.transcript_previews(start_index, count)
        if str(preview.get("transcript") or "").strip()
    ]
    if not previews:
        return None
    preview = max(
        previews,
        key=lambda item: evaluator.match(
            line_index,
            str(item.get("transcript") or ""),
        ),
    )
    transcript = str(preview["transcript"])
    match_score = evaluator.match(line_index, transcript)
    if match_score < float(
        AlignmentSettings.from_value(settings).get(
            "candidate_min_score",
            45.0,
        )
    ):
        return None
    other_score = max(
        (
            evaluator.match(other_index, transcript)
            for other_index, other_line in enumerate(lines)
            if other_index != line_index
            and not is_vocalization_script(other_line["line"])
        ),
        default=0.0,
    )
    duration_score = _duration_plausibility(
        lines[line_index],
        preview,
        expected_word_count=evaluator.line_features[line_index].word_count,
    )
    action = dict(source_action)
    for key in (
        "trim_start_sample",
        "trim_end_sample",
        "trim_word_start",
        "trim_word_end",
        "intra_segment_trim",
        "trimmed_edge_join",
        "fragment_join",
        "fragment_join_fallback",
        "fragment_join_provisional",
        "forced_review_reason",
    ):
        action.pop(key, None)
    action.update(
        {
            "type": "assigned",
            "start_index": start_index,
            "count": count,
            "line_index": line_index,
            "match_score": match_score,
            "confidence_margin": match_score - other_score,
            "transcript": transcript,
            "transcript_source": preview.get(
                "transcript_source",
                "base_segments",
            ),
            "duration_plausibility": duration_score,
            "top_matches": [
                {
                    "line_index": line_index,
                    "match_score": match_score,
                    "ranking_score": match_score,
                    "duration_plausibility": duration_score,
                    "order_hint": float(
                        source_action.get("order_hint", 0.0)
                    ),
                    "confidence_margin": match_score - other_score,
                }
            ],
            **dict(metadata),
        }
    )
    return action


def _boundary_noise_cleanup_actions(
    actions: list[dict[str, Any]],
    *,
    lines: list[dict[str, Any]],
    base_segments: list[dict[str, Any]],
    settings: Mapping[str, Any] | AlignmentSettings,
    evaluator: TranscriptEvaluator,
    span_catalog: SpanCatalog,
) -> list[dict[str, Any]]:
    """Recover clean spans after a standalone boundary vocalization was merged."""

    settings = AlignmentSettings.from_value(settings)
    if not bool(settings.get("boundary_noise_cleanup_enabled", True)):
        return []
    occupied = {
        (
            int(action["line_index"]),
            int(action["start_index"]),
            int(action["count"]),
        )
        for action in actions
    }
    recovered = []
    for source_action in actions:
        line_index = int(source_action["line_index"])
        line_text = str(lines[line_index]["line"])
        source_evaluation = evaluator.evaluate(
            line_index,
            str(source_action.get("transcript") or ""),
        )
        leading_cue, trailing_cue = _script_edge_performance_cues(line_text)
        queue = [
            (
                int(source_action["start_index"]),
                int(source_action["count"]),
            )
        ]
        visited = set(queue)
        while queue:
            start_index, count = queue.pop(0)
            if count <= 1:
                continue
            proposals = []
            if (
                not leading_cue
                and _boundary_segment_is_removable(
                    base_segments[start_index],
                    line_text=line_text,
                )
            ):
                proposals.append((start_index + 1, count - 1))
            end_index = start_index + count - 1
            if (
                not trailing_cue
                and _boundary_segment_is_removable(
                    base_segments[end_index],
                    line_text=line_text,
                )
            ):
                proposals.append((start_index, count - 1))
            for new_start, new_count in proposals:
                local_key = (new_start, new_count)
                if local_key in visited:
                    continue
                visited.add(local_key)
                queue.append(local_key)
                key = (line_index, new_start, new_count)
                if key in occupied:
                    continue
                action = _action_for_rebounded_span(
                    source_action,
                    start_index=new_start,
                    count=new_count,
                    lines=lines,
                    settings=settings,
                    evaluator=evaluator,
                    span_catalog=span_catalog,
                    metadata={"boundary_noise_cleanup": True},
                )
                if action is None:
                    continue
                cleaned_evaluation = evaluator.evaluate(
                    line_index,
                    str(action["transcript"]),
                )
                fidelity = cleaned_evaluation.fidelity
                sentence = cleaned_evaluation.sentence
                if (
                    cleaned_evaluation.match_score
                    < float(
                        settings.get(
                            "boundary_noise_cleanup_min_match_score",
                            85.0,
                        )
                    )
                    or float(fidelity["token_coverage"]) < 0.85
                    or float(fidelity["token_precision"]) < 0.85
                    or int(fidelity["leading_extra_token_count"]) > 0
                    or int(fidelity["trailing_extra_token_count"]) > 0
                    or int(sentence["missing_clause_count"]) > 0
                    or not bool(sentence["clauses_in_order"])
                ):
                    continue
                if (
                    int(fidelity["extra_word_count"])
                    >= int(
                        source_evaluation.fidelity["extra_word_count"]
                    )
                    and float(fidelity["token_precision"])
                    <= float(
                        source_evaluation.fidelity["token_precision"]
                    )
                ):
                    continue
                occupied.add(key)
                recovered.append(action)
    return recovered


def _complete_subspan_recovery_actions(
    actions: list[dict[str, Any]],
    *,
    lines: list[dict[str, Any]],
    settings: Mapping[str, Any] | AlignmentSettings,
    evaluator: TranscriptEvaluator,
    span_catalog: SpanCatalog,
) -> list[dict[str, Any]]:
    """Recover a complete textual subspan hidden by a selected merged span."""

    settings = AlignmentSettings.from_value(settings)
    if not bool(settings.get("complete_subspan_recovery_enabled", True)):
        return []
    minimum_match = float(
        settings.get("boundary_noise_cleanup_min_match_score", 85.0)
    )
    occupied = {
        (
            int(action["line_index"]),
            int(action["start_index"]),
            int(action["count"]),
        )
        for action in actions
    }
    recovered = []
    for source_action in actions:
        check_processing_cancelled()
        source_start = int(source_action["start_index"])
        source_count = int(source_action["count"])
        if source_count <= 1:
            continue
        line_index = int(source_action["line_index"])
        if is_vocalization_script(str(lines[line_index]["line"])):
            continue
        source_end = source_start + source_count
        for new_start in range(source_start, source_end):
            for new_end in range(new_start + 1, source_end + 1):
                new_count = new_end - new_start
                if new_start == source_start and new_count == source_count:
                    continue
                key = (line_index, new_start, new_count)
                if key in occupied:
                    continue
                action = _action_for_rebounded_span(
                    source_action,
                    start_index=new_start,
                    count=new_count,
                    lines=lines,
                    settings=settings,
                    evaluator=evaluator,
                    span_catalog=span_catalog,
                    metadata={"complete_subspan_recovery": True},
                )
                if action is None:
                    continue
                evaluation = evaluator.evaluate(
                    line_index,
                    str(action["transcript"]),
                )
                fidelity = evaluation.fidelity
                sentence = evaluation.sentence
                if (
                    evaluation.match_score < minimum_match
                    or float(fidelity["token_coverage"]) < 0.95
                    or float(fidelity["token_precision"]) < 0.95
                    or int(fidelity["leading_extra_token_count"]) > 0
                    or int(fidelity["trailing_extra_token_count"]) > 0
                    or int(sentence["missing_clause_count"]) > 0
                    or not bool(sentence["clauses_in_order"])
                ):
                    continue
                occupied.add(key)
                recovered.append(action)
    return recovered


def _edge_vocalization_extension_actions(
    actions: list[dict[str, Any]],
    *,
    lines: list[dict[str, Any]],
    base_segments: list[dict[str, Any]],
    settings: Mapping[str, Any] | AlignmentSettings,
    evaluator: TranscriptEvaluator,
    span_catalog: SpanCatalog,
) -> list[dict[str, Any]]:
    """Attach a neighboring vocalization segment required by an edge cue."""

    settings = AlignmentSettings.from_value(settings)
    if not bool(settings.get("edge_cue_extension_enabled", True)):
        return []
    maximum_gap = float(
        settings.get("edge_cue_extension_max_gap_seconds", 0.40)
    )
    occupied = {
        (
            int(action["line_index"]),
            int(action["start_index"]),
            int(action["count"]),
        )
        for action in actions
    }
    recovered = []
    for source_action in actions:
        line_index = int(source_action["line_index"])
        line_text = str(lines[line_index]["line"])
        leading_cue, trailing_cue = _script_edge_performance_cues(line_text)
        if not leading_cue and not trailing_cue:
            continue
        fidelity = evaluator.evaluate(
            line_index,
            str(source_action.get("transcript") or ""),
        ).fidelity
        start_index = int(source_action["start_index"])
        count = int(source_action["count"])
        end_index = start_index + count - 1
        new_start = start_index
        new_end = end_index
        if (
            leading_cue
            and int(fidelity["leading_extra_token_count"]) == 0
            and start_index > 0
        ):
            neighbor = base_segments[start_index - 1]
            gap = (
                float(base_segments[start_index]["start_seconds"])
                - float(neighbor["end_seconds"])
            )
            if (
                gap <= maximum_gap
                and _is_likely_edge_vocalization_segment(
                    neighbor,
                    settings=settings,
                )
            ):
                new_start -= 1
        if (
            trailing_cue
            and int(fidelity["trailing_extra_token_count"]) == 0
            and end_index + 1 < len(base_segments)
        ):
            neighbor = base_segments[end_index + 1]
            gap = (
                float(neighbor["start_seconds"])
                - float(base_segments[end_index]["end_seconds"])
            )
            if (
                gap <= maximum_gap
                and _is_likely_edge_vocalization_segment(
                    neighbor,
                    settings=settings,
                )
            ):
                new_end += 1
        if new_start == start_index and new_end == end_index:
            continue
        new_count = new_end - new_start + 1
        key = (line_index, new_start, new_count)
        if key in occupied:
            continue
        action = _action_for_rebounded_span(
            source_action,
            start_index=new_start,
            count=new_count,
            lines=lines,
            settings=settings,
            evaluator=evaluator,
            span_catalog=span_catalog,
            metadata={
                "edge_vocalization_extension": True,
                "forced_review_reason": "EDGE_VOCALIZATION_UNVERIFIED",
            },
        )
        if action is None:
            continue
        occupied.add(key)
        recovered.append(action)
    return recovered


def _segment_sample_rate(segment: Mapping[str, Any]) -> int | None:
    if (
        segment.get("start_sample") is None
        or segment.get("end_sample") is None
    ):
        return None
    duration = (
        float(segment.get("end_seconds", 0.0))
        - float(segment.get("start_seconds", 0.0))
    )
    if duration <= 0.0:
        return None
    sample_count = int(segment["end_sample"]) - int(segment["start_sample"])
    if sample_count <= 0:
        return None
    return max(1, round(sample_count / duration))


def _word_gap_boundaries(
    segment: Mapping[str, Any],
    *,
    minimum_gap: float,
    maximum_boundaries: int,
    segmentation_settings: Mapping[str, Any] | None = None,
    project_dir: Path | None = None,
) -> list[tuple[int, float]]:
    segmentation_settings = dict(segmentation_settings or {})
    primary_asr = ((segment.get("segment_asr") or {}).get("primary") or {})
    words = [
        word
        for word in primary_asr.get("words") or []
        if str(word.get("word") or "").strip()
        and word.get("start") is not None
        and word.get("end") is not None
    ]
    boundaries: list[tuple[int, float, float]] = []
    for word_index, (left, right) in enumerate(zip(words, words[1:])):
        left_end = float(left["end"])
        right_start = float(right["start"])
        gap = right_start - left_end
        if gap >= minimum_gap:
            boundary = (left_end + right_start) / 2.0
            sample_rate = _segment_sample_rate(segment)
            if (
                project_dir is not None
                and sample_rate is not None
                and segment.get("file")
                and bool(
                    segmentation_settings.get(
                        "word_split_snap_enabled",
                        True,
                    )
                )
            ):
                proposed_sample = round(boundary * sample_rate)
                snapped_sample = quietest_pcm_boundary(
                    resolve_project_path(project_dir, str(segment["file"])),
                    proposed_sample=proposed_sample,
                    minimum_sample=round(left_end * sample_rate),
                    maximum_sample=round(right_start * sample_rate),
                    search_seconds=min(
                        float(
                            segmentation_settings.get(
                                "word_split_snap_search_seconds",
                                0.20,
                            )
                        ),
                        gap / 2.0,
                    ),
                    window_seconds=float(
                        segmentation_settings.get(
                            "word_split_snap_window_seconds",
                            0.02,
                        )
                    ),
                    maximum_rms_dbfs=float(
                        segmentation_settings.get(
                            "word_split_snap_max_rms_dbfs",
                            -42.0,
                        )
                    ),
                    require_quiet=True,
                )
                if snapped_sample is None:
                    continue
                boundary = snapped_sample / sample_rate
            boundaries.append(
                (
                    word_index,
                    boundary,
                    gap,
                )
            )
    boundaries.sort(key=lambda item: item[2], reverse=True)
    selected = boundaries[:maximum_boundaries]
    selected.sort(key=lambda item: item[0])
    return [(word_index, offset) for word_index, offset, _ in selected]


def _trimmed_edge_span_previews(
    *,
    start_index: int,
    count: int,
    base_segments: list[dict[str, Any]],
    settings: Mapping[str, Any] | AlignmentSettings,
    span_catalog: SpanCatalog,
    project_dir: Path | None = None,
    segmentation_settings: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Preview joins whose first or last base segment is split at a word gap."""

    if count < 2:
        return []
    selected = base_segments[start_index : start_index + count]
    if len(selected) != count:
        return []
    first = selected[0]
    last = selected[-1]
    first_rate = _segment_sample_rate(first)
    last_rate = _segment_sample_rate(last)
    if first_rate is None or last_rate is None:
        return []

    settings = AlignmentSettings.from_value(settings)
    minimum_gap = min(
        float(settings.get("edge_trim_min_gap_seconds", 0.30)),
        float(
            settings.get(
                "edge_trim_fallback_min_gap_seconds",
                0.10,
            )
        ),
    )
    maximum_boundaries = max(
        1,
        int(settings.get("intra_segment_trim_max_actions_per_segment", 3)),
    )
    first_words = [
        word
        for word in (
            ((first.get("segment_asr") or {}).get("primary") or {}).get(
                "words"
            )
            or []
        )
        if str(word.get("word") or "").strip()
        and word.get("start") is not None
        and word.get("end") is not None
    ]
    last_words = [
        word
        for word in (
            ((last.get("segment_asr") or {}).get("primary") or {}).get(
                "words"
            )
            or []
        )
        if str(word.get("word") or "").strip()
        and word.get("start") is not None
        and word.get("end") is not None
    ]
    first_boundaries = _word_gap_boundaries(
        first,
        minimum_gap=minimum_gap,
        maximum_boundaries=maximum_boundaries,
        segmentation_settings=segmentation_settings,
        project_dir=project_dir,
    )
    last_boundaries = _word_gap_boundaries(
        last,
        minimum_gap=minimum_gap,
        maximum_boundaries=maximum_boundaries,
        segmentation_settings=segmentation_settings,
        project_dir=project_dir,
    )
    full_start_sample = int(first["start_sample"])
    full_end_sample = int(last["end_sample"])
    previews = []
    seen: set[tuple[int, int, str]] = set()

    def append_preview(
        *,
        transcript: str,
        start_sample: int,
        end_sample: int,
    ) -> None:
        transcript = transcript.strip()
        if not transcript or end_sample <= start_sample:
            return
        key = (
            start_sample,
            end_sample,
            normalize_text(transcript),
        )
        if key in seen:
            return
        seen.add(key)
        start_seconds = start_sample / first_rate
        end_seconds = end_sample / last_rate
        preview = {
            "start_index": start_index,
            "count": count,
            "transcript": transcript,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "trim_start_sample": start_sample,
            "trim_end_sample": end_sample,
            "transcript_source": "segment_asr_trimmed_edge_span",
        }
        previews.append(preview)
        timestamp_preview = span_catalog.transcription_preview_for_bounds(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        if timestamp_preview:
            timestamp_key = (
                start_sample,
                end_sample,
                normalize_text(str(timestamp_preview["transcript"])),
            )
            if timestamp_key not in seen:
                seen.add(timestamp_key)
                previews.append(
                    {
                        **preview,
                        **timestamp_preview,
                        "trim_start_sample": start_sample,
                        "trim_end_sample": end_sample,
                        "transcript_source": (
                            "session_word_trimmed_edge_span"
                        ),
                    }
                )

    trailing_full_text = " ".join(
        str(segment.get("transcript") or "").strip()
        for segment in selected[1:]
        if str(segment.get("transcript") or "").strip()
    )
    for word_index, offset in first_boundaries:
        suffix_text = " ".join(
            str(word.get("word") or "").strip()
            for word in first_words[word_index + 1 :]
            if str(word.get("word") or "").strip()
        )
        append_preview(
            transcript=" ".join(
                part for part in (suffix_text, trailing_full_text) if part
            ),
            start_sample=max(
                full_start_sample,
                full_start_sample + round(offset * first_rate),
            ),
            end_sample=full_end_sample,
        )

    leading_full_text = " ".join(
        str(segment.get("transcript") or "").strip()
        for segment in selected[:-1]
        if str(segment.get("transcript") or "").strip()
    )
    for word_index, offset in last_boundaries:
        prefix_text = " ".join(
            str(word.get("word") or "").strip()
            for word in last_words[: word_index + 1]
            if str(word.get("word") or "").strip()
        )
        append_preview(
            transcript=" ".join(
                part for part in (leading_full_text, prefix_text) if part
            ),
            start_sample=full_start_sample,
            end_sample=min(
                full_end_sample,
                int(last["start_sample"]) + round(offset * last_rate),
            ),
        )
    return previews


def _multisentence_fragment_join_actions(
    actions: list[dict[str, Any]],
    *,
    lines: list[dict[str, Any]],
    base_segments: list[dict[str, Any]],
    settings: Mapping[str, Any] | AlignmentSettings,
    transcription: dict[str, Any] | None = None,
    evaluator: TranscriptEvaluator | None = None,
    span_catalog: SpanCatalog | None = None,
    project_dir: Path | None = None,
    segmentation_settings: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    settings = AlignmentSettings.from_value(settings)
    evaluator = evaluator or TranscriptEvaluator(lines, settings)
    span_catalog = span_catalog or SpanCatalog(
        base_segments,
        settings,
        transcription=transcription,
    )
    if not bool(settings.get("fragment_join_enabled", True)):
        return []
    secondary_seed_minimum_score = float(
        settings.get("fragment_join_secondary_seed_min_match_score", 80.0)
    )
    recovery_actions = list(actions)
    recovery_keys = {
        (
            int(action["line_index"]),
            int(action["start_index"]),
            int(action["count"]),
        )
        for action in recovery_actions
    }
    for action in actions:
        primary_line_index = int(action["line_index"])
        for match in action.get("top_matches") or []:
            line_index = int(match["line_index"])
            match_score = float(match.get("match_score", 0.0))
            key = (
                line_index,
                int(action["start_index"]),
                int(action["count"]),
            )
            if (
                line_index == primary_line_index
                or match_score < secondary_seed_minimum_score
                or key in recovery_keys
            ):
                continue
            secondary = dict(action)
            secondary.update(
                {
                    "line_index": line_index,
                    "match_score": match_score,
                    "duration_plausibility": float(
                        match.get(
                            "duration_plausibility",
                            action.get("duration_plausibility", 0.0),
                        )
                    ),
                    "order_hint": float(
                        match.get(
                            "order_hint",
                            action.get("order_hint", 0.0),
                        )
                    ),
                    "confidence_margin": float(
                        match.get("confidence_margin", 0.0)
                    ),
                    "fragment_secondary_seed": True,
                }
            )
            recovery_actions.append(secondary)
            recovery_keys.add(key)
    ordered_actions = sorted(
        recovery_actions,
        key=lambda action: (
            int(action["start_index"]),
            int(action["count"]),
        ),
    )
    # The primary and recovery aligners should grow together. The optional
    # fragment-specific value remains an override for existing projects, but
    # can no longer make recovery narrower than max_merge_segments.
    maximum_segments = max(
        2,
        int(settings.get("max_merge_segments", 8)),
        int(settings.get("fragment_join_max_segments", 0)),
    )
    maximum_actions_per_line = max(
        1,
        int(settings.get("fragment_join_max_actions", 10)),
    )
    fallback_actions_per_line = max(
        0,
        int(settings.get("fragment_join_fallback_max_actions", 2)),
    )
    fallback_minimum_match = float(
        settings.get("fragment_join_fallback_min_match_score", 90.0)
    )
    minimum_token_coverage = float(
        settings.get("fragment_join_min_token_coverage", 0.85)
    )
    minimum_token_precision = float(
        settings.get("fragment_join_min_token_precision", 0.83)
    )
    maximum_boundary_missing_tokens = max(
        0,
        int(settings.get("reliable_max_boundary_missing_tokens", 0)),
    )
    provisional_minimum_match = float(
        settings.get(
            "fragment_join_provisional_min_match_score",
            settings.get("candidate_min_score", 45.0),
        )
    )
    provisional_minimum_token_coverage = float(
        settings.get("fragment_join_provisional_min_token_coverage", 0.70)
    )
    provisional_minimum_ordered_score = float(
        settings.get("fragment_join_provisional_min_ordered_score", 55.0)
    )
    provisional_minimum_token_precision = float(
        settings.get("fragment_join_provisional_min_token_precision", 0.65)
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
    existing = {
        (
            int(action["line_index"]),
            int(action["start_index"]),
            int(action["count"]),
        )
        for action in recovery_actions
    }
    evaluated: set[tuple[int, int, int]] = set()
    joined = []

    def has_complete_boundaries(fidelity: dict[str, Any]) -> bool:
        return bool(
            int(fidelity["prefix_missing_token_count"])
            <= maximum_boundary_missing_tokens
            and int(fidelity["suffix_missing_token_count"])
            <= maximum_boundary_missing_tokens
            and int(fidelity["leading_missing_token_count"]) == 0
            and int(fidelity["leading_extra_token_count"]) == 0
            and int(fidelity["leading_substitution_count"]) == 0
            and int(fidelity["trailing_missing_token_count"]) == 0
            and int(fidelity["trailing_extra_token_count"]) == 0
            and int(fidelity["trailing_substitution_count"]) == 0
        )

    def score_preview(
        line_index: int,
        preview: dict[str, Any],
    ) -> dict[str, Any]:
        transcript = str(preview.get("transcript") or "")
        evaluation = evaluator.evaluate(line_index, transcript)
        return {
            "preview": preview,
            "match": evaluation.match_score,
            "fidelity": evaluation.fidelity,
            "sentence": evaluation.sentence,
        }

    def preview_quality(scored: dict[str, Any]) -> tuple[Any, ...]:
        sentence = scored["sentence"]
        fidelity = scored["fidelity"]
        return (
            int(sentence["missing_clause_count"]) == 0,
            bool(sentence["clauses_in_order"]),
            has_complete_boundaries(fidelity),
            min(
                float(fidelity["prefix_token_coverage"]),
                float(fidelity["suffix_token_coverage"]),
            ),
            float(fidelity["token_coverage"]),
            float(sentence["minimum_clause_score"]),
            float(scored["match"]),
            float(fidelity["token_precision"]),
            float(fidelity["ordered_similarity"]),
        )

    def is_strict_preview(scored: dict[str, Any]) -> bool:
        sentence = scored["sentence"]
        fidelity = scored["fidelity"]
        return bool(
            (
                sentence["clause_count"] < 2
                or (
                    sentence["missing_clause_count"] == 0
                    and sentence["clauses_in_order"]
                )
            )
            and float(fidelity["token_coverage"])
            >= minimum_token_coverage
            and float(fidelity["ordered_similarity"])
            >= minimum_ordered_score
            and float(fidelity["token_precision"])
            >= minimum_token_precision
            and has_complete_boundaries(fidelity)
        )

    def is_provisional_preview(
        scored: dict[str, Any],
        *,
        boundary_audio_rescue: bool,
    ) -> bool:
        sentence = scored["sentence"]
        fidelity = scored["fidelity"]
        return bool(
            float(scored["match"]) >= provisional_minimum_match
            and (
                float(fidelity["token_coverage"])
                >= provisional_minimum_token_coverage
                or (
                    boundary_audio_rescue
                    and float(fidelity["token_coverage"]) >= 0.50
                )
            )
            and float(fidelity["ordered_similarity"])
            >= provisional_minimum_ordered_score
            and (
                float(fidelity["token_precision"])
                >= provisional_minimum_token_precision
                or (
                    boundary_audio_rescue
                    and float(fidelity["token_precision"]) >= 0.50
                )
            )
            and (
                sentence["missing_clause_count"] > 0
                or bool(sentence["clauses_in_order"])
            )
        )

    def has_complete_script_coverage(fidelity: dict[str, Any]) -> bool:
        """Allow boundary extras while requiring every scripted edge token."""

        return bool(
            float(fidelity["token_coverage"]) >= 1.0
            and int(fidelity["leading_missing_token_count"]) == 0
            and int(fidelity["leading_substitution_count"]) == 0
            and int(fidelity["trailing_missing_token_count"]) == 0
            and int(fidelity["trailing_substitution_count"]) == 0
        )

    complete_action_keys: set[tuple[int, int, int]] = set()
    complete_action_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
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
            scored = score_preview(
                line_index,
                {"transcript": str(action.get("transcript") or "")},
            )
            expected_words = max(
                1,
                evaluator.line_features[line_index].word_count,
            )
            observed_text = str(action.get("transcript") or "")
            observed_words = len(
                normalize_spoken_text(observed_text).split()
            )
            length_ratio = observed_words / expected_words
            if (
                float(scored["match"]) >= minimum_complete_match
                and float(scored["fidelity"]["ordered_similarity"])
                >= minimum_complete_ordered
                and int(scored["sentence"]["missing_clause_count"]) == 0
                and bool(scored["sentence"]["clauses_in_order"])
                and has_complete_script_coverage(scored["fidelity"])
                and minimum_length_ratio
                <= length_ratio
                <= maximum_length_ratio
            ):
                action_start = int(action["start_index"])
                action_count = int(action["count"])
                complete_action_keys.add(
                    (line_index, action_start, action_count)
                )
                complete_action_ranges[line_index].append(
                    (action_start, action_start + action_count - 1)
                )

    for seed_action in ordered_actions:
        check_processing_cancelled()
        line_index = int(seed_action["line_index"])
        line = lines[line_index]
        seed_key = (
            line_index,
            int(seed_action["start_index"]),
            int(seed_action["count"]),
        )
        if (
            is_vocalization_script(line["line"])
            or seed_key in complete_action_keys
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
                if any(
                    start_index <= complete_end
                    and end_index >= complete_start
                    for complete_start, complete_end in complete_action_ranges.get(
                        line_index,
                        [],
                    )
                ):
                    continue
                if (
                    end_index < seed_start - neighbor_radius
                    or start_index > seed_end + neighbor_radius
                ):
                    continue
                key = (line_index, start_index, count)
                if key in evaluated:
                    continue
                span_already_exists = key in existing
                evaluated.add(key)
                if not span_catalog.is_valid(start_index, count):
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

                edge_previews = _trimmed_edge_span_previews(
                    start_index=start_index,
                    count=count,
                    base_segments=base_segments,
                    settings=settings,
                    span_catalog=span_catalog,
                    project_dir=project_dir,
                    segmentation_settings=segmentation_settings,
                )
                previews = (
                    edge_previews
                    if span_already_exists
                    else [
                        *span_catalog.transcript_previews(
                            start_index,
                            count,
                        ),
                        *edge_previews,
                    ]
                )
                leading_cue, trailing_cue = (
                    _script_edge_performance_cues(str(line["line"]))
                )
                if leading_cue:
                    previews = [
                        preview
                        for preview in previews
                        if int(
                            preview.get(
                                "trim_start_sample",
                                base_segments[start_index]["start_sample"],
                            )
                        )
                        <= int(base_segments[start_index]["start_sample"])
                    ]
                if trailing_cue:
                    previews = [
                        preview
                        for preview in previews
                        if int(
                            preview.get(
                                "trim_end_sample",
                                base_segments[end_index]["end_sample"],
                            )
                        )
                        >= int(base_segments[end_index]["end_sample"])
                    ]
                scored_previews = [
                    score_preview(line_index, preview)
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
                fragment_metrics = [
                    score_preview(
                        line_index,
                        {
                            "transcript": str(
                                action.get("transcript") or ""
                            )
                        },
                    )
                    for action in comparison_actions
                ]
                line_clauses = script_clauses(line["line"])
                comparison_start = min(
                    int(action["start_index"])
                    for action in comparison_actions
                )
                comparison_end = max(
                    int(action["start_index"])
                    + int(action["count"])
                    - 1
                    for action in comparison_actions
                )
                leading_boundary_was_incomplete = any(
                    int(
                        metric["fidelity"]["leading_missing_token_count"]
                    )
                    > 0
                    or int(
                        metric["fidelity"]["leading_substitution_count"]
                    )
                    > 0
                    for metric in fragment_metrics
                )
                trailing_boundary_was_incomplete = any(
                    int(
                        metric["fidelity"]["trailing_missing_token_count"]
                    )
                    > 0
                    or int(
                        metric["fidelity"]["trailing_substitution_count"]
                    )
                    > 0
                    for metric in fragment_metrics
                )
                short_leading_clause = bool(
                    line_clauses and word_count(line_clauses[0]) <= 2
                )
                short_trailing_clause = bool(
                    line_clauses and word_count(line_clauses[-1]) <= 2
                )
                boundary_audio_rescue = bool(
                    combined_match
                    >= float(settings.get("candidate_min_score", 45.0))
                    and (
                        (
                            start_index < comparison_start
                            and end_index <= comparison_end
                            and short_leading_clause
                            and leading_boundary_was_incomplete
                        )
                        or (
                            end_index > comparison_end
                            and start_index >= comparison_start
                            and short_trailing_clause
                            and trailing_boundary_was_incomplete
                        )
                    )
                )
                quality_improved = bool(
                    preview_quality(combined)
                    > max(
                        preview_quality(metric)
                        for metric in fragment_metrics
                    )
                    or boundary_audio_rescue
                )
                strict_preview = is_strict_preview(combined)
                provisional_preview = is_provisional_preview(
                    combined,
                    boundary_audio_rescue=boundary_audio_rescue,
                )
                fallback_preview = False
                if not (
                    quality_improved
                    and (strict_preview or provisional_preview)
                ):
                    fallback_options = [
                        scored
                        for scored in scored_previews
                        if float(scored["match"]) >= fallback_minimum_match
                        and is_provisional_preview(
                            scored,
                            boundary_audio_rescue=False,
                        )
                    ]
                    if (
                        fallback_actions_per_line == 0
                        or not fallback_options
                    ):
                        continue
                    combined = max(
                        fallback_options,
                        key=lambda scored: (
                            float(scored["match"]),
                            preview_quality(scored),
                        ),
                    )
                    span = combined["preview"]
                    combined_match = float(combined["match"])
                    combined_fidelity = combined["fidelity"]
                    combined_sentence = combined["sentence"]
                    strict_preview = is_strict_preview(combined)
                    provisional_preview = is_provisional_preview(
                        combined,
                        boundary_audio_rescue=False,
                    )
                    boundary_audio_rescue = False
                    fallback_preview = True

                expected_counts = Counter(
                    evaluator.line_features[line_index].text.tokens
                )
                observed_counts = Counter(
                    evaluator.observed_features(span["transcript"]).tokens
                )
                repeated_excess = any(
                    count >= 2 and count > expected_counts.get(token, 0)
                    for token, count in observed_counts.items()
                )
                if (
                    repeated_excess
                    and int(combined_fidelity["extra_word_count"]) > 0
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
                        evaluator.match(other_index, span["transcript"])
                        for other_index, other_line in enumerate(lines)
                        if other_index != line_index
                        and not is_vocalization_script(other_line["line"])
                    ),
                    default=0.0,
                )
                duration_score = _duration_plausibility(
                    line,
                    span,
                    expected_word_count=evaluator.line_features[
                        line_index
                    ].word_count,
                )
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
                        "confidence_margin": combined_match - other_score,
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
                        "fragment_join_fallback": fallback_preview,
                        "fragment_source_count": count,
                        "fragment_join_provisional": not strict_preview,
                        "intra_segment_trim": bool(
                            span.get("trim_start_sample") is not None
                            or span.get("trim_end_sample") is not None
                        ),
                        "trimmed_edge_join": bool(
                            span.get("trim_start_sample") is not None
                            or span.get("trim_end_sample") is not None
                        ),
                        **(
                            {
                                "trim_start_sample": int(
                                    span["trim_start_sample"]
                                ),
                                "trim_end_sample": int(
                                    span["trim_end_sample"]
                                ),
                            }
                            if (
                                span.get("trim_start_sample") is not None
                                and span.get("trim_end_sample") is not None
                            )
                            else {}
                        ),
                        "forced_review_reason": (
                            "UNCERTAIN_BOUNDARY_AUDIO"
                            if boundary_audio_rescue and not strict_preview
                            else ""
                        ),
                    }
                )
                existing.add(key)
                if strict_preview:
                    break

    joined_by_line: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for action in joined:
        joined_by_line[int(action["line_index"])].append(action)
    selected_joins = []
    for line_index, line_actions in joined_by_line.items():
        def action_quality(action: dict[str, Any]) -> tuple[Any, ...]:
            scored = score_preview(
                line_index,
                {"transcript": str(action.get("transcript") or "")},
            )
            return (
                int(scored["sentence"]["missing_clause_count"]) == 0,
                bool(scored["sentence"]["clauses_in_order"]),
                has_complete_boundaries(scored["fidelity"]),
                min(
                    float(scored["fidelity"]["prefix_token_coverage"]),
                    float(scored["fidelity"]["suffix_token_coverage"]),
                ),
                float(scored["fidelity"]["token_coverage"]),
                float(scored["match"]),
                float(scored["fidelity"]["token_precision"]),
                float(scored["fidelity"]["ordered_similarity"]),
                -int(action["count"]),
            )

        regular_actions = [
            action
            for action in line_actions
            if not action.get("fragment_join_fallback")
        ]
        fallback_actions = [
            action
            for action in line_actions
            if action.get("fragment_join_fallback")
        ]
        regular_actions.sort(key=action_quality, reverse=True)
        fallback_actions.sort(key=action_quality, reverse=True)
        selected_joins.extend(regular_actions[:maximum_actions_per_line])
        selected_joins.extend(fallback_actions[:fallback_actions_per_line])
    selected_joins.sort(
        key=lambda action: (
            int(action["start_index"]),
            int(action["count"]),
            int(action["line_index"]),
        )
    )
    return selected_joins


def _action_span_key(
    action: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    """Identify a whole-segment span or a trimmed portion of one span."""

    return (
        int(action["start_index"]),
        int(action["count"]),
        int(action.get("trim_start_sample", -1)),
        int(action.get("trim_end_sample", -1)),
    )


def _repeated_line_boundary_offsets(
    *,
    line_index: int,
    words: list[dict[str, Any]],
    evaluator: TranscriptEvaluator,
    minimum_match: float,
) -> dict[int, float]:
    """Find adjacent complete repetitions even when Whisper spans the pause."""

    expected_word_count = evaluator.line_features[line_index].word_count
    if expected_word_count < 1 or len(words) < 2:
        return {}
    # Spoken compounds are not stable ASR token boundaries (for example,
    # "Mud hopper" is commonly decoded as one "Mudhopper" token). Search a
    # narrow range of ordered word-window sizes instead of requiring the
    # observed and scripted token counts to be identical.
    minimum_window_words = max(1, expected_word_count - 2)
    maximum_window_words = expected_word_count + 2
    matching_windows = []
    for first_word in range(len(words)):
        best_window = None
        for window_word_count in range(
            minimum_window_words,
            maximum_window_words + 1,
        ):
            last_word = first_word + window_word_count - 1
            if last_word >= len(words):
                break
            window_text = " ".join(
                str(word.get("word") or "").strip()
                for word in words[first_word : last_word + 1]
            )
            evaluation = evaluator.evaluate(line_index, window_text)
            if (
                evaluation.match_score < minimum_match
                or float(evaluation.fidelity["ordered_similarity"])
                < minimum_match
                or int(evaluation.sentence["missing_clause_count"]) > 0
                or not bool(evaluation.sentence["clauses_in_order"])
            ):
                continue
            quality = (
                float(evaluation.match_score),
                -abs(window_word_count - expected_word_count),
                -window_word_count,
            )
            if best_window is None or quality > best_window[0]:
                best_window = (quality, first_word, last_word)
        if best_window is not None:
            matching_windows.append((best_window[1], best_window[2]))
    if len(matching_windows) < 2:
        return {}

    durations = sorted(
        max(0.0, float(word["end"]) - float(word["start"]))
        for word in words
    )
    typical_duration = durations[len(durations) // 2]
    boundaries = {}
    windows_by_start = {
        first_word: (first_word, last_word)
        for first_word, last_word in matching_windows
    }
    for _, left_last in matching_windows:
        right_window = windows_by_start.get(left_last + 1)
        if right_window is None:
            continue
        right_first, _ = right_window
        left_end = float(words[left_last]["end"])
        right_start = float(words[right_first]["start"])
        right_end = float(words[right_first]["end"])
        right_duration = max(0.0, right_end - right_start)
        if (
            right_start - left_end < 0.10
            and right_duration >= max(0.80, typical_duration * 2.5)
        ):
            boundary = (right_start + right_end) / 2.0
        else:
            boundary = (left_end + right_start) / 2.0
        boundaries[left_last] = boundary
    return boundaries


def _may_contain_repeated_line(
    expected: TextFeatures,
    observed: TextFeatures,
) -> bool:
    """Cheaply prefilter the session-wide repeated-line search."""

    expected_tokens = expected.tokens
    observed_tokens = observed.tokens
    expected_count = len(expected_tokens)
    if expected_count < 1 or len(observed_tokens) < 2:
        return False

    occurrence_count = 0
    token_index = 0
    while token_index + expected_count <= len(observed_tokens):
        if (
            observed_tokens[token_index : token_index + expected_count]
            == expected_tokens
        ):
            occurrence_count += 1
            if occurrence_count >= 2:
                return True
            token_index += expected_count
        else:
            token_index += 1

    # Joining spaces catches stable ASR compound changes without invoking the
    # much more expensive fuzzy evaluator for every line in a broad session.
    expected_compact = "".join(expected_tokens)
    observed_compact = "".join(observed_tokens)
    return bool(
        len(expected_compact) >= 5
        and observed_compact.count(expected_compact) >= 2
    )


def _segment_voice_regions_for_trimming(
    segment: Mapping[str, Any],
    *,
    project_dir: Path,
    threshold: float,
    stored_key: str = "speech_regions",
) -> list[tuple[int, int]]:
    """Return segment-local speech regions, reusing stored VAD when possible."""

    base_start = int(segment.get("start_sample") or 0)
    base_end = int(segment.get("end_sample") or base_start)
    metadata = segment.get("voice_bounds")
    if isinstance(metadata, Mapping):
        stored = metadata.get(stored_key)
        if isinstance(stored, list):
            regions = []
            for region in stored:
                if not isinstance(region, Mapping):
                    regions = []
                    break
                start = max(base_start, int(region["start_sample"]))
                end = min(base_end, int(region["end_sample"]))
                if end > start:
                    regions.append((start - base_start, end - base_start))
            if regions:
                return regions

    if not segment.get("file") or base_end <= base_start:
        return []
    return pcm_voice_regions(
        resolve_project_path(project_dir, str(segment["file"])),
        start_sample=0,
        end_sample=base_end - base_start,
        threshold=threshold,
    )


def _snap_repeated_boundaries_to_voice_gaps(
    boundaries: Mapping[int, float],
    *,
    segment: Mapping[str, Any],
    project_dir: Path,
    sample_rate: int,
    minimum_gap: float,
    segmentation_settings: Mapping[str, Any],
) -> dict[int, float]:
    """Replace unreliable repeated-word midpoints with acoustic gaps.

    Whisper often stretches the first word of a repeated take backward across
    the pause. A midpoint inside that word can therefore retain the previous
    take or cut the next one. Repetition trims are admitted only when Silero
    VAD provides enough ordered inter-region gaps to place every boundary.
    """

    if not boundaries:
        return {}
    regions = _segment_voice_regions_for_trimming(
        segment,
        project_dir=project_dir,
        threshold=float(
            segmentation_settings.get(
                "voice_boundary_breath_vad_threshold",
                0.70,
            )
        ),
        stored_key="strict_speech_regions",
    )
    minimum_voice_gap = max(0.10, min(float(minimum_gap), 0.30))
    gaps = [
        (left_end, right_start)
        for (_, left_end), (right_start, _) in zip(regions, regions[1:])
        if (right_start - left_end) / sample_rate >= minimum_voice_gap
    ]
    ordered = sorted(boundaries.items())
    if len(gaps) < len(ordered):
        return {}

    selected: list[tuple[int, int]] = []
    first_gap = 0
    for position, (word_index, proposed_seconds) in enumerate(ordered):
        remaining = len(ordered) - position - 1
        last_gap = len(gaps) - remaining - 1
        gap_index = min(
            range(first_gap, last_gap + 1),
            key=lambda index: abs(
                ((gaps[index][0] + gaps[index][1]) / 2.0) / sample_rate
                - proposed_seconds
            ),
        )
        selected.append((word_index, gap_index))
        first_gap = gap_index + 1

    audio_path = resolve_project_path(project_dir, str(segment["file"]))
    snapped = {}
    for word_index, gap_index in selected:
        gap_start, gap_end = gaps[gap_index]
        proposed_sample = (gap_start + gap_end) // 2
        quiet_sample = quietest_pcm_boundary(
            audio_path,
            proposed_sample=proposed_sample,
            minimum_sample=gap_start,
            maximum_sample=gap_end,
            search_seconds=(gap_end - gap_start) / (2.0 * sample_rate),
            window_seconds=float(
                segmentation_settings.get(
                    "word_split_snap_window_seconds",
                    0.02,
                )
            ),
            maximum_rms_dbfs=float(
                segmentation_settings.get(
                    "word_split_snap_max_rms_dbfs",
                    -42.0,
                )
            ),
            require_quiet=False,
        )
        snapped[word_index] = quiet_sample / sample_rate
    return snapped


def _acoustic_take_trim_proposals(
    *,
    source_action: Mapping[str, Any],
    segment: Mapping[str, Any],
    base_index: int,
    line_index: int,
    line: dict[str, Any],
    sample_rate: int,
    settings: AlignmentSettings,
    evaluator: TranscriptEvaluator,
    project_dir: Path,
    segmentation_settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Propose take-sized windows when ASR collapses repeated delivery.

    Whisper sometimes returns only one copy of a line even when the recording
    contains several takes.  In that case word repetition cannot supply split
    points.  Strict VAD gaps provide bounded acoustic hypotheses; the normal
    exact-span ASR stage must still independently verify every proposal.
    """

    if not bool(settings.get("acoustic_take_trim_enabled", True)):
        return []
    transcript = str(
        ((segment.get("segment_asr") or {}).get("primary") or {}).get(
            "transcript"
        )
        or segment.get("transcript")
        or source_action.get("transcript")
        or ""
    )
    full = evaluator.evaluate(line_index, transcript)
    if (
        full.match_score
        < float(settings.get("intra_segment_trim_min_match_score", 85.0))
        or float(full.fidelity["token_coverage"]) < 0.85
        or float(full.fidelity["token_precision"]) < 0.85
        or int(full.sentence["missing_clause_count"]) > 0
        or not bool(full.sentence["clauses_in_order"])
    ):
        return []

    base_start = int(segment["start_sample"])
    base_end = int(segment["end_sample"])
    full_duration_score = _duration_plausibility(
        line,
        {
            "start_seconds": base_start / sample_rate,
            "end_seconds": base_end / sample_rate,
        },
        expected_word_count=evaluator.line_features[line_index].word_count,
    )
    if full_duration_score > float(
        settings.get("acoustic_take_trim_max_full_duration_plausibility", 70.0)
    ):
        return []

    regions = _segment_voice_regions_for_trimming(
        segment,
        project_dir=project_dir,
        threshold=float(
            segmentation_settings.get(
                "voice_boundary_breath_vad_threshold",
                0.70,
            )
        ),
        stored_key="strict_speech_regions",
    )
    minimum_gap_samples = round(
        float(settings.get("intra_segment_trim_min_gap_seconds", 0.40))
        * sample_rate
    )
    gaps = [
        (left_end, right_start)
        for (_, left_end), (right_start, _) in zip(regions, regions[1:])
        if right_start - left_end >= minimum_gap_samples
    ]
    if len(gaps) < 2:
        return []

    audio_path = resolve_project_path(project_dir, str(segment["file"]))
    boundaries = []
    for gap_start, gap_end in gaps:
        proposed = (gap_start + gap_end) // 2
        quiet = quietest_pcm_boundary(
            audio_path,
            proposed_sample=proposed,
            minimum_sample=gap_start,
            maximum_sample=gap_end,
            search_seconds=(gap_end - gap_start) / (2.0 * sample_rate),
            window_seconds=float(
                segmentation_settings.get(
                    "word_split_snap_window_seconds",
                    0.02,
                )
            ),
            maximum_rms_dbfs=float(
                segmentation_settings.get(
                    "word_split_snap_max_rms_dbfs",
                    -42.0,
                )
            ),
            require_quiet=False,
        )
        boundaries.append(int(quiet))

    points = [0, *boundaries, base_end - base_start]
    leading_cue, trailing_cue = _script_edge_performance_cues(str(line["line"]))
    proposals = []
    minimum_window_score = float(
        settings.get("acoustic_take_trim_min_duration_plausibility", 45.0)
    )
    for first in range(len(points) - 1):
        for last in range(first + 1, len(points)):
            if first == 0 and last == len(points) - 1:
                continue
            if (leading_cue and first > 0) or (
                trailing_cue and last < len(points) - 1
            ):
                continue
            start_sample = base_start + points[first]
            end_sample = base_start + points[last]
            duration_score = _duration_plausibility(
                line,
                {
                    "start_seconds": start_sample / sample_rate,
                    "end_seconds": end_sample / sample_rate,
                },
                expected_word_count=evaluator.line_features[line_index].word_count,
            )
            if duration_score < minimum_window_score:
                continue
            proposals.append(
                {
                    "type": "assigned",
                    "start_index": base_index,
                    "count": 1,
                    "line_index": line_index,
                    "match_score": full.match_score,
                    "confidence_margin": float(
                        source_action.get("confidence_margin", 0.0)
                    ),
                    "transcript": str(line["line"]),
                    "transcript_source": "acoustic_take_trim_preview",
                    "duration_plausibility": duration_score,
                    "order_hint": float(source_action.get("order_hint", 0.0)),
                    "top_matches": [],
                    "trim_start_sample": start_sample,
                    "trim_end_sample": end_sample,
                    "intra_segment_trim": True,
                    "repeated_take_trim": True,
                    "acoustic_take_trim": True,
                    "is_primary_match": True,
                    "segment_match_rank": 1,
                }
            )
    proposals.sort(
        key=lambda action: (
            float(action["duration_plausibility"]),
            -(int(action["trim_end_sample"]) - int(action["trim_start_sample"])),
        ),
        reverse=True,
    )
    return proposals[
        : max(1, int(settings.get("acoustic_take_trim_max_actions_per_line", 8)))
    ]


def _intra_segment_trim_actions(
    actions: list[dict[str, Any]],
    *,
    lines: list[dict[str, Any]],
    base_segments: list[dict[str, Any]],
    sample_rate: int,
    settings: Mapping[str, Any] | AlignmentSettings,
    evaluator: TranscriptEvaluator,
    project_dir: Path | None = None,
    segmentation_settings: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Recover line-sized candidates inside an oversized base segment.

    Trimming is allowed only at a clear independently-transcribed word gap.
    The trimmed text must be complete and materially more precise than the
    full base transcript. The resulting WAV is independently decoded later,
    exactly like every other serious alignment candidate.
    """

    settings = AlignmentSettings.from_value(settings)
    segmentation_settings = dict(segmentation_settings or {})
    if not bool(settings.get("intra_segment_trim_enabled", True)):
        return []

    minimum_gap = float(
        settings.get("intra_segment_trim_min_gap_seconds", 0.40)
    )
    minimum_match = float(
        settings.get("intra_segment_trim_min_match_score", 85.0)
    )
    minimum_coverage = float(
        settings.get("intra_segment_trim_min_token_coverage", 0.85)
    )
    minimum_precision = float(
        settings.get("intra_segment_trim_min_token_precision", 0.85)
    )
    minimum_ordered = float(
        settings.get("intra_segment_trim_min_ordered_score", 70.0)
    )
    maximum_per_line = max(
        0,
        int(settings.get("intra_segment_trim_max_actions_per_line", 2)),
    )
    maximum_per_segment = max(
        0,
        int(settings.get("intra_segment_trim_max_actions_per_segment", 3)),
    )
    if maximum_per_line == 0 or maximum_per_segment == 0:
        return []

    proposals: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    repetition_search_lines = [
        (
            line_index,
            evaluator.line_features[line_index],
        )
        for line_index, line in enumerate(lines)
        if not is_vocalization_script(line["line"])
    ]
    source_action_by_base: dict[int, dict[str, Any]] = {}
    for source_action in actions:
        for base_index in range(
            int(source_action["start_index"]),
            int(source_action["start_index"])
            + int(source_action["count"]),
        ):
            if 0 <= base_index < len(base_segments):
                source_action_by_base.setdefault(base_index, source_action)
    source_actions_by_base = [
        (source_action_by_base.get(base_index), base_index)
        for base_index in range(len(base_segments))
    ]
    for source_action, base_index in source_actions_by_base:
        check_processing_cancelled()
        segment = base_segments[base_index]
        segment_asr = segment.get("segment_asr") or {}
        primary_asr = segment_asr.get("primary") or {}
        words = [
            word
            for word in primary_asr.get("words") or []
            if str(word.get("word") or "").strip()
            and word.get("start") is not None
            and word.get("end") is not None
        ]
        if len(words) < 2:
            continue
        observed_words = evaluator.observed_features(
            " ".join(str(word["word"]).strip() for word in words)
        )

        gap_boundary_offsets: dict[int, float] = {}
        for word_index, (left, right) in enumerate(
            zip(words, words[1:])
        ):
            left_end = float(left["end"])
            right_start = float(right["start"])
            if right_start - left_end < minimum_gap:
                continue
            gap = right_start - left_end
            boundary = (left_end + right_start) / 2.0
            if (
                project_dir is not None
                and segment.get("file")
                and bool(
                    segmentation_settings.get(
                        "word_split_snap_enabled",
                        True,
                    )
                )
            ):
                snapped_sample = quietest_pcm_boundary(
                    resolve_project_path(project_dir, str(segment["file"])),
                    proposed_sample=round(boundary * sample_rate),
                    minimum_sample=round(left_end * sample_rate),
                    maximum_sample=round(right_start * sample_rate),
                    search_seconds=min(
                        float(
                            segmentation_settings.get(
                                "word_split_snap_search_seconds",
                                0.20,
                            )
                        ),
                        gap / 2.0,
                    ),
                    window_seconds=float(
                        segmentation_settings.get(
                            "word_split_snap_window_seconds",
                            0.02,
                        )
                    ),
                    maximum_rms_dbfs=float(
                        segmentation_settings.get(
                            "word_split_snap_max_rms_dbfs",
                            -42.0,
                        )
                    ),
                    require_quiet=True,
                )
                if snapped_sample is None:
                    continue
                boundary = snapped_sample / sample_rate
            gap_boundary_offsets[word_index] = boundary

        line_indexes = []
        if source_action is not None:
            line_indexes.append(int(source_action["line_index"]))
            line_indexes.extend(
                int(match["line_index"])
                for match in source_action.get("top_matches") or []
                if float(match.get("match_score", 0.0))
                >= float(settings.get("candidate_min_score", 45.0))
            )
        repeated_boundaries_by_line: dict[int, dict[int, float]] = {}
        for repeated_line_index, repeated_line_features in repetition_search_lines:
            expected_word_count = repeated_line_features.word_count
            if len(words) < max(2, 2 * max(1, expected_word_count - 2)):
                continue
            if not _may_contain_repeated_line(
                repeated_line_features.text,
                observed_words,
            ):
                continue
            repeated = _repeated_line_boundary_offsets(
                line_index=repeated_line_index,
                words=words,
                evaluator=evaluator,
                minimum_match=minimum_match,
            )
            if repeated:
                repeated_boundaries_by_line[repeated_line_index] = repeated
                line_indexes.append(repeated_line_index)
        action_hint = source_action or {}
        for line_index in dict.fromkeys(line_indexes):
            line = lines[line_index]
            if is_vocalization_script(line["line"]):
                continue
            leading_cue, trailing_cue = _script_edge_performance_cues(
                str(line["line"])
            )
            boundary_offsets = dict(gap_boundary_offsets)
            repeated_boundaries = repeated_boundaries_by_line.get(line_index)
            if repeated_boundaries is None:
                repeated_boundaries = _repeated_line_boundary_offsets(
                    line_index=line_index,
                    words=words,
                    evaluator=evaluator,
                    minimum_match=minimum_match,
                )
            if repeated_boundaries and project_dir is not None:
                repeated_boundaries = _snap_repeated_boundaries_to_voice_gaps(
                    repeated_boundaries,
                    segment=segment,
                    project_dir=project_dir,
                    sample_rate=sample_rate,
                    minimum_gap=minimum_gap,
                    segmentation_settings=segmentation_settings,
                )
            boundary_offsets.update(repeated_boundaries)
            if project_dir is not None:
                for acoustic in _acoustic_take_trim_proposals(
                    source_action=action_hint,
                    segment=segment,
                    base_index=base_index,
                    line_index=line_index,
                    line=line,
                    sample_rate=sample_rate,
                    settings=settings,
                    evaluator=evaluator,
                    project_dir=project_dir,
                    segmentation_settings=segmentation_settings,
                ):
                    acoustic_key = (
                        line_index,
                        base_index,
                        int(acoustic["trim_start_sample"]),
                        int(acoustic["trim_end_sample"]),
                    )
                    if acoustic_key not in seen:
                        seen.add(acoustic_key)
                        proposals.append(acoustic)
            if not boundary_offsets:
                continue
            ordered_boundaries = sorted(boundary_offsets)
            end_words = [*ordered_boundaries, len(words) - 1]
            start_words = [
                0,
                *(word_index + 1 for word_index in ordered_boundaries),
            ]
            word_windows = [
                (first_word, last_word)
                for first_word in start_words
                for last_word in end_words
                if first_word <= last_word
                and not (
                    first_word == 0
                    and last_word == len(words) - 1
                )
            ]
            full = evaluator.evaluate(
                line_index,
                str(
                    primary_asr.get("transcript")
                    or segment.get("transcript")
                    or ""
                ),
            )
            for first_word, last_word in word_windows:
                if (
                    (leading_cue and first_word > 0)
                    or (
                        trailing_cue
                        and last_word < len(words) - 1
                    )
                ):
                    continue
                trim_start_offset = (
                    boundary_offsets[first_word - 1]
                    if first_word
                    else 0.0
                )
                trim_end_offset = (
                    boundary_offsets[last_word]
                    if last_word < len(words) - 1
                    else float(
                        segment["end_seconds"]
                        - segment["start_seconds"]
                    )
                )
                window_text = " ".join(
                    str(word["word"]).strip()
                    for word in words[first_word : last_word + 1]
                )
                trimmed = evaluator.evaluate(line_index, window_text)
                fidelity = trimmed.fidelity
                sentence = trimmed.sentence
                is_repeated_window = bool(
                    (
                        first_word > 0
                        and first_word - 1 in repeated_boundaries
                    )
                    or (
                        last_word < len(words) - 1
                        and last_word in repeated_boundaries
                    )
                )
                # ASR commonly joins or splits compounds differently from the
                # script ("Mud hopper" -> "Mudhopper"). An otherwise near-
                # exact ordered repeated window is safe to propose with one
                # unmatched token; exact-span ASR still verifies it later.
                required_coverage = minimum_coverage
                if (
                    is_repeated_window
                    and float(fidelity["ordered_similarity"]) >= 95.0
                    and float(fidelity["token_precision"]) >= 0.95
                ):
                    required_coverage = min(required_coverage, 0.75)
                if (
                    trimmed.match_score < minimum_match
                    or float(fidelity["token_coverage"]) < required_coverage
                    or float(fidelity["token_precision"]) < minimum_precision
                    or float(fidelity["ordered_similarity"]) < minimum_ordered
                    or int(sentence["missing_clause_count"]) > 0
                    or not bool(sentence["clauses_in_order"])
                    or int(fidelity["leading_extra_token_count"]) > 0
                    or int(fidelity["trailing_extra_token_count"]) > 0
                ):
                    continue
                if (
                    int(fidelity["extra_word_count"])
                    >= int(full.fidelity["extra_word_count"])
                    and float(fidelity["token_precision"])
                    <= float(full.fidelity["token_precision"])
                ):
                    continue

                base_start_sample = int(segment["start_sample"])
                start_sample = max(
                    base_start_sample,
                    base_start_sample
                    + round(trim_start_offset * sample_rate),
                )
                end_sample = min(
                    int(segment["end_sample"]),
                    base_start_sample
                    + round(trim_end_offset * sample_rate),
                )
                key = (line_index, base_index, start_sample, end_sample)
                if end_sample <= start_sample or key in seen:
                    continue
                seen.add(key)

                other_score = max(
                    (
                        evaluator.match(other_index, window_text)
                        for other_index, other_line in enumerate(lines)
                        if other_index != line_index
                        and not is_vocalization_script(other_line["line"])
                    ),
                    default=0.0,
                )
                preview = {
                    "start_seconds": start_sample / sample_rate,
                    "end_seconds": end_sample / sample_rate,
                }
                duration_score = _duration_plausibility(
                    line,
                    preview,
                    expected_word_count=evaluator.line_features[
                        line_index
                    ].word_count,
                )
                proposals.append(
                    {
                        "type": "assigned",
                        "start_index": base_index,
                        "count": 1,
                        "line_index": line_index,
                        "match_score": trimmed.match_score,
                        "confidence_margin": (
                            trimmed.match_score - other_score
                        ),
                        "transcript": window_text,
                        "transcript_source": (
                            "segment_asr_word_trim_preview"
                        ),
                        "duration_plausibility": duration_score,
                        "order_hint": float(
                            action_hint.get("order_hint", 0.0)
                        ),
                        "top_matches": [
                            {
                                "line_index": line_index,
                                "match_score": trimmed.match_score,
                                "ranking_score": trimmed.match_score,
                                "duration_plausibility": duration_score,
                                "order_hint": float(
                                    action_hint.get("order_hint", 0.0)
                                ),
                                "confidence_margin": (
                                    trimmed.match_score - other_score
                                ),
                            }
                        ],
                        "trim_start_sample": start_sample,
                        "trim_end_sample": end_sample,
                        "trim_word_start": first_word,
                        "trim_word_end": last_word,
                        "intra_segment_trim": True,
                        "repeated_take_trim": is_repeated_window,
                        "is_primary_match": True,
                        "segment_match_rank": 1,
                    }
                )

    proposals.sort(
        key=lambda action: (
            bool(action.get("repeated_take_trim")),
            float(action["match_score"]),
            float(
                evaluator.evaluate(
                    int(action["line_index"]),
                    str(action["transcript"]),
                ).fidelity["token_precision"]
            ),
            float(action["duration_plausibility"]),
            -(
                int(action["trim_end_sample"])
                - int(action["trim_start_sample"])
            ),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    line_counts: Counter[int] = Counter()
    segment_counts: Counter[int] = Counter()
    for action in proposals:
        line_index = int(action["line_index"])
        base_index = int(action["start_index"])
        line_limit = (
            max(
                maximum_per_line,
                int(settings.get("acoustic_take_trim_max_actions_per_line", 8)),
            )
            if (
                action.get("acoustic_take_trim")
                or action.get("repeated_take_trim")
            )
            else maximum_per_line
        )
        if (
            line_counts[line_index] >= line_limit
            or segment_counts[base_index] >= maximum_per_segment
        ):
            continue
        selected.append(action)
        line_counts[line_index] += 1
        segment_counts[base_index] += 1
    return sorted(
        selected,
        key=lambda action: (
            int(action["start_index"]),
            int(action["trim_start_sample"]),
            int(action["line_index"]),
        ),
    )


def _action_word_sample_bounds(
    *,
    source_action: Mapping[str, Any],
    base_segments: list[dict[str, Any]],
    sample_rate: int,
    raw_start: int,
    raw_end: int,
) -> tuple[int, int, int, int, bool] | None:
    """Return outer and edge-word ASR bounds inside an action's PCM span."""

    start_index = int(source_action["start_index"])
    end_index = start_index + int(source_action["count"])
    word_bounds: list[tuple[int, int, bool]] = []
    for segment in base_segments[start_index:end_index]:
        primary_words = (
            ((segment.get("segment_asr") or {}).get("primary") or {}).get(
                "words"
            )
            or []
        )
        words = primary_words or segment.get("words") or []
        words_are_local = bool(primary_words) or str(
            segment.get("transcript_source") or ""
        ) not in {"", "session_asr_fallback"}
        segment_start = int(segment["start_sample"])
        for word in words:
            if word.get("start") is None or word.get("end") is None:
                continue
            offset = segment_start if words_are_local else 0
            word_start = offset + round(float(word["start"]) * sample_rate)
            word_end = offset + round(float(word["end"]) * sample_rate)
            if word_end <= raw_start or word_start >= raw_end:
                continue
            word_bounds.append(
                (
                    max(raw_start, word_start),
                    min(raw_end, word_end),
                    word_start < raw_start,
                )
            )
    if not word_bounds:
        return None
    word_bounds.sort()
    return (
        word_bounds[0][0],
        max(end for _, end, _ in word_bounds),
        word_bounds[0][1],
        max(word_bounds, key=lambda item: item[1])[0],
        word_bounds[0][2],
    )


def _stored_action_voice_bounds(
    *,
    source_action: Mapping[str, Any],
    base_segments: list[dict[str, Any]],
    raw_start: int,
    raw_end: int,
    key: str,
) -> tuple[bool, tuple[int, int] | None]:
    """Compose segmentation-time VAD bounds for an untrimmed base span."""

    start_index = int(source_action["start_index"])
    end_index = start_index + int(source_action["count"])
    if (
        raw_start != int(base_segments[start_index]["start_sample"])
        or raw_end != int(base_segments[end_index - 1]["end_sample"])
    ):
        return False, None

    bounds = []
    for segment in base_segments[start_index:end_index]:
        metadata = segment.get("voice_bounds")
        if not isinstance(metadata, Mapping) or key not in metadata:
            return False, None
        stored = metadata.get(key)
        if stored is None:
            continue
        if not isinstance(stored, Mapping):
            return False, None
        bounds.append(
            (
                max(raw_start, int(stored["start_sample"])),
                min(raw_end, int(stored["end_sample"])),
            )
        )
    if not bounds:
        return True, None
    return True, (
        min(start for start, _ in bounds),
        max(end for _, end in bounds),
    )


def _boundary_voice_trim_actions(
    actions: list[dict[str, Any]],
    *,
    project_dir: Path,
    session_entry: dict[str, Any],
    lines: list[dict[str, Any]],
    base_segments: list[dict[str, Any]],
    settings: Mapping[str, Any] | AlignmentSettings,
    evaluator: TranscriptEvaluator,
    segmentation_settings: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create clean alternatives when VAD finds non-speech at span edges."""

    settings = AlignmentSettings.from_value(settings)
    segmentation_settings = dict(segmentation_settings or {})
    if not bool(settings.get("boundary_voice_trim_enabled", True)):
        return []
    if not bool(
        segmentation_settings.get("voice_boundary_detection_enabled", True)
    ):
        return []
    minimum_match = float(
        settings.get("boundary_noise_cleanup_min_match_score", 85.0)
    )
    minimum_edge_seconds = float(
        settings.get("boundary_voice_trim_min_edge_seconds", 0.30)
    )
    sample_rate = int(session_entry["sample_rate"])
    pre_padding = round(
        float(
            settings.get(
                "boundary_voice_trim_pre_padding_seconds",
                0.08,
            )
        )
        * sample_rate
    )
    post_padding = round(
        float(
            settings.get(
                "boundary_voice_trim_post_padding_seconds",
                0.12,
            )
        )
        * sample_rate
    )
    threshold = float(
        segmentation_settings.get("voice_boundary_vad_threshold", 0.50)
    )
    breath_threshold = float(
        segmentation_settings.get(
            "voice_boundary_breath_vad_threshold",
            0.70,
        )
    )
    audio_path = resolve_project_path(
        project_dir,
        session_entry.get("working_audio", session_entry["audio"]),
    )
    voice_cache: dict[tuple[int, int], tuple[int, int] | None] = {}
    breath_voice_cache: dict[tuple[int, int], tuple[int, int] | None] = {}
    voice_region_cache: dict[tuple[int, int], list[tuple[int, int]]] = {}
    strict_voice_region_cache: dict[
        tuple[int, int], list[tuple[int, int]]
    ] = {}
    stored_detection = session_entry.get("voice_boundary_detection")
    stored_detection_current = bool(
        isinstance(stored_detection, Mapping)
        and bool(stored_detection.get("enabled", True))
        and math.isclose(
            float(stored_detection.get("vad_threshold", -1.0)),
            threshold,
        )
        and math.isclose(
            float(stored_detection.get("breath_vad_threshold", -1.0)),
            breath_threshold,
        )
    )
    occupied = {
        (
            int(action["line_index"]),
            int(action["start_index"]),
            int(action["count"]),
            int(action.get("trim_start_sample", -1)),
            int(action.get("trim_end_sample", -1)),
        )
        for action in actions
    }
    source_variants = []
    for root_action in actions:
        source_variants.append((root_action, root_action))
        primary_line_index = int(root_action["line_index"])
        for match in root_action.get("top_matches") or []:
            alternate_line_index = int(match["line_index"])
            if (
                alternate_line_index == primary_line_index
                or float(match.get("match_score", 0.0))
                < float(settings.get("candidate_min_score", 45.0))
            ):
                continue
            alternate = dict(root_action)
            alternate.update(
                {
                    "line_index": alternate_line_index,
                    "match_score": float(match.get("match_score", 0.0)),
                    "confidence_margin": float(
                        match.get(
                            "confidence_margin",
                            root_action.get("confidence_margin", 0.0),
                        )
                    ),
                    "top_matches": [dict(match)],
                }
            )
            source_variants.append((alternate, root_action))

    recovered = []
    for source_action, root_action in source_variants:
        check_processing_cancelled()
        line_index = int(source_action["line_index"])
        line = lines[line_index]
        if is_nonverbal_script(str(line["line"])):
            continue
        evaluation = evaluator.evaluate(
            line_index,
            str(source_action.get("transcript") or ""),
        )
        fidelity = evaluation.fidelity
        sentence = evaluation.sentence
        complete_match = bool(
            evaluation.match_score >= minimum_match
            and float(fidelity["token_coverage"]) >= 0.85
            and float(fidelity["token_precision"]) >= 0.85
            and int(fidelity["leading_extra_token_count"]) == 0
            and int(fidelity["trailing_extra_token_count"]) == 0
            and int(sentence["missing_clause_count"]) == 0
            and bool(sentence["clauses_in_order"])
        )
        # Basic VAD edge cleanup is text-independent. Keep a lower-scoring
        # action alive so exact-span ASR can reroute its cleaned audio to the
        # correct secondary line later. More invasive detached-speech recovery
        # still requires a complete textual match below.
        if evaluation.match_score < float(
            settings.get("candidate_min_score", 45.0)
        ):
            continue

        start_index = int(source_action["start_index"])
        count = int(source_action["count"])
        end_index = start_index + count - 1
        raw_start = int(
            source_action.get(
                "trim_start_sample",
                base_segments[start_index]["start_sample"],
            )
        )
        raw_end = int(
            source_action.get(
                "trim_end_sample",
                base_segments[end_index]["end_sample"],
            )
        )
        if raw_end <= raw_start:
            continue
        raw_key = (raw_start, raw_end)
        stored_available, voice_bounds = (False, None)
        if stored_detection_current:
            stored_available, voice_bounds = _stored_action_voice_bounds(
                source_action=source_action,
                base_segments=base_segments,
                raw_start=raw_start,
                raw_end=raw_end,
                key="speech",
            )
        if not stored_available:
            if raw_key not in voice_cache:
                voice_cache[raw_key] = pcm_voice_bounds(
                    audio_path,
                    start_sample=raw_start,
                    end_sample=raw_end,
                    threshold=threshold,
                )
            voice_bounds = voice_cache[raw_key]
        if voice_bounds is None:
            continue

        leading_cue, trailing_cue = (
            _script_edge_performance_cues(str(line["line"]))
            if complete_match
            else (False, False)
        )
        voice_start, voice_end = voice_bounds
        detected_voice_start = voice_start
        word_bounds = _action_word_sample_bounds(
            source_action=source_action,
            base_segments=base_segments,
            sample_rate=sample_rate,
            raw_start=raw_start,
            raw_end=raw_end,
        )
        if word_bounds is not None:
            # Preserve a complete quiet first word only when a more permissive
            # VAD pass independently finds speech in its timestamp interval.
            # Whisper can otherwise stretch a word far back into room tone.
            if (
                int(word_bounds[2]) <= detected_voice_start
                and not bool(word_bounds[4])
            ):
                weak_bounds = pcm_voice_bounds(
                    audio_path,
                    start_sample=max(raw_start, int(word_bounds[0])),
                    end_sample=detected_voice_start,
                    threshold=float(
                        settings.get(
                            "boundary_voice_trim_weak_vad_threshold",
                            0.30,
                        )
                    ),
                )
                if (
                    weak_bounds is not None
                    and int(weak_bounds[0]) < detected_voice_start
                    and int(weak_bounds[1]) > int(word_bounds[0])
                ):
                    voice_start = min(
                        voice_start,
                        int(weak_bounds[0]),
                        int(word_bounds[0]),
                    )
        sibilant_start = pcm_leading_sibilant_start(
            audio_path,
            start_sample=raw_start,
            voice_start_sample=detected_voice_start,
            maximum_lookback_seconds=float(
                settings.get(
                    "boundary_voice_trim_sibilant_lookback_seconds",
                    0.50,
                )
            ),
        )
        if sibilant_start is not None:
            voice_start = min(voice_start, sibilant_start)
        if (
            not trailing_cue
            and word_bounds is not None
            and breath_threshold > threshold
        ):
            stored_breath_available, breath_voice_bounds = (False, None)
            if stored_detection_current:
                (
                    stored_breath_available,
                    breath_voice_bounds,
                ) = _stored_action_voice_bounds(
                    source_action=source_action,
                    base_segments=base_segments,
                    raw_start=raw_start,
                    raw_end=raw_end,
                    key="strict_speech",
                )
            if not stored_breath_available:
                if raw_key not in breath_voice_cache:
                    breath_voice_cache[raw_key] = pcm_voice_bounds(
                        audio_path,
                        start_sample=raw_start,
                        end_sample=raw_end,
                        threshold=breath_threshold,
                    )
                breath_voice_bounds = breath_voice_cache[raw_key]
            if breath_voice_bounds is not None:
                guarded_breath_end = max(
                    int(breath_voice_bounds[1]),
                    int(word_bounds[1]),
                )
                if (
                    raw_end - guarded_breath_end
                    >= round(minimum_edge_seconds * sample_rate)
                ):
                    voice_end = min(voice_end, guarded_breath_end)
        detached_trailing = False
        if (
            bool(settings.get("detached_edge_voice_trim_enabled", True))
            and not trailing_cue
            and word_bounds is not None
        ):
            if raw_key not in strict_voice_region_cache:
                strict_voice_region_cache[raw_key] = pcm_voice_regions(
                    audio_path,
                    start_sample=raw_start,
                    end_sample=raw_end,
                    threshold=breath_threshold,
                )
            strict_regions = strict_voice_region_cache[raw_key]
            if len(strict_regions) >= 2:
                previous_end = int(strict_regions[-2][1])
                tail_start, tail_end = map(int, strict_regions[-1])
                minimum_detached_gap = round(
                    float(
                        settings.get(
                            "detached_edge_voice_min_gap_seconds",
                            0.30,
                        )
                    )
                    * sample_rate
                )
                maximum_tail = round(
                    float(
                        settings.get(
                            "detached_edge_voice_max_tail_seconds",
                            0.50,
                        )
                    )
                    * sample_rate
                )
                if (
                    tail_start - previous_end >= minimum_detached_gap
                    and tail_end - tail_start <= maximum_tail
                    and int(word_bounds[1]) <= previous_end
                ):
                    voice_end = min(voice_end, previous_end)
                    detached_trailing = True

        leading_seconds = (voice_start - raw_start) / sample_rate
        trailing_seconds = (raw_end - voice_end) / sample_rate
        if (
            not complete_match
            and max(leading_seconds, trailing_seconds)
            < float(
                settings.get(
                    "boundary_voice_trim_incomplete_min_edge_seconds",
                    0.50,
                )
            )
        ):
            continue
        clean_start = raw_start
        clean_end = raw_end
        if not leading_cue and leading_seconds >= minimum_edge_seconds:
            clean_start = max(raw_start, voice_start - pre_padding)
        if not trailing_cue and trailing_seconds >= minimum_edge_seconds:
            proposed_end = min(raw_end, voice_end + post_padding)
            if proposed_end < raw_end:
                # Strict VAD supplies only a proposal.  A real quiet window
                # after it is required so an unvoiced release cannot be cut
                # merely because VAD and ASR both ended too early.
                maximum_release_end = min(
                    raw_end,
                    voice_end
                    + round(
                        float(
                            settings.get(
                                "boundary_voice_trim_max_release_seconds",
                                0.35,
                            )
                        )
                        * sample_rate
                    ),
                )
                quiet_end = first_quiet_pcm_boundary(
                    audio_path,
                    start_sample=proposed_end,
                    end_sample=maximum_release_end,
                    window_seconds=float(
                        segmentation_settings.get(
                            "word_split_snap_window_seconds",
                            0.02,
                        )
                    ),
                    maximum_rms_dbfs=float(
                        segmentation_settings.get(
                            "word_split_snap_max_rms_dbfs",
                            -42.0,
                        )
                    ),
                )
                if quiet_end is not None:
                    clean_end = max(
                        proposed_end,
                        min(maximum_release_end, quiet_end),
                    )
                else:
                    # Processed room tone may never cross the absolute quiet
                    # threshold.  Keep a bounded release allowance instead of
                    # retaining the entire noisy tail.
                    clean_end = maximum_release_end
        detached_start = None
        if (
            bool(settings.get("detached_edge_voice_trim_enabled", True))
            and complete_match
            and not leading_cue
            and source_action.get("trim_start_sample") is None
            and source_action.get("trim_end_sample") is None
            and evaluator.line_features[line_index].word_count
            >= int(settings.get("detached_edge_voice_min_script_words", 4))
            and float(source_action.get("duration_plausibility") or 0.0)
            <= float(
                settings.get(
                    "detached_edge_voice_max_duration_plausibility",
                    90.0,
                )
            )
        ):
            if raw_key not in voice_region_cache:
                voice_region_cache[raw_key] = pcm_voice_regions(
                    audio_path,
                    start_sample=raw_start,
                    end_sample=raw_end,
                    threshold=threshold,
                )
            minimum_detached_gap = round(
                float(
                    settings.get(
                        "detached_edge_voice_min_gap_seconds",
                        0.30,
                    )
                )
                * sample_rate
            )
            regions = voice_region_cache[raw_key]
            for (_, left_end), (right_start, _) in zip(
                regions,
                regions[1:],
            ):
                if right_start - left_end < minimum_detached_gap:
                    continue
                prefix_voice_seconds = (
                    left_end - regions[0][0]
                ) / sample_rate
                removed_fraction = (right_start - raw_start) / (
                    raw_end - raw_start
                )
                if (
                    prefix_voice_seconds
                    > float(
                        settings.get(
                            "detached_edge_voice_max_prefix_seconds",
                            2.0,
                        )
                    )
                    or removed_fraction
                    > float(
                        settings.get(
                            "detached_edge_voice_max_removed_fraction",
                            0.40,
                        )
                    )
                ):
                    break
                proposed_start = max(raw_start, right_start - pre_padding)
                detached_start = proposed_start
                break

        variants = []
        if clean_end > clean_start and (
            clean_start != raw_start or clean_end != raw_end
        ):
            variants.append((clean_start, clean_end, False))
        if (
            detached_start is not None
            and clean_end > detached_start
            and detached_start != clean_start
        ):
            variants.append((detached_start, clean_end, True))
        if not variants:
            continue

        root_action["unclean_boundary_audio"] = True
        root_action["boundary_voice_leading_seconds"] = leading_seconds
        root_action["boundary_voice_trailing_seconds"] = trailing_seconds
        if detached_start is not None:
            root_action["detached_edge_speech"] = True
        if detached_trailing:
            root_action["detached_trailing_speech"] = True

        for variant_start, variant_end, detached in variants:
            clean_key = (
                line_index,
                start_index,
                count,
                variant_start,
                variant_end,
            )
            if clean_key in occupied:
                continue
            occupied.add(clean_key)
            cleaned = dict(source_action)
            cleaned.update(
                {
                    "trim_start_sample": variant_start,
                    "trim_end_sample": variant_end,
                    "intra_segment_trim": True,
                    "boundary_voice_trim": True,
                    "unclean_boundary_audio": bool(
                        detached_start is not None and not detached
                    ),
                    "detached_leading_voice_trim": detached,
                    "detached_trailing_voice_trim": detached_trailing,
                    "duration_plausibility": _duration_plausibility(
                        line,
                        {
                            "start_seconds": variant_start / sample_rate,
                            "end_seconds": variant_end / sample_rate,
                        },
                        expected_word_count=evaluator.line_features[
                            line_index
                        ].word_count,
                    ),
                }
            )
            recovered.append(cleaned)
    return recovered


def _expand_alignment_actions(
    actions: list[dict[str, Any]],
    *,
    lines: list[dict[str, Any]],
    settings: Mapping[str, Any] | AlignmentSettings,
) -> list[dict[str, Any]]:
    """Expand only identical-row targets, not every diagnostic top match."""

    settings = AlignmentSettings.from_value(settings)
    policy = str(settings["duplicate_line_policy"])
    duplicate_indexes: dict[str, list[int]] = defaultdict(list)
    for line_index, line in enumerate(lines):
        normalized = normalize_spoken_text(
            line["line"],
            remove_parenthetical_cues=True,
        )
        if normalized:
            duplicate_indexes[normalized].append(line_index)

    expanded = []
    for action in actions:
        primary_line_index = int(action["line_index"])
        start_index = int(action["start_index"])
        count = int(action["count"])
        normalized = normalize_spoken_text(
            lines[primary_line_index]["line"],
            remove_parenthetical_cues=True,
        )
        identical_indexes = duplicate_indexes.get(
            normalized,
            [primary_line_index],
        )
        if policy in {"review", "reuse"} and len(identical_indexes) > 1:
            target_indexes = identical_indexes
        else:
            target_indexes = [primary_line_index]
        for target_index in target_indexes:
            duplicate_resolved = bool(
                action.get("duplicate_resolved", False)
                or (
                    policy == "reuse"
                    and len(identical_indexes) > 1
                )
            )
            expanded_action = dict(action)
            expanded_action.update(
                {
                    "type": "assigned",
                    "start_index": start_index,
                    "count": count,
                    "line_index": target_index,
                    "primary_line_index": target_index,
                    "segment_match_rank": 1,
                    "is_primary_match": True,
                    "duplicate_resolution": (
                        action.get("duplicate_resolution")
                        or (
                            policy
                            if len(identical_indexes) > 1
                            else None
                        )
                    ),
                    "duplicate_resolved": duplicate_resolved,
                }
            )
            expanded.append(expanded_action)

    expanded.sort(
        key=lambda action: (
            int(action["start_index"]),
            int(action["count"]),
            int(action["line_index"]),
        )
    )
    return expanded


def _reroute_actions_to_exact_asr_primary_matches(
    actions: list[dict[str, Any]],
    *,
    materialized_by_span: dict[
        tuple[int, ...],
        dict[str, Any],
    ],
    exact_scores_by_segment: dict[str, list[float]],
    minimum_score: float,
) -> list[dict[str, Any]]:
    """Choose one exact-ASR primary line per unique audio span."""

    actions_by_span: dict[
        tuple[int, int, int, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for action in actions:
        actions_by_span[_action_span_key(action)].append(action)

    routed_actions = []
    for span_key, span_actions in actions_by_span.items():
        segment = materialized_by_span.get(span_key)
        if segment is None and span_key[2:] == (-1, -1):
            # Preserve the original private-helper contract for callers that
            # still key ordinary spans by just (start_index, count).
            segment = materialized_by_span.get(span_key[:2])
        scores = (
            exact_scores_by_segment.get(str(segment["segment_id"]))
            if segment is not None
            else None
        )
        if scores:
            best_line_index = max(
                range(len(scores)),
                key=lambda line_index: scores[line_index],
            )
            best_score = float(scores[best_line_index])
        else:
            best_line_index = -1
            best_score = 0.0

        if best_score >= minimum_score:
            matching_actions = [
                action
                for action in span_actions
                if int(action["line_index"]) == best_line_index
            ]
            source = max(
                matching_actions or span_actions,
                key=lambda action: (
                    not bool(action.get("forced_review_reason")),
                    float(action.get("match_score", 0.0)),
                    bool(action.get("fragment_join", False)),
                ),
            )
            routed = dict(source)
            second_score = max(
                (
                    float(score)
                    for line_index, score in enumerate(scores)
                    if line_index != best_line_index
                ),
                default=0.0,
            )
            routed.update(
                {
                    "line_index": best_line_index,
                    "primary_line_index": best_line_index,
                    "match_score": best_score,
                    "confidence_margin": best_score - second_score,
                    "segment_match_rank": 1,
                    "is_primary_match": True,
                    "exact_asr_rerouted": (
                        int(source["line_index"]) != best_line_index
                    ),
                    "forced_review_reason": (
                        str(source.get("forced_review_reason") or "")
                        if int(source["line_index"]) == best_line_index
                        else ""
                    ),
                }
            )
        else:
            routed = dict(
                max(
                    span_actions,
                    key=lambda action: (
                        float(action.get("match_score", 0.0)),
                        bool(action.get("fragment_join", False)),
                    ),
                )
            )
            routed["primary_line_index"] = int(routed["line_index"])
            routed["segment_match_rank"] = 1
            routed["is_primary_match"] = True
        routed_actions.append(routed)

    return sorted(
        routed_actions,
        key=lambda action: (
            int(action["start_index"]),
            int(action["count"]),
            int(action["line_index"]),
        ),
    )


def _exact_line_scores(
    lines: list[dict[str, Any]],
    evaluator: TranscriptEvaluator,
    transcript: str,
) -> list[float]:
    """Score only normal dialogue lines during exact-ASR routing."""

    return [
        (
            0.0
            if is_vocalization_script(line["line"])
            else evaluator.match(line_index, transcript)
        )
        for line_index, line in enumerate(lines)
    ]


def _missing_clause_indexes(
    sentence: Mapping[str, Any],
    *,
    settings: Mapping[str, Any] | AlignmentSettings,
) -> list[int]:
    settings = AlignmentSettings.from_value(settings)
    minimum_score = float(settings.get("reliable_min_clause_score", 55.0))
    short_max_words = int(
        settings.get("reliable_short_clause_max_words", 4)
    )
    short_minimum_coverage = float(
        settings.get("reliable_short_clause_min_token_coverage", 1.0)
    )
    return [
        index
        for index, (score, coverage, word_count_value) in enumerate(
            zip(
                sentence["clause_scores"],
                sentence["clause_token_coverages"],
                sentence["clause_word_counts"],
            )
        )
        if float(score) < minimum_score
        or (
            int(word_count_value) <= short_max_words
            and float(coverage) < short_minimum_coverage
        )
    ]


def _boundary_clause_consensus_transcript(
    *,
    line_index: int,
    exact_transcript: str,
    preliminary_transcript: str,
    recording_transcript: str,
    evaluator: TranscriptEvaluator,
    settings: Mapping[str, Any] | AlignmentSettings,
) -> str | None:
    """Prefer corroborated constituent ASR for one short edge clause."""

    settings = AlignmentSettings.from_value(settings)
    if not bool(settings.get("boundary_clause_consensus_enabled", True)):
        return None
    minimum_exact_score = float(
        settings.get("boundary_clause_consensus_min_exact_score", 85.0)
    )
    minimum_support_score = float(
        settings.get("boundary_clause_consensus_min_support_score", 95.0)
    )
    maximum_clause_words = int(
        settings.get("boundary_clause_consensus_max_words", 3)
    )
    exact = evaluator.evaluate(line_index, exact_transcript)
    preliminary = evaluator.evaluate(line_index, preliminary_transcript)
    recording = evaluator.evaluate(line_index, recording_transcript)
    if (
        exact.match_score < minimum_exact_score
        or preliminary.match_score < minimum_support_score
        or int(exact.sentence["clause_count"]) < 2
        or not bool(preliminary.sentence["clauses_in_order"])
        or int(preliminary.sentence["missing_clause_count"]) > 0
        or float(preliminary.fidelity["token_coverage"]) < 0.95
        or float(preliminary.fidelity["token_precision"]) < 0.95
        or int(preliminary.fidelity["leading_extra_token_count"]) > 0
        or int(preliminary.fidelity["trailing_extra_token_count"]) > 0
    ):
        return None
    missing = _missing_clause_indexes(exact.sentence, settings=settings)
    clause_count = int(exact.sentence["clause_count"])
    if (
        len(missing) != 1
        or missing[0] not in {0, clause_count - 1}
    ):
        return None
    boundary_index = missing[0]
    expected_clauses = script_clauses(
        str(evaluator.lines[line_index]["line"])
    )
    hesitation_boundary = (
        boundary_index < len(expected_clauses)
        and "hesitation"
        in _text_features(expected_clauses[boundary_index]).tokens
    )
    if (
        not hesitation_boundary
        and int(exact.sentence["clause_word_counts"][boundary_index])
        > maximum_clause_words
    ):
        return None
    if hesitation_boundary:
        # Merged Whisper decoding commonly drops a performed hesitation even
        # when the independently decoded edge segment transcribes it.
        return preliminary_transcript
    if (
        recording.match_score < minimum_support_score
        or not bool(recording.sentence["clauses_in_order"])
    ):
        return None
    exact_clause_score = float(
        exact.sentence["clause_scores"][boundary_index]
    )
    recording_clause_score = float(
        recording.sentence["clause_scores"][boundary_index]
    )
    recording_clause_coverage = float(
        recording.sentence["clause_token_coverages"][boundary_index]
    )
    if (
        recording_clause_coverage < 0.50
        or recording_clause_score < exact_clause_score + 20.0
    ):
        return None
    recording_missing = set(
        _missing_clause_indexes(recording.sentence, settings=settings)
    )
    if any(
        index != boundary_index
        for index in recording_missing
    ):
        return None
    return preliminary_transcript


def _candidate_reliability(
    *,
    line: dict[str, Any],
    match_score: float,
    margin: float,
    settings: Mapping[str, Any] | AlignmentSettings,
    observed: str | None = None,
    ambiguity_resolved: bool = False,
    unsafe_untranscribed_merge: bool = False,
    duration_plausibility: float | None = None,
    evaluation: TranscriptEvaluation | None = None,
    technical_score: float | None = None,
    clipping_samples: int = 0,
) -> tuple[bool, str]:
    settings = AlignmentSettings.from_value(settings)
    if is_nonverbal_script(line["line"]):
        return False, "NONVERBAL_SCRIPT"
    if (
        clipping_samples > 0
        and bool(settings.get("auto_reject_clipping", True))
    ):
        return False, "TECHNICAL_CLIPPING"
    if (
        technical_score is not None
        and technical_score
        < float(settings.get("auto_min_technical_score", 0.0))
    ):
        return False, "LOW_TECHNICAL_QUALITY"
    if unsafe_untranscribed_merge:
        return False, "MERGED_UNTRANSCRIBED_AUDIO"
    if (
        duration_plausibility is not None
        and duration_plausibility
        < float(settings.get("reliable_min_duration_plausibility", 25.0))
    ):
        return False, "POSSIBLE_REPEATED_TAKES"
    if evaluation is None:
        fidelity = transcript_fidelity(
            line["line"],
            observed or "",
            token_min_similarity=float(
                settings.get("fidelity_token_min_similarity", 78.0)
            ),
            boundary_window_tokens=int(
                settings.get("reliable_boundary_window_tokens", 4)
            ),
            boundary_observed_slack_tokens=int(
                settings.get(
                    "reliable_boundary_observed_slack_tokens",
                    2,
                )
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
    else:
        fidelity = evaluation.fidelity
        sentence = evaluation.sentence
    is_short_line = (
        len(
            normalize_spoken_text(
                line["line"],
                remove_parenthetical_cues=True,
            ).split()
        )
        <= 3
    )
    if is_short_line:
        minimum_score = float(settings.get("short_line_min_score", 88.0))
        minimum_margin = float(settings.get("short_line_min_margin", 15.0))
        minimum_coverage = float(
            settings.get("short_line_min_token_coverage", 1.0)
        )
        minimum_precision = float(
            settings.get("short_line_min_token_precision", 1.0)
        )
        minimum_ordered = float(
            settings.get("short_line_min_ordered_score", 70.0)
        )
        low_score_reason = "SHORT_LINE_LOW_SCORE"
        ambiguous_reason = "SHORT_LINE_AMBIGUOUS"
        incomplete_reason = "SHORT_LINE_INCOMPLETE_TRANSCRIPT"
        extra_reason = "SHORT_LINE_EXTRA_WORDS"
        order_reason = "SHORT_LINE_ORDER_MISMATCH"
    else:
        minimum_score = float(settings.get("reliable_min_score", 72.0))
        minimum_margin = float(settings.get("reliable_min_margin", 8.0))
        minimum_coverage = float(
            settings.get("reliable_min_token_coverage", 0.60)
        )
        minimum_precision = float(
            settings.get("reliable_min_token_precision", 0.70)
        )
        minimum_ordered = float(
            settings.get("reliable_min_ordered_score", 55.0)
        )
        low_score_reason = "LOW_MATCH_SCORE"
        ambiguous_reason = "AMBIGUOUS_MATCH"
        incomplete_reason = "INCOMPLETE_TRANSCRIPT"
        extra_reason = "EXCESS_TRANSCRIPT_WORDS"
        order_reason = "LOW_ORDERED_SIMILARITY"

    if match_score < minimum_score:
        return False, low_score_reason
    if margin < minimum_margin and not ambiguity_resolved:
        return False, ambiguous_reason
    if (
        sentence["clause_count"] >= 2
        and sentence["missing_clause_count"] > 0
    ):
        return False, "MISSING_SENTENCE"
    if (
        is_short_line
        and float(fidelity["ordered_similarity"]) < minimum_ordered
    ):
        return False, order_reason
    if (
        is_short_line
        and float(fidelity["token_coverage"]) < minimum_coverage
    ):
        return False, incomplete_reason
    if float(fidelity["token_precision"]) < minimum_precision:
        return False, extra_reason
    if (
        not is_short_line
        and sentence["clause_count"] >= 2
        and not bool(sentence["clauses_in_order"])
    ):
        return False, "SENTENCE_ORDER_MISMATCH"

    if not is_short_line:
        maximum_boundary_missing_tokens = max(
            0,
            int(settings.get("reliable_max_boundary_missing_tokens", 0)),
        )
        leading_extra = int(fidelity["leading_extra_token_count"]) > 0
        trailing_extra = int(fidelity["trailing_extra_token_count"]) > 0
        leading_anchor_mismatch = bool(
            int(fidelity["leading_missing_token_count"]) > 0
            or int(fidelity["leading_substitution_count"]) > 0
        )
        trailing_anchor_mismatch = bool(
            int(fidelity["trailing_missing_token_count"]) > 0
            or int(fidelity["trailing_substitution_count"]) > 0
        )
        if leading_extra and not leading_anchor_mismatch:
            return False, "EXTRA_LINE_START"
        if trailing_extra and not trailing_anchor_mismatch:
            return False, "EXTRA_LINE_END"
        leading_boundary_mismatch = bool(
            leading_anchor_mismatch
            or int(fidelity["prefix_missing_token_count"])
            > maximum_boundary_missing_tokens
        )
        trailing_boundary_mismatch = bool(
            trailing_anchor_mismatch
            or int(fidelity["suffix_missing_token_count"])
            > maximum_boundary_missing_tokens
        )
        if leading_boundary_mismatch and trailing_boundary_mismatch:
            return False, "MISSING_LINE_BOUNDARIES"
        if leading_boundary_mismatch:
            return False, "MISSING_LINE_START"
        if trailing_boundary_mismatch:
            return False, "MISSING_LINE_END"
        if leading_extra:
            return False, "EXTRA_LINE_START"
        if trailing_extra:
            return False, "EXTRA_LINE_END"

    if (
        not is_short_line
        and float(fidelity["token_coverage"]) < minimum_coverage
    ):
        return False, incomplete_reason
    if float(fidelity["ordered_similarity"]) < minimum_ordered:
        return False, order_reason
    if (
        bool(settings.get("edge_cue_auto_review", True))
        and any(_script_edge_performance_cues(str(line["line"])))
    ):
        return False, "EDGE_VOCALIZATION_UNVERIFIED"
    return True, ""


def _preliminary_transcript_shows_collapsed_repetition(
    preliminary: str,
    exact: str,
) -> bool:
    preliminary_tokens = _text_features(preliminary).tokens
    exact_tokens = _text_features(exact).tokens
    if not preliminary_tokens or not exact_tokens:
        return False
    preliminary_counts = Counter(preliminary_tokens)
    exact_counts = Counter(exact_tokens)
    matched_words = sum(
        min(count, preliminary_counts.get(token, 0))
        for token, count in exact_counts.items()
    )
    extra_words = len(preliminary_tokens) - matched_words
    repeated_exact_words = sum(
        max(0, preliminary_counts.get(token, 0) - count)
        for token, count in exact_counts.items()
    )
    minimum_repeated_words = max(1, math.ceil(len(exact_tokens) * 0.5))
    return bool(
        extra_words >= minimum_repeated_words
        and repeated_exact_words >= minimum_repeated_words
    )


def _has_unsafe_untranscribed_merge(
    *,
    action: dict[str, Any],
    base_segments: list[dict[str, Any]],
    settings: Mapping[str, Any] | AlignmentSettings,
    segment: dict[str, Any] | None = None,
) -> bool:
    settings = AlignmentSettings.from_value(settings)
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
    for base_segment in base_segments[start_index : start_index + count]:
        silence_rejected = bool(
            ((base_segment.get("segment_asr") or {}).get("silence_rejected"))
        )
        if (
            str(base_segment.get("transcript") or "").strip()
            and not silence_rejected
        ):
            continue
        duration = float(
            (base_segment.get("metrics") or {}).get("duration_seconds")
            or (
                float(base_segment.get("end_seconds", 0.0))
                - float(base_segment.get("start_seconds", 0.0))
            )
        )
        rms = (base_segment.get("metrics") or {}).get("rms_dbfs")
        if (
            duration >= minimum_seconds
            and (rms is None or float(rms) >= minimum_rms)
        ):
            return True
    if segment is not None:
        exact_tokens = set(
            _text_features(str(segment.get("transcript") or "")).tokens
        )
        edge_segments = [
            base_segments[start_index],
            base_segments[start_index + count - 1],
        ]
        for edge_segment in edge_segments:
            edge_transcript = _segment_transcript(edge_segment)
            edge_tokens = set(_text_features(edge_transcript).tokens)
            if not edge_tokens or not exact_tokens.isdisjoint(edge_tokens):
                continue
            if _is_paralinguistic_transcript(edge_transcript):
                return True
            edge_duration = float(
                (edge_segment.get("metrics") or {}).get("duration_seconds")
                or (
                    float(edge_segment.get("end_seconds", 0.0))
                    - float(edge_segment.get("start_seconds", 0.0))
                )
            )
            edge_rms = (edge_segment.get("metrics") or {}).get("rms_dbfs")
            # Exact-span Whisper can omit a long audible tail while an
            # independent base decode hallucinates nonempty text for it. Such
            # a segment is just as unsafe as one with an empty transcript.
            if (
                edge_duration >= max(2.0, minimum_seconds * 3.0)
                and (edge_rms is None or float(edge_rms) >= minimum_rms)
            ):
                return True
    if (
        segment is not None
        and _preliminary_transcript_shows_collapsed_repetition(
            str(action.get("transcript") or ""),
            str(segment.get("transcript") or ""),
        )
    ):
        words = [
            word
            for word in segment.get("words") or []
            if word.get("start") is not None and word.get("end") is not None
        ]
        if words:
            duration = float(
                (segment.get("metrics") or {}).get("duration_seconds")
                or (
                    float(segment.get("end_seconds", 0.0))
                    - float(segment.get("start_seconds", 0.0))
                )
            )
            leading_untranscribed = max(0.0, float(words[0]["start"]))
            trailing_untranscribed = max(
                0.0,
                duration - float(words[-1]["end"]),
            )
            if (
                leading_untranscribed >= minimum_seconds
                or trailing_untranscribed >= minimum_seconds
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
    settings = AlignmentSettings.from_value(project.get("alignment"))
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
        check_processing_cancelled()
        session_entry = manifest_session_by_id.get(session["id"])
        if not session_entry:
            raise KeyError(f"Segments missing for session: {session['id']}")
        base_segments = session_entry.get("segments", [])
        session_lines = lines_for_session(source_data, session)
        if not session_lines:
            continue
        evaluator = TranscriptEvaluator(session_lines, settings)
        print(
            f"[align {session_index}] {session['id']}: "
            f"{len(base_segments)} segments → {len(session_lines)} lines",
            flush=True,
        )
        transcription_path = (
            project_dir / "transcripts" / f"{session['id']}.json"
        )
        session_transcription = (
            read_json(transcription_path)
            if transcription_path.is_file()
            else None
        )
        span_catalog = SpanCatalog(
            base_segments,
            settings,
            transcription=session_transcription,
        )
        primary_actions = order_independent_align(
            base_segments,
            session_lines,
            settings,
            evaluator=evaluator,
            span_catalog=span_catalog,
        )
        check_processing_cancelled()
        boundary_noise_cleanup_actions = _boundary_noise_cleanup_actions(
            primary_actions,
            lines=session_lines,
            base_segments=base_segments,
            settings=settings,
            evaluator=evaluator,
            span_catalog=span_catalog,
        )
        complete_subspan_actions = _complete_subspan_recovery_actions(
            primary_actions,
            lines=session_lines,
            settings=settings,
            evaluator=evaluator,
            span_catalog=span_catalog,
        )
        edge_vocalization_extension_actions = (
            _edge_vocalization_extension_actions(
                primary_actions,
                lines=session_lines,
                base_segments=base_segments,
                settings=settings,
                evaluator=evaluator,
                span_catalog=span_catalog,
            )
        )
        check_processing_cancelled()
        fragment_join_actions = _multisentence_fragment_join_actions(
            primary_actions,
            lines=session_lines,
            base_segments=base_segments,
            settings=settings,
            transcription=session_transcription,
            evaluator=evaluator,
            span_catalog=span_catalog,
            project_dir=project_dir,
            segmentation_settings=project.get("segmentation"),
        )
        check_processing_cancelled()
        intra_segment_trim_actions = _intra_segment_trim_actions(
            primary_actions,
            lines=session_lines,
            base_segments=base_segments,
            sample_rate=int(session_entry["sample_rate"]),
            settings=settings,
            evaluator=evaluator,
            project_dir=project_dir,
            segmentation_settings=project.get("segmentation"),
        )
        check_processing_cancelled()
        candidate_actions = [
            *primary_actions,
            *boundary_noise_cleanup_actions,
            *complete_subspan_actions,
            *edge_vocalization_extension_actions,
            *fragment_join_actions,
            *intra_segment_trim_actions,
        ]
        boundary_voice_trim_actions = _boundary_voice_trim_actions(
            candidate_actions,
            project_dir=project_dir,
            session_entry=session_entry,
            lines=session_lines,
            base_segments=base_segments,
            settings=settings,
            evaluator=evaluator,
            segmentation_settings=project.get("segmentation"),
        )
        candidate_actions = sorted(
            [*candidate_actions, *boundary_voice_trim_actions],
            key=lambda action: (
                int(action["start_index"]),
                int(action["count"]),
                int(action.get("trim_start_sample", -1)),
                int(action.get("trim_end_sample", -1)),
            ),
        )
        actions = candidate_actions
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
            tuple[int, int, int, int],
            dict[str, Any],
        ] = {}
        verification_segments: dict[str, dict[str, Any]] = {}
        for action in actions:
            check_processing_cancelled()
            span_key = _action_span_key(action)
            segment = materialized_by_span.get(span_key)
            if segment is None:
                if span_key[2:] != (-1, -1):
                    segment = materialize_trimmed_segment(
                        project_dir=project_dir,
                        project=project,
                        session_entry=session_entry,
                        base_segments=base_segments,
                        base_index=span_key[0],
                        count=span_key[1],
                        start_sample=span_key[2],
                        end_sample=span_key[3],
                        transcript=str(action.get("transcript") or ""),
                        asr_probability=None,
                    )
                else:
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
            normalize_spoken_text(
                line["line"],
                remove_parenthetical_cues=True,
            )
            for line in session_lines
        ]
        exact_scores_by_segment: dict[str, list[float]] = {}
        for segment_id, exact_asr in exact_asr_by_segment.items():
            check_processing_cancelled()
            exact_text = str(exact_asr.get("transcript") or "").strip()
            if exact_text and not exact_asr.get("error"):
                exact_scores_by_segment[segment_id] = _exact_line_scores(
                    session_lines,
                    evaluator,
                    exact_text,
                )
        actions = _reroute_actions_to_exact_asr_primary_matches(
            actions,
            materialized_by_span=materialized_by_span,
            exact_scores_by_segment=exact_scores_by_segment,
            minimum_score=float(settings.get("candidate_min_score", 45.0)),
        )
        _apply_duplicate_line_policy(
            actions,
            lines=session_lines,
            base_segments=base_segments,
            settings=settings,
        )
        actions = _expand_alignment_actions(
            actions,
            lines=session_lines,
            settings=settings,
        )
        reliable_coverage: set[int] = set()
        assignment_by_base: dict[int, list[dict[str, Any]]] = defaultdict(list)
        serialized_actions = []

        for action in actions:
            check_processing_cancelled()
            line = session_lines[action["line_index"]]
            segment = materialized_by_span[_action_span_key(action)]
            preliminary_transcript = str(
                action.get("transcript")
                or segment.get("transcript")
                or ""
            )
            observed_transcript = preliminary_transcript
            match_score = float(action["match_score"])
            exact_span_asr_verified = False
            exact_span_asr_error = ""
            candidate_scores: list[float] | None = None
            boundary_clause_consensus = False
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
                        candidate_scores = exact_scores_by_segment[
                            str(segment["segment_id"])
                        ]
                        match_score = candidate_scores[
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

            if (
                exact_span_asr_verified
                and int(action["count"]) > 1
                and action.get("transcript_source")
                == "segment_asr_span"
            ):
                recording_preview = (
                    span_catalog.transcription_preview_for_bounds(
                        start_seconds=float(segment["start_seconds"]),
                        end_seconds=float(segment["end_seconds"]),
                    )
                )
                consensus_transcript = (
                    _boundary_clause_consensus_transcript(
                        line_index=int(action["line_index"]),
                        exact_transcript=observed_transcript,
                        preliminary_transcript=preliminary_transcript,
                        recording_transcript=str(
                            (recording_preview or {}).get("transcript")
                            or ""
                        ),
                        evaluator=evaluator,
                        settings=settings,
                    )
                )
                if consensus_transcript is not None:
                    observed_transcript = consensus_transcript
                    candidate_scores = _exact_line_scores(
                        session_lines,
                        evaluator,
                        observed_transcript,
                    )
                    match_score = candidate_scores[
                        int(action["line_index"])
                    ]
                    transcript_source = (
                        "constituent_recording_boundary_consensus"
                    )
                    boundary_clause_consensus = True

            if exact_span_asr_verified:
                assert candidate_scores is not None
                second_score = max(
                    (
                        score
                        for other_index, score in enumerate(candidate_scores)
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
                        evaluator.match(other_index, observed_transcript),
                        other_line["line_id"],
                    )
                    for other_index, other_line in enumerate(session_lines)
                    if other_line["line_id"] != line["line_id"]
                    and not is_vocalization_script(other_line["line"])
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
                and normalize_spoken_text(
                    primary_line["line"],
                    remove_parenthetical_cues=True,
                )
                == normalize_spoken_text(
                    line["line"],
                    remove_parenthetical_cues=True,
                )
            )
            evaluation = evaluator.evaluate(
                int(action["line_index"]),
                observed_transcript,
            )
            fidelity = evaluation.fidelity
            sentence = evaluation.sentence
            unsafe_untranscribed_merge = _has_unsafe_untranscribed_merge(
                action=action,
                base_segments=base_segments,
                settings=settings,
                segment=segment,
            )
            technical_score = _technical_score(segment)
            reliable, reason = _candidate_reliability(
                line=line,
                match_score=match_score,
                margin=margin,
                settings=settings,
                observed=observed_transcript,
                ambiguity_resolved=bool(
                    action.get("duplicate_resolved", False)
                    or reusable_duplicate
                ),
                unsafe_untranscribed_merge=unsafe_untranscribed_merge,
                duration_plausibility=(
                    float(action["duration_plausibility"])
                    if action.get("duration_plausibility") is not None
                    else None
                ),
                evaluation=evaluation,
                technical_score=technical_score,
                clipping_samples=int(
                    (segment.get("metrics") or {}).get(
                        "clipping_samples",
                        0,
                    )
                    or 0
                ),
            )
            if verification_required and not exact_span_asr_verified:
                reliable = False
                reason = "EXACT_SPAN_ASR_FAILED"
            candidate_is_primary = bool(
                action.get("is_primary_match", True)
            )
            if exact_span_asr_verified:
                assert candidate_scores is not None
                line_index = int(action["line_index"])
                best_other_score = max(
                    (
                        score
                        for other_index, score in enumerate(candidate_scores)
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
            if action.get("unclean_boundary_audio") and reliable:
                reliable = False
                reason = "EDGE_NON_SPEECH_AUDIO"
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
                "prefix_token_coverage": fidelity[
                    "prefix_token_coverage"
                ],
                "suffix_token_coverage": fidelity[
                    "suffix_token_coverage"
                ],
                "prefix_missing_token_count": fidelity[
                    "prefix_missing_token_count"
                ],
                "suffix_missing_token_count": fidelity[
                    "suffix_missing_token_count"
                ],
                "leading_missing_token_count": fidelity[
                    "leading_missing_token_count"
                ],
                "leading_extra_token_count": fidelity[
                    "leading_extra_token_count"
                ],
                "trailing_missing_token_count": fidelity[
                    "trailing_missing_token_count"
                ],
                "trailing_extra_token_count": fidelity[
                    "trailing_extra_token_count"
                ],
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
                "duration_plausibility": float(
                    action.get("duration_plausibility", 0.0)
                ),
                "order_hint": float(action.get("order_hint", 0.0)),
                "duplicate_resolution": action.get("duplicate_resolution"),
                "transcript_source": transcript_source,
                "exact_span_asr_verified": exact_span_asr_verified,
                "exact_span_asr_error": exact_span_asr_error,
                "boundary_clause_consensus": boundary_clause_consensus,
                "unsafe_untranscribed_merge": unsafe_untranscribed_merge,
                "fragment_join": bool(action.get("fragment_join", False)),
                "fragment_source_count": int(
                    action.get("fragment_source_count", 0)
                ),
                "fragment_join_provisional": bool(
                    action.get("fragment_join_provisional", False)
                ),
                "fragment_join_fallback": bool(
                    action.get("fragment_join_fallback", False)
                ),
                "intra_segment_trim": bool(
                    action.get("intra_segment_trim", False)
                ),
                "repeated_take_trim": bool(
                    action.get("repeated_take_trim", False)
                ),
                "acoustic_take_trim": bool(
                    action.get("acoustic_take_trim", False)
                ),
                "trimmed_edge_join": bool(
                    action.get("trimmed_edge_join", False)
                ),
                "boundary_noise_cleanup": bool(
                    action.get("boundary_noise_cleanup", False)
                ),
                "complete_subspan_recovery": bool(
                    action.get("complete_subspan_recovery", False)
                ),
                "boundary_voice_trim": bool(
                    action.get("boundary_voice_trim", False)
                ),
                "detached_edge_speech": bool(
                    action.get("detached_edge_speech", False)
                ),
                "detached_leading_voice_trim": bool(
                    action.get("detached_leading_voice_trim", False)
                ),
                "detached_trailing_voice_trim": bool(
                    action.get("detached_trailing_voice_trim", False)
                ),
                "unclean_boundary_audio": bool(
                    action.get("unclean_boundary_audio", False)
                ),
                "boundary_voice_leading_seconds": action.get(
                    "boundary_voice_leading_seconds"
                ),
                "boundary_voice_trailing_seconds": action.get(
                    "boundary_voice_trailing_seconds"
                ),
                "edge_vocalization_extension": bool(
                    action.get("edge_vocalization_extension", False)
                ),
            }
            if (
                bool(action.get("fragment_join_fallback"))
                and unsafe_untranscribed_merge
            ):
                continue
            if match_score >= float(
                settings.get("candidate_min_score", 45.0)
            ):
                candidates.append(candidate)
            for base_index in segment["base_indices"]:
                assignment_by_base[base_index].append(candidate)
                if reliable:
                    reliable_coverage.add(base_index)
            serialized_actions.append(candidate)

        for base_index, segment in enumerate(base_segments):
            check_processing_cancelled()
            if base_index in reliable_coverage:
                continue
            transcript = segment.get("transcript", "")
            suggestions = sorted(
                (
                    evaluator.match(line_index, transcript),
                    line["line_id"],
                    line["line"],
                )
                for line_index, line in enumerate(session_lines)
                if not is_vocalization_script(line["line"])
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
        check_processing_cancelled()
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
    check_processing_cancelled()
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
