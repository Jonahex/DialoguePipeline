from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


# This is the single source of truth for alignment defaults. Existing projects
# may continue to use the original flat keys; newly created projects use the
# smaller grouped configuration returned by default_alignment_config().
ALIGNMENT_DEFAULTS: dict[str, Any] = {
    "max_merge_segments": 10,
    "max_merge_gap_seconds": 2.5,
    "max_span_seconds": 35.0,
    "candidate_top_k": 8,
    "take_group_gap_seconds": 12.0,
    "duration_hint_weight": 1.0,
    "order_hint_weight": 0.0,
    "merge_span_penalty": 0.2,
    "merge_require_text_boundaries": True,
    "primary_match_bonus": 2.0,
    "duplicate_line_policy": "weak_order",
    "candidate_min_score": 45.0,
    "reliable_min_score": 72.0,
    "reliable_min_margin": 8.0,
    "reliable_min_ordered_score": 55.0,
    "reliable_min_token_coverage": 0.60,
    "reliable_min_token_precision": 0.70,
    "reliable_boundary_window_tokens": 4,
    "reliable_boundary_observed_slack_tokens": 2,
    "reliable_max_boundary_missing_tokens": 0,
    "reliable_min_clause_score": 55.0,
    "reliable_short_clause_max_words": 4,
    "reliable_short_clause_min_token_coverage": 1.0,
    "reliable_min_duration_plausibility": 25.0,
    "clause_completeness_weight": 5.0,
    "fragment_join_enabled": True,
    "fragment_join_max_actions": 10,
    "fragment_join_fallback_max_actions": 2,
    "fragment_join_fallback_min_match_score": 90.0,
    "fragment_join_max_segments": 10,
    "fragment_join_require_text_boundaries": True,
    "fragment_join_only_incomplete_lines": True,
    "fragment_join_neighbor_radius": 1,
    "fragment_join_min_token_coverage": 0.85,
    "fragment_join_min_ordered_score": 70.0,
    "fragment_join_min_token_precision": 0.83,
    "fragment_join_provisional_min_match_score": 45.0,
    "fragment_join_provisional_min_token_coverage": 0.70,
    "fragment_join_provisional_min_ordered_score": 55.0,
    "fragment_join_provisional_min_token_precision": 0.65,
    "fragment_join_secondary_seed_min_match_score": 80.0,
    "fragment_join_complete_min_match_score": 72.0,
    "fragment_join_complete_min_ordered_score": 70.0,
    "fragment_join_complete_min_length_ratio": 0.75,
    "fragment_join_complete_max_length_ratio": 1.35,
    "intra_segment_trim_enabled": True,
    "intra_segment_trim_max_actions_per_line": 2,
    "intra_segment_trim_max_actions_per_segment": 3,
    "intra_segment_trim_min_gap_seconds": 0.40,
    "intra_segment_trim_min_match_score": 85.0,
    "intra_segment_trim_min_token_coverage": 0.85,
    "intra_segment_trim_min_token_precision": 0.85,
    "intra_segment_trim_min_ordered_score": 70.0,
    "span_word_min_overlap_seconds": 0.20,
    "span_word_min_overlap_fraction": 0.20,
    "short_line_min_score": 88.0,
    "short_line_min_margin": 15.0,
    "short_line_min_ordered_score": 70.0,
    "short_line_min_token_coverage": 1.0,
    "short_line_min_token_precision": 1.0,
    "fidelity_token_min_similarity": 78.0,
    "auto_reject_untranscribed_merge": True,
    "untranscribed_merge_min_seconds": 0.5,
    "untranscribed_merge_min_rms_dbfs": -45.0,
    "noise_penalty": 2.2,
    "auto_reject_clipping": True,
    "auto_min_technical_score": 0.0,
}


_GROUPED_KEYS: dict[tuple[str, ...], str] = {
    ("span_search", "max_segments"): "max_merge_segments",
    ("span_search", "max_gap_seconds"): "max_merge_gap_seconds",
    ("span_search", "max_duration_seconds"): "max_span_seconds",
    ("span_search", "candidate_top_k"): "candidate_top_k",
    ("span_search", "minimum_score"): "candidate_min_score",
    ("span_search", "require_text_boundaries"): "merge_require_text_boundaries",
    ("ranking", "duration_hint_weight"): "duration_hint_weight",
    ("ranking", "order_hint_weight"): "order_hint_weight",
    ("ranking", "merge_span_penalty"): "merge_span_penalty",
    ("ranking", "noise_penalty"): "noise_penalty",
    ("ranking", "primary_match_bonus"): "primary_match_bonus",
    ("ranking", "clause_completeness_weight"): "clause_completeness_weight",
    ("duplicates", "policy"): "duplicate_line_policy",
    ("duplicates", "take_group_gap_seconds"): "take_group_gap_seconds",
    ("reliability", "normal", "minimum_score"): "reliable_min_score",
    ("reliability", "normal", "minimum_margin"): "reliable_min_margin",
    ("reliability", "normal", "minimum_ordered_score"): (
        "reliable_min_ordered_score"
    ),
    ("reliability", "normal", "minimum_token_coverage"): (
        "reliable_min_token_coverage"
    ),
    ("reliability", "normal", "minimum_token_precision"): (
        "reliable_min_token_precision"
    ),
    ("reliability", "short", "minimum_score"): "short_line_min_score",
    ("reliability", "short", "minimum_margin"): "short_line_min_margin",
    ("reliability", "short", "minimum_ordered_score"): (
        "short_line_min_ordered_score"
    ),
    ("reliability", "short", "minimum_token_coverage"): (
        "short_line_min_token_coverage"
    ),
    ("reliability", "short", "minimum_token_precision"): (
        "short_line_min_token_precision"
    ),
    ("reliability", "boundary", "window_tokens"): (
        "reliable_boundary_window_tokens"
    ),
    ("reliability", "boundary", "observed_slack_tokens"): (
        "reliable_boundary_observed_slack_tokens"
    ),
    ("reliability", "boundary", "maximum_missing_tokens"): (
        "reliable_max_boundary_missing_tokens"
    ),
    ("reliability", "clauses", "minimum_score"): "reliable_min_clause_score",
    ("reliability", "minimum_duration_plausibility"): (
        "reliable_min_duration_plausibility"
    ),
    ("reliability", "fuzzy_token_minimum_similarity"): (
        "fidelity_token_min_similarity"
    ),
    ("recovery", "enabled"): "fragment_join_enabled",
    ("recovery", "max_candidates_per_line"): "fragment_join_max_actions",
    ("recovery", "fallback_candidates_per_line"): (
        "fragment_join_fallback_max_actions"
    ),
    ("recovery", "fallback_minimum_score"): (
        "fragment_join_fallback_min_match_score"
    ),
    ("recovery", "max_segments"): "fragment_join_max_segments",
    ("recovery", "neighbor_radius"): "fragment_join_neighbor_radius",
    ("recovery", "trim_oversized_segments"): "intra_segment_trim_enabled",
    ("recovery", "trim_candidates_per_line"): (
        "intra_segment_trim_max_actions_per_line"
    ),
    ("recovery", "trim_minimum_gap_seconds"): (
        "intra_segment_trim_min_gap_seconds"
    ),
    ("quality", "reject_clipping"): "auto_reject_clipping",
    ("quality", "minimum_technical_score"): "auto_min_technical_score",
    ("quality", "reject_untranscribed_merge"): (
        "auto_reject_untranscribed_merge"
    ),
    ("quality", "untranscribed_min_seconds"): (
        "untranscribed_merge_min_seconds"
    ),
    ("quality", "untranscribed_min_rms_dbfs"): (
        "untranscribed_merge_min_rms_dbfs"
    ),
}


def default_alignment_config() -> dict[str, Any]:
    """Return the compact, grouped configuration written for new projects."""

    return {
        "span_search": {
            "max_segments": 10,
            "max_gap_seconds": 2.5,
            "max_duration_seconds": 35.0,
            "candidate_top_k": 8,
            "minimum_score": 45.0,
            "require_text_boundaries": True,
        },
        "ranking": {
            "duration_hint_weight": 1.0,
            "order_hint_weight": 0.0,
            "merge_span_penalty": 0.2,
            "noise_penalty": 2.2,
            "primary_match_bonus": 2.0,
            "clause_completeness_weight": 5.0,
        },
        "duplicates": {
            "policy": "weak_order",
            "take_group_gap_seconds": 12.0,
        },
        "reliability": {
            "normal": {
                "minimum_score": 72.0,
                "minimum_margin": 8.0,
                "minimum_ordered_score": 55.0,
                "minimum_token_coverage": 0.60,
                "minimum_token_precision": 0.70,
            },
            "short": {
                "minimum_score": 88.0,
                "minimum_margin": 15.0,
                "minimum_ordered_score": 70.0,
                "minimum_token_coverage": 1.0,
                "minimum_token_precision": 1.0,
            },
            "boundary": {
                "window_tokens": 4,
                "observed_slack_tokens": 2,
                "maximum_missing_tokens": 0,
            },
            "clauses": {"minimum_score": 55.0},
            "minimum_duration_plausibility": 25.0,
            "fuzzy_token_minimum_similarity": 78.0,
        },
        "recovery": {
            "enabled": True,
            "max_candidates_per_line": 10,
            "fallback_candidates_per_line": 2,
            "fallback_minimum_score": 90.0,
            "max_segments": 10,
            "neighbor_radius": 1,
            "trim_oversized_segments": True,
            "trim_candidates_per_line": 2,
            "trim_minimum_gap_seconds": 0.40,
        },
        "quality": {
            "reject_clipping": True,
            "minimum_technical_score": 0.0,
            "reject_untranscribed_merge": True,
            "untranscribed_min_seconds": 0.5,
            "untranscribed_min_rms_dbfs": -45.0,
        },
    }


@dataclass(frozen=True)
class AlignmentSettings(Mapping[str, Any]):
    """Validated flat view over grouped or legacy alignment configuration."""

    _values: dict[str, Any]

    @classmethod
    def from_value(
        cls,
        configured: Mapping[str, Any] | "AlignmentSettings" | None,
    ) -> "AlignmentSettings":
        if isinstance(configured, cls):
            return configured
        raw = dict(configured or {})
        values = dict(ALIGNMENT_DEFAULTS)

        for path, legacy_key in _GROUPED_KEYS.items():
            current: Any = raw
            for component in path:
                if not isinstance(current, Mapping) or component not in current:
                    break
                current = current[component]
            else:
                values[legacy_key] = current

        # Flat legacy values deliberately win when both forms are present.
        for key in ALIGNMENT_DEFAULTS:
            if key in raw:
                values[key] = raw[key]

        cls._validate(values)
        return cls(values)

    @staticmethod
    def _validate(values: dict[str, Any]) -> None:
        policy = str(values["duplicate_line_policy"]).lower()
        if policy not in {"review", "weak_order", "reuse"}:
            raise ValueError(
                "alignment duplicate policy must be 'review', "
                f"'weak_order', or 'reuse', got {policy!r}"
            )
        values["duplicate_line_policy"] = policy

        for key in (
            "max_merge_segments",
            "candidate_top_k",
            "fragment_join_max_actions",
            "fragment_join_max_segments",
            "reliable_boundary_window_tokens",
        ):
            if int(values[key]) < 1:
                raise ValueError(f"alignment.{key} must be at least 1")
        for key in (
            "fragment_join_fallback_max_actions",
            "intra_segment_trim_max_actions_per_line",
            "intra_segment_trim_max_actions_per_segment",
        ):
            if int(values[key]) < 0:
                raise ValueError(f"alignment.{key} cannot be negative")
        for key in (
            "max_merge_gap_seconds",
            "max_span_seconds",
            "take_group_gap_seconds",
            "intra_segment_trim_min_gap_seconds",
            "untranscribed_merge_min_seconds",
        ):
            if float(values[key]) < 0.0:
                raise ValueError(f"alignment.{key} cannot be negative")
        for key in (
            "candidate_min_score",
            "reliable_min_score",
            "reliable_min_ordered_score",
            "reliable_min_clause_score",
            "short_line_min_score",
            "short_line_min_ordered_score",
            "fidelity_token_min_similarity",
            "fragment_join_fallback_min_match_score",
            "intra_segment_trim_min_match_score",
            "intra_segment_trim_min_ordered_score",
            "auto_min_technical_score",
        ):
            if not 0.0 <= float(values[key]) <= 100.0:
                raise ValueError(f"alignment.{key} must be between 0 and 100")
        for key in (
            "reliable_min_token_coverage",
            "reliable_min_token_precision",
            "short_line_min_token_coverage",
            "short_line_min_token_precision",
            "intra_segment_trim_min_token_coverage",
            "intra_segment_trim_min_token_precision",
        ):
            if not 0.0 <= float(values[key]) <= 1.0:
                raise ValueError(f"alignment.{key} must be between 0 and 1")

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)
