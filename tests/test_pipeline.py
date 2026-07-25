from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from openpyxl import load_workbook

from dialogue_pipeline.alignment import (
    _apply_duplicate_line_policy,
    _apply_local_asr_rescue,
    _candidate_reliability,
    _has_unsafe_untranscribed_merge,
    _multisentence_fragment_join_actions,
    _vocalization_review_actions,
    _write_review_workbook,
    align_project,
    order_independent_align,
    sentence_fidelity,
    sequence_align,
    text_similarity,
    transcript_fidelity,
)
from dialogue_pipeline.audio import cut_pcm_wav, prepare_pcm_segmentation_source
from dialogue_pipeline.finalize import finalize_review
from dialogue_pipeline.segmentation import (
    segment_project,
    split_regions_on_word_gaps,
)
from dialogue_pipeline.transcription import (
    transcribe_candidate_span,
    transcribe_segments_project,
)
from dialogue_pipeline.util import (
    default_model_cache_root,
    read_json,
    resolve_model_cache_root,
    sha256_file,
    write_json,
)
from dialogue_pipeline.workbook_io import parse_workbook


def _write_tone(path: Path, duration_seconds: float = 1.0) -> None:
    sample_rate = 48000
    time = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    samples = np.rint(np.sin(2 * math.pi * 440 * time) * 8000).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())


def _write_pattern(path: Path) -> None:
    sample_rate = 48000
    total = np.zeros(sample_rate * 5, dtype="<i2")
    for start, end in ((0.25, 0.9), (1.4, 2.05), (3.0, 3.75)):
        first = int(start * sample_rate)
        last = int(end * sample_rate)
        time = np.arange(last - first) / sample_rate
        total[first:last] = np.rint(
            np.sin(2 * math.pi * 440 * time) * 8000
        ).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(total.tobytes())


def test_shared_model_cache_resolution(tmp_path: Path, monkeypatch) -> None:
    shared = tmp_path / "shared-model-cache"
    monkeypatch.setenv("DIALOGUE_VA_MODEL_CACHE", str(shared))
    assert default_model_cache_root() == shared.resolve()
    assert resolve_model_cache_root(tmp_path, {}) == shared.resolve()
    assert resolve_model_cache_root(
        tmp_path,
        {"model_cache": "project-override"},
    ) == (tmp_path / "project-override").resolve()


def test_float_wav_normalization(tmp_path: Path) -> None:
    integer_source = tmp_path / "integer.wav"
    float_source = tmp_path / "float.wav"
    normalized = tmp_path / "normalized.wav"
    _write_tone(integer_source, duration_seconds=0.25)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(integer_source),
            "-ar",
            "44100",
            "-c:a",
            "pcm_f32le",
            str(float_source),
        ],
        check=True,
    )

    working, shape, was_normalized = prepare_pcm_segmentation_source(
        float_source,
        normalized,
        sample_rate=48000,
        channels=1,
        bits_per_sample=16,
    )
    assert working == normalized
    assert was_normalized
    assert shape["sample_rate"] == 48000
    assert shape["channels"] == 1
    assert shape["bits_per_sample"] == 16
    with wave.open(str(normalized), "rb") as reader:
        assert reader.getframerate() == 48000
        assert reader.getsampwidth() == 2


def test_sample_workbook_schema() -> None:
    workbook_path = (
        Path(__file__).parents[1] / "MaleElfYoung" / "ARG1RMElfYoung.xlsm"
    )
    result = parse_workbook(workbook_path)
    assert result["sheet_count"] == 21
    assert result["line_count"] == 457
    assert len({line["target_filename"] for line in result["lines"]}) == 457


def test_text_similarity_and_repeated_take_alignment() -> None:
    lines = [
        {"line_id": "a", "line": "Hello there."},
        {"line_id": "b", "line": "Where are you going?"},
        {"line_id": "c", "line": "Goodbye."},
    ]
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "transcript": "Hello there.",
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 1.5,
            "end_seconds": 2.5,
            "transcript": "Hello there!",
            "asr_probability": 0.94,
        },
        {
            "start_seconds": 3.0,
            "end_seconds": 4.0,
            "transcript": "Where are you going?",
            "asr_probability": 0.96,
        },
        {
            "start_seconds": 4.5,
            "end_seconds": 5.2,
            "transcript": "Goodbye.",
            "asr_probability": 0.98,
        },
    ]
    settings = {
        "max_merge_segments": 2,
        "max_merge_gap_seconds": 2.0,
        "max_span_seconds": 20.0,
        "lookahead_lines": 3,
        "path_min_score": 35.0,
        "noise_penalty": 2.2,
        "skip_line_penalty": 1.8,
        "repeat_take_penalty": 0.8,
    }
    actions = sequence_align(segments, lines, settings)
    assert [action["line_index"] for action in actions] == [0, 0, 1, 2]
    assert text_similarity("Where are you going?", "where are you going") > 95


def test_order_independent_alignment_handles_reordered_lines_and_takes() -> None:
    lines = [
        {"line_id": "a", "line": "Hello there."},
        {"line_id": "b", "line": "Where are you going?"},
        {"line_id": "c", "line": "Goodbye friend."},
    ]
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "transcript": "Goodbye friend.",
            "asr_probability": 0.98,
        },
        {
            "start_seconds": 1.5,
            "end_seconds": 2.5,
            "transcript": "Hello there.",
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 3.0,
            "end_seconds": 4.0,
            "transcript": "Goodbye friend!",
            "asr_probability": 0.97,
        },
        {
            "start_seconds": 4.5,
            "end_seconds": 5.8,
            "transcript": "Where are you going?",
            "asr_probability": 0.96,
        },
    ]
    settings = {
        "max_merge_segments": 2,
        "max_merge_gap_seconds": 2.0,
        "max_span_seconds": 20.0,
        "path_min_score": 35.0,
        "candidate_min_score": 45.0,
        "candidate_top_k": 3,
        "noise_penalty": 2.2,
        "duration_hint_weight": 1.0,
        "order_hint_weight": 0.0,
    }

    actions = order_independent_align(segments, lines, settings)

    assert [action["line_index"] for action in actions] == [2, 0, 2, 1]
    assert all(
        action["top_matches"][0]["line_index"] == action["line_index"]
        for action in actions
    )


def test_text_similarity_handles_reordered_and_repeated_take_text() -> None:
    assert text_similarity("Good afternoon.", "Afternoon Good") > 88
    assert text_similarity("Take this!", "This Take This") > 88
    assert text_similarity("Run away! Run away!", "Runaway Runaway") >= 72


def test_exact_short_match_is_reliable_unless_script_text_is_duplicated() -> None:
    settings = {
        "short_line_min_score": 88.0,
        "short_line_min_margin": 15.0,
    }
    line = {"line": "Goodbye."}

    assert _candidate_reliability(
        line=line,
        match_score=100.0,
        margin=5.0,
        settings=settings,
        observed="Goodbye",
        duplicate_text=False,
    ) == (True, "")
    assert _candidate_reliability(
        line=line,
        match_score=100.0,
        margin=5.0,
        settings=settings,
        observed="Goodbye",
        duplicate_text=True,
    ) == (False, "SHORT_LINE_AMBIGUOUS")


def test_short_line_auto_reliability_requires_order_and_no_extra_words() -> None:
    settings = {
        "short_line_min_score": 88.0,
        "short_line_min_margin": 15.0,
        "short_line_min_ordered_score": 70.0,
        "short_line_min_token_coverage": 1.0,
        "short_line_min_token_precision": 1.0,
    }
    line = {"line": "Not... remember..."}

    assert text_similarity(line["line"], "Remember Not") >= 88.0
    assert _candidate_reliability(
        line=line,
        match_score=text_similarity(line["line"], "Remember Not"),
        margin=40.0,
        settings=settings,
        observed="Remember Not",
    ) == (False, "SHORT_LINE_ORDER_MISMATCH")
    assert _candidate_reliability(
        line=line,
        match_score=text_similarity(line["line"], "Not remember not"),
        margin=40.0,
        settings=settings,
        observed="Not remember not",
    ) == (False, "SHORT_LINE_EXTRA_WORDS")
    assert _candidate_reliability(
        line=line,
        match_score=100.0,
        margin=40.0,
        settings=settings,
        observed="Not remember",
    ) == (True, "")


def test_suspicious_duration_prevents_exact_match_auto_acceptance() -> None:
    line = {"line": "Pathetic!"}

    assert _candidate_reliability(
        line=line,
        match_score=100.0,
        margin=40.0,
        settings={"reliable_min_duration_plausibility": 25.0},
        observed="Pathetic",
        duration_plausibility=20.2,
    ) == (False, "POSSIBLE_REPEATED_TAKES")
    assert _candidate_reliability(
        line=line,
        match_score=100.0,
        margin=40.0,
        settings={"reliable_min_duration_plausibility": 25.0},
        observed="Pathetic",
        duration_plausibility=38.4,
    ) == (True, "")


@pytest.mark.parametrize(
    ("expected", "repeated"),
    [
        (
            "You'll break before I will!",
            "Break Before I Will You'll Break Before I Will Will "
            "You'll Break Before I Will",
        ),
        (
            "Right! Bring it on!",
            "Bring It On Right Bring It On Right Bring It On",
        ),
    ],
)
def test_transcribed_repeated_takes_prevent_auto_acceptance(
    expected: str,
    repeated: str,
) -> None:
    reliable, reason = _candidate_reliability(
        line={"line": expected},
        match_score=text_similarity(expected, repeated),
        margin=40.0,
        settings={},
        observed=repeated,
        duration_plausibility=50.0,
    )

    assert reliable is False
    assert reason == "EXCESS_TRANSCRIPT_WORDS"


def test_transcript_fidelity_tolerates_minor_asr_spelling_errors() -> None:
    fidelity = transcript_fidelity(
        "Good afternoon.",
        "Good afternon",
    )

    assert fidelity["ordered_similarity"] >= 90.0
    assert fidelity["token_coverage"] == 1.0
    assert fidelity["token_precision"] == 1.0
    assert fidelity["extra_word_count"] == 0


@pytest.mark.parametrize(
    ("expected", "partial"),
    [
        (
            "You've been avoiding the sun, haven't you? "
            "Can't say I blame you with this heat.",
            "Can't say I blame you with this heat",
        ),
        (
            "Got too much to do. Feels like I barely started...",
            "Feels like I barely started",
        ),
        (
            "Ever hear the story about the frozen Hist? "
            "Wonder if any of it's true...",
            "Ever hear the story about the frozen Hist",
        ),
        (
            "Been some strange storms lately. "
            "You wouldn't know anything about that, would you?",
            "You wouldn't know anything about that would you",
        ),
        ("Maybe over here... No, nothing.", "Maybe over here"),
    ],
)
def test_missing_sentence_prevents_auto_acceptance(
    expected: str,
    partial: str,
) -> None:
    sentence = sentence_fidelity(expected, partial)

    assert sentence["clause_count"] == 2
    assert sentence["missing_clause_count"] >= 1
    assert _candidate_reliability(
        line={"line": expected},
        match_score=max(90.0, text_similarity(expected, partial)),
        margin=40.0,
        settings={
            "reliable_min_score": 70.0,
            "reliable_min_margin": 5.0,
            "reliable_min_clause_score": 55.0,
        },
        observed=partial,
    ) == (False, "MISSING_SENTENCE")


def test_reordered_sentences_prevent_auto_acceptance() -> None:
    expected = (
        "Hmm... I don't suppose we'll be able to count myself among that "
        "latter group. Tell me, how goes the war?"
    )
    observed = (
        "Tell me how goes the war Hmm I don't suppose we'll be able to "
        "count myself among that latter group"
    )

    sentence = sentence_fidelity(expected, observed)

    assert sentence["missing_clause_count"] == 0
    assert sentence["clauses_in_order"] is False
    assert _candidate_reliability(
        line={"line": expected},
        match_score=94.0,
        margin=40.0,
        settings={},
        observed=observed,
        duration_plausibility=70.0,
    ) == (False, "SENTENCE_ORDER_MISMATCH")


def test_adjacent_sentence_fragments_create_complete_take_candidates() -> None:
    line_text = (
        "You've been avoiding the sun, haven't you? "
        "Can't say I blame you with this heat."
    )
    lines = [{"line_id": "line", "line": line_text}]
    transcripts = [
        "Been avoiding the sun haven't you",
        "Can't say I blame you with this heat",
        "You've been avoiding the sun haven't you",
        "Can't say I blame you with this heat",
    ]
    segments = [
        {
            "start_seconds": index * 2.0,
            "end_seconds": index * 2.0 + 1.5,
            "transcript": transcript,
            "asr_probability": 0.95,
        }
        for index, transcript in enumerate(transcripts)
    ]
    actions = [
        {
            "type": "assigned",
            "start_index": index,
            "count": 1,
            "line_index": 0,
            "match_score": text_similarity(line_text, transcript),
            "transcript": transcript,
            "duration_plausibility": 80.0,
            "order_hint": 0.0,
            "top_matches": [],
        }
        for index, transcript in enumerate(transcripts)
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_segments": 2,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
    )

    assert [(action["start_index"], action["count"]) for action in joined] == [
        (0, 2),
        (2, 2),
    ]
    assert all(action["fragment_join"] for action in joined)
    assert all(action["fragment_source_count"] == 2 for action in joined)
    assert all(
        sentence_fidelity(line_text, action["transcript"])[
            "missing_clause_count"
        ]
        == 0
        for action in joined
    )


def test_fragment_join_uses_shortest_text_bounded_span() -> None:
    line_text = (
        "Slow, but picking up. Lost a lot of good customers when Darius "
        "let half his workers go."
    )
    lines = [{"line_id": "line", "line": line_text}]
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 1.5,
            "transcript": "Slow but picking up",
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 1.8,
            "end_seconds": 5.0,
            "transcript": (
                "Lost a lot of good customers when Darius let half his "
                "workers go"
            ),
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 5.3,
            "end_seconds": 6.0,
            "transcript": "",
            "asr_probability": None,
        },
    ]
    actions = [
        {
            "type": "assigned",
            "start_index": 0,
            "count": 1,
            "line_index": 0,
            "match_score": text_similarity(
                line_text,
                segments[0]["transcript"],
            ),
            "transcript": segments[0]["transcript"],
            "duration_plausibility": 50.0,
            "order_hint": 0.0,
            "top_matches": [],
        },
        {
            "type": "assigned",
            "start_index": 1,
            "count": 2,
            "line_index": 0,
            "match_score": text_similarity(
                line_text,
                segments[1]["transcript"],
            ),
            "transcript": segments[1]["transcript"],
            "duration_plausibility": 80.0,
            "order_hint": 0.0,
            "top_matches": [],
        },
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_segments": 3,
            "fragment_join_max_segments": 3,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
    )

    assert [(action["start_index"], action["count"]) for action in joined] == [
        (0, 2)
    ]


def test_fragment_join_recovers_unselected_preceding_sentence_from_word_span() -> None:
    line_text = (
        "Some old sailor superstition, maybe. Taunting death. "
        "Seems more like tempting fate, but what would I know? "
        "I've never been one for the sea."
    )
    lines = [{"line_id": "line", "line": line_text}]
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "transcript": "Old sailor superstition maybe",
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 1.2,
            "end_seconds": 2.0,
            "transcript": "Death Seems",
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 2.2,
            "end_seconds": 5.0,
            "transcript": (
                "More like tempting fate but what would I know "
                "I've never been one for the sea"
            ),
            "asr_probability": 0.95,
        },
    ]
    normalized_words = line_text.translate(
        str.maketrans({".": "", ",": "", "?": ""})
    ).split()
    word_duration = 4.8 / len(normalized_words)
    transcription = {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "words": [
                    {
                        "start": index * word_duration,
                        "end": (index + 1) * word_duration,
                        "word": f" {word}",
                        "probability": 0.99,
                    }
                    for index, word in enumerate(normalized_words)
                ],
            }
        ]
    }
    partial = f"{segments[1]['transcript']} {segments[2]['transcript']}"
    actions = [
        {
            "type": "assigned",
            "start_index": 1,
            "count": 2,
            "line_index": 0,
            "match_score": text_similarity(line_text, partial),
            "transcript": partial,
            "duration_plausibility": 80.0,
            "order_hint": 0.0,
            "top_matches": [],
        }
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_segments": 3,
            "fragment_join_max_segments": 3,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
        transcription=transcription,
    )

    assert len(joined) == 1
    assert joined[0]["start_index"] == 0
    assert joined[0]["count"] == 3
    assert "Taunting" in joined[0]["transcript"]
    assert (
        sentence_fidelity(line_text, joined[0]["transcript"])[
            "missing_clause_count"
        ]
        == 0
    )


def test_unordered_alignment_does_not_merge_empty_boundary_segment() -> None:
    lines = [
        {
            "line_id": "line",
            "line": (
                "Lost a lot of good customers when Darius let half his "
                "workers go, and the new ships do not make up for it."
            ),
        }
    ]
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "transcript": lines[0]["line"],
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 4.3,
            "end_seconds": 6.0,
            "transcript": "",
            "asr_probability": None,
        },
    ]

    actions = order_independent_align(
        segments,
        lines,
        {
            "max_merge_segments": 2,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
            "path_min_score": 35.0,
            "candidate_min_score": 45.0,
            "merge_require_text_boundaries": True,
        },
    )

    assert len(actions) == 1
    assert actions[0]["start_index"] == 0
    assert actions[0]["count"] == 1


def test_untranscribed_audio_in_merge_prevents_auto_acceptance() -> None:
    unsafe = _has_unsafe_untranscribed_merge(
        action={"start_index": 0, "count": 2},
        base_segments=[
            {
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "transcript": "Hello there",
                "metrics": {"duration_seconds": 1.0, "rms_dbfs": -20.0},
            },
            {
                "start_seconds": 1.2,
                "end_seconds": 2.2,
                "transcript": "",
                "metrics": {"duration_seconds": 1.0, "rms_dbfs": -24.0},
            },
        ],
        settings={},
    )

    assert unsafe is True
    assert _candidate_reliability(
        line={"line": "Hello there."},
        match_score=100.0,
        margin=50.0,
        settings={},
        observed="Hello there",
        unsafe_untranscribed_merge=unsafe,
    ) == (False, "MERGED_UNTRANSCRIBED_AUDIO")


def test_word_timestamp_gaps_split_a_region_into_take_candidates() -> None:
    transcription = {
        "segments": [
            {
                "start": 0.0,
                "end": 4.0,
                "text": "Take this Take this",
                "words": [
                    {"start": 0.2, "end": 0.5, "word": " Take"},
                    {"start": 0.5, "end": 0.8, "word": " this"},
                    {"start": 1.5, "end": 1.8, "word": " Take"},
                    {"start": 1.8, "end": 2.1, "word": " this"},
                ],
            }
        ]
    }
    regions = [
        {
            "speech_start": 0.0,
            "speech_end": 4.0,
            "start": 0.0,
            "end": 4.0,
        }
    ]

    refined = split_regions_on_word_gaps(
        regions,
        transcription,
        duration_seconds=4.0,
        settings={
            "word_split_enabled": True,
            "word_split_gap_seconds": 0.55,
            "word_split_min_region_seconds": 1.0,
            "word_split_max_boundaries": 2,
            "minimum_segment_seconds": 0.15,
            "pre_padding_seconds": 0.1,
            "post_padding_seconds": 0.1,
        },
    )

    assert len(refined) == 2
    assert all(region["split_source"] == "word_gap" for region in refined)


def test_duplicate_weak_order_assigns_separate_take_groups() -> None:
    lines = [
        {"line_id": "a", "line": "Found you!"},
        {"line_id": "b", "line": "Found you!"},
    ]
    segments = [
        {"start_seconds": 0.0, "end_seconds": 1.0},
        {"start_seconds": 20.0, "end_seconds": 21.0},
    ]
    actions = [
        {
            "start_index": 0,
            "count": 1,
            "line_index": 0,
            "match_score": 100.0,
            "top_matches": [
                {"line_index": 0, "match_score": 100.0},
                {"line_index": 1, "match_score": 100.0},
            ],
        },
        {
            "start_index": 1,
            "count": 1,
            "line_index": 0,
            "match_score": 100.0,
            "top_matches": [
                {"line_index": 0, "match_score": 100.0},
                {"line_index": 1, "match_score": 100.0},
            ],
        },
    ]

    _apply_duplicate_line_policy(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "duplicate_line_policy": "weak_order",
            "take_group_gap_seconds": 5.0,
        },
    )

    assert [action["line_index"] for action in actions] == [0, 1]
    assert all(action["duplicate_resolved"] for action in actions)


def test_nonverbal_review_policy_does_not_supply_random_candidates() -> None:
    actions = _vocalization_review_actions(
        base_segments=[
            {
                "start_seconds": 0.0,
                "end_seconds": 0.8,
                "transcript": "completely unrelated",
                "word_count": 2,
            }
        ],
        lines=[{"line_id": "a", "line": "(cough)"}],
        session_id="session",
        settings={"nonverbal_policy": "review"},
        existing_actions=[],
    )

    assert actions == []


def test_vocalization_weak_order_policy_supplies_review_candidates() -> None:
    actions = _vocalization_review_actions(
        base_segments=[
            {
                "start_seconds": 0.0,
                "end_seconds": 0.8,
                "transcript": "eaargh",
                "word_count": 1,
            }
        ],
        lines=[{"line_id": "a", "line": "Aaaarraaagh..."}],
        session_id="session",
        settings={
            "nonverbal_policy": "weak_order",
            "vocalization_alignment": {
                "enabled": True,
                "candidates_per_line": 1,
            }
        },
        existing_actions=[],
    )

    assert len(actions) == 1
    assert actions[0]["force_candidate"] is True
    assert actions[0]["forced_review_reason"] == "VOCALIZATION_REVIEW"


@pytest.mark.parametrize(
    ("policy", "expected_status"),
    [
        ("review", "NONVERBAL_REVIEW"),
        ("skip", "SKIP"),
    ],
)
def test_review_workbook_does_not_preselect_nonverbal_lines(
    tmp_path: Path,
    policy: str,
    expected_status: str,
) -> None:
    line = {
        "line_id": "Sheet::R3",
        "sheet": "Sheet",
        "sheet_index": 0,
        "excel_row": 3,
        "quest": "",
        "context": "",
        "line": "(cough)",
        "acting_note": "",
        "emotion": "",
        "target_filename": "cough_target",
    }
    review_path = tmp_path / f"{policy}.xlsx"
    _write_review_workbook(
        review_path=review_path,
        project_dir=tmp_path,
        source_lines=[line],
        candidates_by_line={},
        unmatched_rows=[],
        nonverbal_policy=policy,
    )

    workbook = load_workbook(review_path, read_only=False, data_only=False)
    lines_sheet = workbook["Lines"]
    headers = {
        cell.value: cell.column for cell in lines_sheet[1] if cell.value
    }
    assert lines_sheet.cell(2, headers["Candidate Count"]).value == 0
    assert lines_sheet.cell(2, headers["Suggested Best Segment"]).value is None
    assert lines_sheet.cell(2, headers["Selected Segment"]).value is None
    assert lines_sheet.cell(2, headers["Status"]).value == expected_status
    assert lines_sheet.cell(2, headers["Candidate Summary"]).value
    summary_statuses = {
        lines_sheet.cell(row, 1).value
        for row in range(1, lines_sheet.max_row + 1)
    }
    assert "NONVERBAL_REVIEW" in summary_statuses
    assert len(lines_sheet._charts) == 1
    workbook.close()


@pytest.mark.parametrize("aligner", [sequence_align, order_independent_align])
def test_text_aligners_exclude_nonverbal_lines(aligner) -> None:
    actions = aligner(
        [
            {
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "transcript": "cough",
                "asr_probability": 0.9,
            },
            {
                "start_seconds": 2.0,
                "end_seconds": 3.0,
                "transcript": "hello there",
                "asr_probability": 0.9,
            },
        ],
        [
            {"line_id": "nonverbal", "line": "(cough)"},
            {"line_id": "spoken", "line": "Hello there."},
        ],
        {
            "max_merge_segments": 1,
            "path_min_score": 35.0,
            "candidate_min_score": 35.0,
            "lookahead_lines": 8,
        },
    )

    assert actions
    assert {action["line_index"] for action in actions} == {1}


def test_local_asr_rescue_accepts_a_better_cached_segment_transcript(
    tmp_path: Path,
) -> None:
    segment_file = tmp_path / "segment.wav"
    _write_tone(segment_file)
    fake_word = SimpleNamespace(probability=0.95)
    fake_segment = SimpleNamespace(
        text="Good afternoon",
        words=[fake_word, fake_word],
    )

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            return iter([fake_segment]), SimpleNamespace()

    segment = {
        "segment_id": "session__s00001",
        "file": "segment.wav",
        "source_sha256": "abc",
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "transcript": "Afternon",
        "word_count": 1,
        "asr_probability": 0.3,
    }
    project = {
        "language": "en",
        "transcription": {
            "model": "large-v3",
            "device": "cpu",
            "compute_type": "int8",
        },
    }

    accepted = _apply_local_asr_rescue(
        project_dir=tmp_path,
        project=project,
        session={"id": "session"},
        session_lines=[
            {
                "line_id": "a",
                "line": "Good afternoon.",
            }
        ],
        base_segments=[segment],
        settings={
            "local_asr_rescue": {
                "enabled": True,
                "trigger_score": 101.0,
                "minimum_probability": 0.5,
                "minimum_score_gain": 0.0,
                "minimum_score": 72.0,
            }
        },
        runtime={
            "model": FakeModel(),
            "device": "cpu",
            "compute_type": "int8",
        },
    )

    assert accepted == 1
    assert segment["transcript"] == "Good afternoon"
    assert segment["transcript_source"] == "local_asr_rescue"


def test_segment_asr_transcribes_empty_base_clip_without_vad_and_caches(
    tmp_path: Path,
) -> None:
    segment_file = tmp_path / "segment.wav"
    _write_tone(segment_file)
    calls = []
    fake_word = SimpleNamespace(
        start=0.1,
        end=0.4,
        word=" Yes",
        probability=0.96,
    )
    fake_segment = SimpleNamespace(
        start=0.1,
        end=0.4,
        text="Yes",
        avg_logprob=-0.05,
        no_speech_prob=0.01,
        words=[fake_word],
    )

    class FakeModel:
        def transcribe(self, *_args, **kwargs):
            calls.append(kwargs)
            return iter([fake_segment]), SimpleNamespace(language="en")

    line = {
        "line_id": "Actor::R2",
        "sheet": "Actor",
        "sheet_index": 0,
        "excel_row": 2,
        "line": "Yes?",
    }
    write_json(tmp_path / "source_lines.json", {"lines": [line]})
    write_json(
        tmp_path / "segments_manifest.json",
        {
            "sessions": [
                {
                    "session_id": "session",
                    "segments": [
                        {
                            "segment_id": "session__s00001",
                            "kind": "base",
                            "file": "segment.wav",
                            "source_sha256": "source-hash",
                            "start_sample": 0,
                            "end_sample": 48000,
                            "start_seconds": 0.0,
                            "end_seconds": 1.0,
                            "transcript": "",
                            "words": [],
                            "word_count": 0,
                            "asr_probability": None,
                            "metrics": {"duration_seconds": 1.0},
                        }
                    ],
                    "derived_segments": [],
                }
            ]
        },
    )
    project = {
        "source_lines": "source_lines.json",
        "language": "en",
        "transcription": {
            "model": "large-v3",
            "device": "cpu",
            "compute_type": "int8",
        },
        "segment_transcription": {
            "enabled": True,
            "prompt_fallback_enabled": False,
        },
        "sessions": [
            {
                "id": "session",
                "enabled": True,
                "sheets": ["Actor"],
                "excel_rows": [],
                "line_ids": [],
            }
        ],
    }
    runtime = {"model": FakeModel()}
    transcribe_segments_project(
        project_dir=tmp_path,
        project=project,
        runtime=runtime,
    )

    manifest = read_json(tmp_path / "segments_manifest.json")
    segment = manifest["sessions"][0]["segments"][0]
    assert segment["session_transcript"] == ""
    assert segment["transcript"] == "Yes"
    assert segment["transcript_source"] == "segment_asr"
    assert segment["segment_asr"]["primary"]["asr_probability"] == pytest.approx(
        0.96
    )
    assert calls[0]["vad_filter"] is False
    assert calls[0]["condition_on_previous_text"] is False

    transcribe_segments_project(
        project_dir=tmp_path,
        project=project,
        runtime={},
    )
    assert len(calls) == 1


def test_prompted_segment_asr_is_evidence_but_cannot_verify_auto_ok(
    tmp_path: Path,
) -> None:
    segment_file = tmp_path / "segment.wav"
    _write_tone(segment_file)
    outputs = [
        SimpleNamespace(
            start=0.0,
            end=1.0,
            text="",
            avg_logprob=-1.0,
            no_speech_prob=0.8,
            words=[],
        ),
        SimpleNamespace(
            start=0.0,
            end=1.0,
            text="Yes",
            avg_logprob=-0.1,
            no_speech_prob=0.01,
            words=[
                SimpleNamespace(
                    start=0.1,
                    end=0.4,
                    word=" Yes",
                    probability=0.95,
                )
            ],
        ),
    ]

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            return iter([outputs.pop(0)]), SimpleNamespace(language="en")

    line = {
        "line_id": "Actor::R2",
        "sheet": "Actor",
        "sheet_index": 0,
        "excel_row": 2,
        "line": "Yes?",
    }
    write_json(tmp_path / "source_lines.json", {"lines": [line]})
    write_json(
        tmp_path / "segments_manifest.json",
        {
            "sessions": [
                {
                    "session_id": "session",
                    "segments": [
                        {
                            "segment_id": "session__s00001",
                            "kind": "base",
                            "file": "segment.wav",
                            "source_sha256": "source-hash",
                            "start_sample": 0,
                            "end_sample": 48000,
                            "start_seconds": 0.0,
                            "end_seconds": 1.0,
                            "transcript": "",
                            "words": [],
                            "word_count": 0,
                            "asr_probability": None,
                            "metrics": {"duration_seconds": 1.0},
                        }
                    ],
                    "derived_segments": [],
                }
            ]
        },
    )
    project = {
        "source_lines": "source_lines.json",
        "language": "en",
        "transcription": {
            "model": "large-v3",
            "device": "cpu",
            "compute_type": "int8",
        },
        "segment_transcription": {
            "enabled": True,
            "prompt_fallback_enabled": True,
        },
        "sessions": [
            {
                "id": "session",
                "enabled": True,
                "sheets": ["Actor"],
                "excel_rows": [],
                "line_ids": [],
            }
        ],
    }
    runtime = {"model": FakeModel()}
    transcribe_segments_project(
        project_dir=tmp_path,
        project=project,
        runtime=runtime,
    )
    segment = read_json(tmp_path / "segments_manifest.json")["sessions"][0][
        "segments"
    ][0]

    assert segment["transcript"] == "Yes"
    assert segment["transcript_source"] == "segment_asr_prompted"
    assert segment["segment_asr"]["primary"]["transcript"] == ""
    assert (
        segment["segment_asr"]["prompted_fallback"]["transcript"]
        == "Yes"
    )
    exact = transcribe_candidate_span(
        project_dir=tmp_path,
        project=project,
        segment=segment,
        runtime=runtime,
    )
    assert exact["transcript"] == ""


def test_merged_candidate_asr_decodes_the_exact_span_once(
    tmp_path: Path,
) -> None:
    segment_file = tmp_path / "merged.wav"
    _write_tone(segment_file, duration_seconds=2.0)
    calls = []
    fake_segment = SimpleNamespace(
        start=0.0,
        end=1.8,
        text="Hello there hello there",
        avg_logprob=-0.1,
        no_speech_prob=0.01,
        words=[],
    )

    class FakeModel:
        def transcribe(self, *_args, **kwargs):
            calls.append(kwargs)
            return iter([fake_segment]), SimpleNamespace(language="en")

    project = {
        "language": "en",
        "transcription": {
            "model": "large-v3",
            "device": "cpu",
            "compute_type": "int8",
        },
        "segment_transcription": {"enabled": True},
        "segmentation": {"fade_ms": 5.0},
        "export": {
            "sample_rate": 48000,
            "channels": 1,
            "bits_per_sample": 16,
        },
    }
    segment = {
        "segment_id": "session__m00001_00002",
        "kind": "merged",
        "file": "merged.wav",
        "source_sha256": "source-hash",
        "start_sample": 0,
        "end_sample": 96000,
    }
    result = transcribe_candidate_span(
        project_dir=tmp_path,
        project=project,
        segment=segment,
        runtime={"model": FakeModel()},
    )
    cached = transcribe_candidate_span(
        project_dir=tmp_path,
        project=project,
        segment=segment,
        runtime={},
    )

    assert result["transcript"] == "Hello there hello there"
    assert cached["cache_key"] == result["cache_key"]
    assert len(calls) == 1
    assert calls[0]["vad_filter"] is False
    assert calls[0]["condition_on_previous_text"] is False
    assert "initial_prompt" not in calls[0]


def test_sample_accurate_cut_and_finalize(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    segment_file = tmp_path / "segments" / "session" / "session__s00001.wav"
    _write_tone(source)
    metrics = cut_pcm_wav(
        source,
        segment_file,
        start_sample=4800,
        end_sample=14400,
        fade_ms=5.0,
    )
    assert metrics["frame_count"] == 9600
    assert metrics["sample_rate"] == 48000

    line = {
        "line_id": "Sheet::R3",
        "sheet": "Sheet",
        "sheet_index": 0,
        "excel_row": 3,
        "quest": "Quest",
        "context": "Context",
        "line": "Hello there.",
        "acting_note": "",
        "emotion": "",
        "target_filename": "TargetFile_00000001_1",
    }
    candidate = {
        "line_id": line["line_id"],
        "session_id": "session",
        "segment_id": "session__s00001",
        "segment_file": segment_file.relative_to(tmp_path).as_posix(),
        "transcript": "Hello there.",
        "match_score": 100.0,
        "ordered_similarity": 100.0,
        "token_coverage": 1.0,
        "token_precision": 1.0,
        "extra_word_count": 0,
        "clause_count": 1,
        "minimum_clause_score": 100.0,
        "missing_clause_count": 0,
        "fragment_join": False,
        "fragment_source_count": 0,
        "confidence_margin": 50.0,
        "technical_score": 100.0,
        "selection_score": 115.0,
        "reliable": True,
        "reliability_reason": "",
        "source_audio": "source.wav",
        "start_seconds": 0.1,
        "end_seconds": 0.3,
        "duration_seconds": 0.2,
        "asr_probability": 0.99,
        "base_indices": [0],
        "rank": 1,
    }
    review_path = tmp_path / "A_line_review.xlsx"
    _write_review_workbook(
        review_path=review_path,
        project_dir=tmp_path,
        source_lines=[line],
        candidates_by_line={line["line_id"]: [candidate]},
        unmatched_rows=[
            {
                "segment_id": candidate["segment_id"],
                "segment_file": candidate["segment_file"],
                "source_wav": candidate["source_audio"],
                "start_seconds": candidate["start_seconds"],
                "end_seconds": candidate["end_seconds"],
                "duration_seconds": candidate["duration_seconds"],
                "transcript": "Unmatched introduction",
                "asr_confidence": candidate["asr_probability"],
                "reason": "NO_RELIABLE_MATCH",
                "suggested_line_1": line["line_id"],
                "suggested_line_1_text": line["line"],
                "suggested_line_1_score": 40.0,
                "suggested_line_2": "",
                "suggested_line_2_text": "",
                "suggested_line_2_score": 0.0,
                "technical_flags": "",
            }
        ],
    )
    workbook = load_workbook(review_path, read_only=False, data_only=False)
    lines_sheet = workbook["Lines"]
    headers = {
        cell.value: cell.column for cell in lines_sheet[1] if cell.value
    }
    assert lines_sheet.cell(2, headers["Candidate 1"]).value == "session__s00001"
    assert lines_sheet.cell(2, headers["Suggested Best Segment"]).value == (
        "session__s00001"
    )
    assert lines_sheet.cell(2, headers["Selected Segment"]).value == (
        "session__s00001"
    )
    assert lines_sheet.cell(2, headers["Candidate 1"]).hyperlink is not None
    assert (
        lines_sheet.cell(2, headers["Suggested Best Segment"]).hyperlink
        is not None
    )
    assert lines_sheet.cell(2, headers["Selected Segment"]).hyperlink is not None
    assert len(lines_sheet._charts) == 1
    assert lines_sheet["B7"].value.startswith("=COUNTIF(")
    assert "Unmatched Segments" in workbook.sheetnames
    candidates_sheet = workbook["Candidates"]
    candidate_headers = {
        cell.value: cell.column for cell in candidates_sheet[1] if cell.value
    }
    assert (
        candidates_sheet.cell(
            2,
            candidate_headers["Minimum Clause Score"],
        ).value
        == 100.0
    )
    assert (
        candidates_sheet.cell(2, candidate_headers["Fragment Join"]).value
        is False
    )
    unmatched_sheet = workbook["Unmatched Segments"]
    assert unmatched_sheet["A2"].value == "session__s00001"
    assert unmatched_sheet["A2"].hyperlink is not None
    assert unmatched_sheet["B2"].hyperlink is not None
    workbook.close()

    manifest = {
        "schema_version": 1,
        "sessions": [
            {
                "session_id": "session",
                "segments": [
                    {
                        "segment_id": "session__s00001",
                        "file": segment_file.relative_to(tmp_path).as_posix(),
                        "source_audio": "source.wav",
                        "start_seconds": 0.1,
                        "end_seconds": 0.3,
                        "transcript": "Hello there.",
                    }
                ],
                "derived_segments": [],
            }
        ],
    }
    write_json(tmp_path / "segments_manifest.json", manifest)
    project = {
        "export": {
            "extension": ".wav",
            "sample_rate": 48000,
            "channels": 1,
            "bits_per_sample": 16,
        }
    }
    output_dir = tmp_path / "final"
    result = finalize_review(
        project_dir=tmp_path,
        project=project,
        review_path=review_path,
        output_dir=output_dir,
    )
    assert result["error_count"] == 0
    assert (output_dir / "TargetFile_00000001_1.wav").is_file()


def test_finalize_can_optionally_reuse_one_segment(tmp_path: Path) -> None:
    segment_file = tmp_path / "segment.wav"
    _write_tone(segment_file)
    lines = [
        {
            "line_id": f"Sheet::R{row}",
            "sheet": "Sheet",
            "sheet_index": 0,
            "excel_row": row,
            "quest": "",
            "context": "",
            "line": "Found you!",
            "acting_note": "",
            "emotion": "",
            "target_filename": f"target_{row}",
        }
        for row in (3, 4)
    ]
    candidates = {}
    for line in lines:
        candidates[line["line_id"]] = [
            {
                "line_id": line["line_id"],
                "session_id": "session",
                "segment_id": "session__s00001",
                "segment_file": "segment.wav",
                "transcript": "Found you",
                "match_score": 100.0,
                "confidence_margin": 0.0,
                "technical_score": 100.0,
                "selection_score": 115.0,
                "reliable": True,
                "reliability_reason": "",
                "source_audio": "source.wav",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "duration_seconds": 1.0,
                "asr_probability": 0.99,
                "base_indices": [0],
                "rank": 1,
            }
        ]
    review_path = tmp_path / "review.xlsx"
    _write_review_workbook(
        review_path=review_path,
        project_dir=tmp_path,
        source_lines=lines,
        candidates_by_line=candidates,
        unmatched_rows=[],
    )
    write_json(
        tmp_path / "segments_manifest.json",
        {
            "sessions": [
                {
                    "session_id": "session",
                    "segments": [
                        {
                            "segment_id": "session__s00001",
                            "file": "segment.wav",
                            "source_audio": "source.wav",
                            "start_seconds": 0.0,
                            "end_seconds": 1.0,
                            "transcript": "Found you",
                        }
                    ],
                    "derived_segments": [],
                }
            ]
        },
    )
    project = {
        "export": {
            "extension": ".wav",
            "sample_rate": 48000,
            "channels": 1,
            "bits_per_sample": 16,
        }
    }

    with pytest.raises(ValueError, match="Finalization stopped"):
        finalize_review(
            project_dir=tmp_path,
            project=project,
            review_path=review_path,
            output_dir=tmp_path / "blocked",
            dry_run=True,
        )
    result = finalize_review(
        project_dir=tmp_path,
        project=project,
        review_path=review_path,
        output_dir=tmp_path / "allowed",
        allow_segment_reuse=True,
        dry_run=True,
    )

    assert result["error_count"] == 0
    assert result["export_count"] == 2
    assert result["allow_segment_reuse"] is True


def test_segmentation_and_alignment_integration(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_pattern(source)
    source_hash = sha256_file(source)
    source_lines = {
        "schema_version": 1,
        "line_count": 2,
        "sheet_count": 1,
        "sheets": [
            {
                "name": "Actor",
                "index": 0,
                "voice_header": "Actor",
                "line_ids": ["Actor::R3", "Actor::R4"],
                "line_count": 2,
            }
        ],
        "lines": [
            {
                "line_id": "Actor::R3",
                "sheet": "Actor",
                "sheet_index": 0,
                "excel_row": 3,
                "quest": "",
                "context": "",
                "line": "Hello there",
                "acting_note": "",
                "emotion": "",
                "target_filename": "hello_target",
            },
            {
                "line_id": "Actor::R4",
                "sheet": "Actor",
                "sheet_index": 0,
                "excel_row": 4,
                "quest": "",
                "context": "",
                "line": "Goodbye friend",
                "acting_note": "",
                "emotion": "",
                "target_filename": "goodbye_target",
            },
        ],
    }
    write_json(tmp_path / "source_lines.json", source_lines)
    write_json(
        tmp_path / "audio_inventory.json",
        {
            "files": [
                {
                    "path": str(source.resolve()),
                    "sha256": source_hash,
                    "duration_seconds": 5.0,
                    "sample_rate": 48000,
                    "channels": 1,
                    "bits_per_sample": 16,
                }
            ]
        },
    )
    project = {
        "source_lines": "source_lines.json",
        "audio_inventory": "audio_inventory.json",
        "language": "en",
        "segmentation": {
            "silence_noise_db": -45.0,
            "silence_detection_min_seconds": 0.2,
            "split_gap_seconds": 0.4,
            "minimum_segment_seconds": 0.1,
            "pre_padding_seconds": 0.05,
            "post_padding_seconds": 0.05,
            "fade_ms": 5.0,
        },
        "alignment": {
            "max_merge_segments": 2,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
            "lookahead_lines": 2,
            "path_min_score": 35.0,
            "candidate_min_score": 45.0,
            "reliable_min_score": 72.0,
            "reliable_min_margin": 8.0,
            "short_line_min_score": 88.0,
            "short_line_min_margin": 15.0,
            "noise_penalty": 2.2,
            "skip_line_penalty": 1.8,
            "repeat_take_penalty": 0.8,
        },
        "export": {
            "extension": ".wav",
            "sample_rate": 48000,
            "channels": 1,
            "bits_per_sample": 16,
        },
        "sessions": [
            {
                "id": "session",
                "enabled": True,
                "audio": "source.wav",
                "sheets": ["Actor"],
                "excel_rows": [],
                "line_ids": [],
            }
        ],
    }
    write_json(
        tmp_path / "transcripts" / "session.json",
        {
            "schema_version": 1,
            "session_id": "session",
            "audio": "source.wav",
            "audio_sha256": source_hash,
            "cache_key": "synthetic",
            "segments": [
                {
                    "id": 0,
                    "start": 0.25,
                    "end": 0.9,
                    "text": "Hello there",
                    "words": [
                        {
                            "start": 0.25,
                            "end": 0.5,
                            "word": " Hello",
                            "probability": 0.99,
                        },
                        {
                            "start": 0.55,
                            "end": 0.9,
                            "word": " there",
                            "probability": 0.99,
                        },
                    ],
                },
                {
                    "id": 1,
                    "start": 1.4,
                    "end": 2.05,
                    "text": "Hello there",
                    "words": [
                        {
                            "start": 1.4,
                            "end": 1.65,
                            "word": " Hello",
                            "probability": 0.98,
                        },
                        {
                            "start": 1.7,
                            "end": 2.05,
                            "word": " there",
                            "probability": 0.98,
                        },
                    ],
                },
                {
                    "id": 2,
                    "start": 3.0,
                    "end": 3.75,
                    "text": "Goodbye friend",
                    "words": [
                        {
                            "start": 3.0,
                            "end": 3.35,
                            "word": " Goodbye",
                            "probability": 0.99,
                        },
                        {
                            "start": 3.4,
                            "end": 3.75,
                            "word": " friend",
                            "probability": 0.99,
                        },
                    ],
                },
            ],
        },
    )
    manifest_path = segment_project(project_dir=tmp_path, project=project)
    assert manifest_path.is_file()
    stale_unmatched = tmp_path / "B_unmatched_segments.tsv"
    stale_unmatched.write_text("stale", encoding="utf-8")
    outputs = align_project(
        project_dir=tmp_path,
        project=project,
        session_filter={"session"},
    )
    assert outputs["review"].is_file()
    assert "unmatched" not in outputs
    assert not stale_unmatched.exists()
    workbook = load_workbook(outputs["review"], read_only=False, data_only=False)
    lines_sheet = workbook["Lines"]
    headers = {
        cell.value: cell.column for cell in lines_sheet[1] if cell.value
    }
    assert lines_sheet.cell(2, headers["Selected Segment"]).value
    assert lines_sheet.cell(3, headers["Selected Segment"]).value
    assert lines_sheet.cell(2, headers["Candidate 1"]).hyperlink is not None
    assert len(lines_sheet._charts) == 1
    assert "Unmatched Segments" in workbook.sheetnames
    workbook.close()
