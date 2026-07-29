from __future__ import annotations

import math
import subprocess
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dialogue_pipeline.alignment import (
    _apply_duplicate_line_policy,
    _candidate_reliability,
    _exact_line_scores,
    _expand_alignment_actions,
    _has_unsafe_untranscribed_merge,
    _intra_segment_trim_actions,
    _multisentence_fragment_join_actions,
    _reroute_actions_to_exact_asr_primary_matches,
    TranscriptEvaluator,
    align_project,
    order_independent_align,
    sentence_fidelity,
    text_similarity,
    transcript_fidelity,
)
from dialogue_pipeline.alignment_settings import (
    ALIGNMENT_DEFAULTS,
    AlignmentSettings,
    default_alignment_config,
)
from dialogue_pipeline.audio import cut_pcm_wav, prepare_pcm_segmentation_source
from dialogue_pipeline.cancellation import (
    ProcessingCancelled,
    cancellation_scope,
    check_processing_cancelled,
)
from dialogue_pipeline.finalize import finalize_review
from dialogue_pipeline.project import (
    create_project,
    editable_project_settings,
    infer_sessions,
    migrate_project_config,
)
from dialogue_pipeline.review import (
    build_line_review,
    load_line_review,
    preserve_manual_selections,
    prune_line_candidates,
    save_line_review,
)
from dialogue_pipeline.segmentation import (
    prevent_region_overlaps,
    segment_project,
    split_regions_on_word_gaps,
)
from dialogue_pipeline.transcription import (
    _automatic_batch_size,
    _decode_clips_batched,
    _likely_silence_hallucination,
    transcribe_candidate_span,
    transcribe_candidate_spans,
    transcribe_project,
    transcribe_segments_project,
)
from dialogue_pipeline.ui import (
    DialogueReviewApp,
    _project_settings_from_values,
    _selected_segment_score,
    _uses_unmatched_candidates,
)
from dialogue_pipeline.util import (
    default_model_cache_root,
    read_json,
    resolve_model_cache_root,
    sha256_file,
    stable_hash,
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


def test_new_project_settings_are_validated_and_merged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dialogue_pipeline.project as project_module

    workbook = tmp_path / "lines.xlsx"
    workbook.touch()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_tone(audio_dir / "Actor.wav")
    monkeypatch.setattr(
        project_module,
        "parse_workbook",
        lambda _path: {
            "sheet_count": 1,
            "line_count": 0,
            "sheets": [
                {
                    "name": "Actor",
                    "line_count": 0,
                    "voice_header": "",
                }
            ],
            "lines": [],
        },
    )
    values = editable_project_settings()
    values["language"] = "de"
    values["transcription"].update(
        {
            "model": "medium",
            "device": "cuda",
            "compute_type": "float16",
            "batch_size": "8",
            "batch_size_max": "24",
        }
    )
    values["segment_transcription"]["enabled"] = False
    settings = _project_settings_from_values(
        values,
        editable_project_settings(),
    )

    project = create_project(
        workbook_path=workbook,
        audio_dir=audio_dir,
        project_dir=tmp_path / "work",
        project_settings=settings,
    )

    assert project["language"] == "de"
    assert project["transcription"]["model"] == "medium"
    assert project["transcription"]["batch_size"] == 8
    assert project["transcription"]["batch_size_max"] == 24
    assert project["transcription"]["vad_filter"] is True
    assert project["segment_transcription"]["enabled"] is False
    assert project["segment_transcription"]["prompt_fallback_enabled"] is True
    assert set(settings) == {
        "language",
        "transcription",
        "segment_transcription",
        "segmentation",
        "alignment",
        "export",
    }

    invalid = editable_project_settings()
    invalid["transcription"]["batch_size"] = "zero"
    with pytest.raises(ValueError, match="Batch size"):
        _project_settings_from_values(
            invalid,
            editable_project_settings(),
        )


def test_create_button_shows_existing_settings_before_reprocessing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dialogue_pipeline.ui as ui_module

    project_dir = tmp_path / "existing"
    project_dir.mkdir()
    write_json(project_dir / "project.json", {"schema_version": 1})
    app = DialogueReviewApp.__new__(DialogueReviewApp)
    selected = []
    shown = []

    def settings_dialog(settings, *, reprocessing):
        shown.append((settings, reprocessing))
        return settings

    app.ask_project_settings = settings_dialog
    app.run_existing_project = lambda path, settings: selected.append(
        (path, settings)
    )
    app.run_new_project = lambda **_kwargs: pytest.fail(
        "Existing projects must not be initialized again"
    )
    monkeypatch.setattr(
        ui_module.filedialog,
        "askdirectory",
        lambda **_kwargs: str(project_dir),
    )
    monkeypatch.setattr(
        ui_module.filedialog,
        "askopenfilename",
        lambda **_kwargs: pytest.fail(
            "Existing projects must not ask for a workbook"
        ),
    )

    app.choose_new_project()

    assert len(shown) == 1
    assert shown[0][1] is True
    assert shown[0][0] == editable_project_settings({"schema_version": 1})
    assert selected == [(project_dir.resolve(), shown[0][0])]


def test_forced_metadata_refresh_backs_up_and_preserves_existing_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dialogue_pipeline.project as project_module

    project_dir = tmp_path / "work"
    project_dir.mkdir()
    workbook = tmp_path / "lines.xlsx"
    workbook.touch()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "Actor.wav"
    audio.touch()
    write_json(
        project_dir / "project.json",
        {
            "schema_version": 1,
            "language": "fr",
            "alignment": {"max_merge_segments": 6},
            "sessions": [
                {
                    "id": "actor",
                    "audio": "../audio/Actor.wav",
                    "enabled": False,
                    "sheets": ["Manual"],
                    "excel_rows": [],
                    "line_ids": [],
                    "pass": "main",
                    "needs_mapping_review": False,
                }
            ],
        },
    )
    source_data = {
        "sheet_count": 1,
        "line_count": 0,
        "sheets": [
            {
                "name": "Actor",
                "line_count": 0,
                "voice_header": "You are voicing Actor",
            }
        ],
        "lines": [],
    }
    monkeypatch.setattr(project_module, "parse_workbook", lambda _path: source_data)
    monkeypatch.setattr(
        project_module,
        "probe_audio",
        lambda path, include_hash: {
            "path": str(path.resolve()),
            "sha256": "hash",
        },
    )

    refreshed = create_project(
        workbook_path=workbook,
        audio_dir=audio_dir,
        project_dir=project_dir,
        force=True,
    )

    assert refreshed["language"] == "fr"
    assert AlignmentSettings.from_value(refreshed["alignment"])[
        "max_merge_segments"
    ] == 6
    assert refreshed["sessions"][0]["sheets"] == ["Manual"]
    assert refreshed["sessions"][0]["enabled"] is False
    assert read_json(project_dir / "project.before-force.json")["language"] == "fr"


def test_create_button_collects_settings_only_for_new_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dialogue_pipeline.ui as ui_module

    project_dir = tmp_path / "new"
    workbook = tmp_path / "lines.xlsx"
    audio_dir = tmp_path / "audio"
    settings = {"language": "en", "transcription": {"model": "small"}}
    selected_directories = iter((str(project_dir), str(audio_dir)))
    monkeypatch.setattr(
        ui_module.filedialog,
        "askdirectory",
        lambda **_kwargs: next(selected_directories),
    )
    monkeypatch.setattr(
        ui_module.filedialog,
        "askopenfilename",
        lambda **_kwargs: str(workbook),
    )
    app = DialogueReviewApp.__new__(DialogueReviewApp)
    app.ask_project_settings = (
        lambda current, *, reprocessing: (
            settings
            if not reprocessing and current == editable_project_settings()
            else pytest.fail("Unexpected settings dialog state")
        )
    )
    captured = {}
    app.run_new_project = lambda **kwargs: captured.update(kwargs)
    app.run_existing_project = lambda _path, _settings: pytest.fail(
        "A new folder must initialize a project"
    )

    app.choose_new_project()

    assert captured == {
        "workbook_path": workbook,
        "audio_dir": audio_dir,
        "project_dir": project_dir.resolve(),
        "project_settings": settings,
    }


def test_existing_project_reprocess_pipeline_preserves_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dialogue_pipeline.ui as ui_module

    project_dir = tmp_path / "existing"
    project_dir.mkdir()
    write_json(
        project_dir / "project.json",
        {"schema_version": 1, "custom_setting": "keep-me"},
    )
    sentinel = project_dir / "do-not-delete.txt"
    sentinel.write_text("preserved", encoding="utf-8")
    calls = []

    def record(phase: str):
        def run(*, project_dir: Path, project: dict, **_kwargs):
            calls.append((phase, project_dir, project["custom_setting"]))

        return run

    monkeypatch.setattr(ui_module, "transcribe_project", record("transcribe"))
    monkeypatch.setattr(ui_module, "segment_project", record("segment"))
    monkeypatch.setattr(
        ui_module,
        "transcribe_segments_project",
        record("transcribe_segments"),
    )
    monkeypatch.setattr(ui_module, "align_project", record("align"))
    monkeypatch.setattr(
        ui_module,
        "create_project",
        lambda **_kwargs: pytest.fail(
            "Reprocessing must not recreate project.json"
        ),
    )

    app = DialogueReviewApp.__new__(DialogueReviewApp)
    progress = []
    finished = []
    app.show_progress = lambda path, *, title: progress.append((path, title))
    app._pipeline_finished = lambda path: finished.append(path)
    app._start_worker = lambda function, callback: callback(function())

    project_settings = editable_project_settings(
        {
            "schema_version": 1,
            "custom_setting": "keep-me",
        }
    )
    project_settings["language"] = "fr"
    app.run_existing_project(project_dir, project_settings)

    assert [phase for phase, _path, _setting in calls] == [
        "transcribe",
        "segment",
        "transcribe_segments",
        "align",
    ]
    assert all(path == project_dir.resolve() for _phase, path, _setting in calls)
    assert all(setting == "keep-me" for _phase, _path, setting in calls)
    assert progress == [(project_dir, "Reprocessing project")]
    assert finished == [project_dir.resolve()]
    assert sentinel.read_text(encoding="utf-8") == "preserved"
    saved_project = read_json(project_dir / "project.json")
    assert saved_project["custom_setting"] == "keep-me"
    assert saved_project["language"] == "fr"


def test_processing_cancellation_scope_is_thread_local() -> None:
    event = threading.Event()
    with cancellation_scope(event):
        check_processing_cancelled()
        event.set()
        with pytest.raises(ProcessingCancelled):
            check_processing_cancelled()
    check_processing_cancelled()


def test_project_creation_stops_before_writing_when_already_cancelled(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "lines.xlsx"
    workbook.touch()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    event = threading.Event()
    event.set()

    with cancellation_scope(event), pytest.raises(ProcessingCancelled):
        create_project(
            workbook_path=workbook,
            audio_dir=audio_dir,
            project_dir=tmp_path / "work",
        )

    assert not (tmp_path / "work").exists()


def test_processing_screen_cancel_signals_worker_and_returns_to_start() -> None:
    class FakeRoot:
        def after(self, _delay, _callback):
            return None

    app = DialogueReviewApp.__new__(DialogueReviewApp)
    app.root = FakeRoot()
    app._cancel_event = threading.Event()
    app._cancel_button = None
    app._cancel_status = None
    app._worker_messages = None
    app._worker_thread = None
    app._progress = None
    app._log_text = None
    starts = []
    completions = []
    app.show_start = lambda: starts.append(True)
    app._pipeline_failed = lambda error: pytest.fail(str(error))

    app.cancel_processing()
    app._start_worker(
        lambda: check_processing_cancelled(),
        lambda result: completions.append(result),
    )
    assert app._worker_thread is not None
    app._worker_thread.join(timeout=2)
    app._poll_worker()

    assert starts == [True]
    assert completions == []
    assert app._worker_messages is None


def test_text_similarity() -> None:
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


@pytest.mark.parametrize(
    ("expected", "observed"),
    [
        ("C'mon... get up!", "Come on, get up!"),
        ("Could've asked.", "Could have asked."),
        ("I've had enough.", "I have had enough."),
        ("You're finished!", "You are finished!"),
        ("We won't survive.", "We will not survive."),
        ("You gotta be joking.", "You got to be joking."),
    ],
)
def test_spoken_form_normalization_accepts_equivalent_phrasing(
    expected: str,
    observed: str,
) -> None:
    fidelity = transcript_fidelity(expected, observed)

    assert text_similarity(expected, observed) == pytest.approx(100.0)
    assert fidelity["ordered_similarity"] == pytest.approx(100.0)
    assert fidelity["token_coverage"] == 1.0
    assert fidelity["token_precision"] == 1.0
    assert _candidate_reliability(
        line={"line": expected},
        match_score=100.0,
        margin=20.0,
        settings={},
        observed=observed,
        duration_plausibility=80.0,
    ) == (True, "")


def test_missing_single_sentence_boundary_prevents_auto_acceptance() -> None:
    expected = (
        "I live with that woman, and there is no way anyone who's heard "
        "her speak would want to bed her, much less live with her."
    )
    missing_start = (
        "And there is no way anyone who's heard her speak would want to "
        "bed her, much less live with her."
    )
    missing_end = (
        "I live with that woman, and there's no way anyone who's heard "
        "her speak would want to bed her."
    )

    start_fidelity = transcript_fidelity(expected, missing_start)
    end_fidelity = transcript_fidelity(expected, missing_end)
    assert start_fidelity["token_coverage"] > 0.75
    assert start_fidelity["prefix_token_coverage"] < 0.60
    assert end_fidelity["token_coverage"] > 0.75
    assert end_fidelity["suffix_token_coverage"] < 0.60
    assert _candidate_reliability(
        line={"line": expected},
        match_score=text_similarity(expected, missing_start),
        margin=40.0,
        settings={},
        observed=missing_start,
        duration_plausibility=80.0,
    ) == (False, "MISSING_LINE_START")
    assert _candidate_reliability(
        line={"line": expected},
        match_score=text_similarity(expected, missing_end),
        margin=40.0,
        settings={},
        observed=missing_end,
        duration_plausibility=80.0,
    ) == (False, "MISSING_LINE_END")


def test_single_missing_or_extra_opening_word_prevents_auto_acceptance() -> None:
    missing_expected = "Good, I've had enough of you for one day."
    missing_observed = "I've had enough of you for one day."
    prepended_expected = (
        "About time. Do you know how long I've been waiting for this?"
    )
    prepended_observed = (
        "Innkeeper. About time. Do you know how long I've been waiting "
        "for this?"
    )

    missing_fidelity = transcript_fidelity(
        missing_expected,
        missing_observed,
    )
    prepended_fidelity = transcript_fidelity(
        prepended_expected,
        prepended_observed,
    )
    assert missing_fidelity["prefix_missing_token_count"] == 1
    assert missing_fidelity["leading_missing_token_count"] == 1
    assert prepended_fidelity["leading_extra_token_count"] == 1
    assert _candidate_reliability(
        line={"line": missing_expected},
        match_score=text_similarity(missing_expected, missing_observed),
        margin=20.0,
        settings={},
        observed=missing_observed,
        duration_plausibility=80.0,
    ) == (False, "MISSING_LINE_START")
    assert _candidate_reliability(
        line={"line": prepended_expected},
        match_score=text_similarity(prepended_expected, prepended_observed),
        margin=20.0,
        settings={},
        observed=prepended_observed,
        duration_plausibility=80.0,
    ) == (False, "EXTRA_LINE_START")


def test_repeated_short_clauses_keep_their_earliest_positions() -> None:
    line = (
        "No, no! A staff! That's it! Should have turned you into a staff! "
        "Then you'd actually be useful."
    )

    fidelity = sentence_fidelity(line, line)

    assert fidelity["missing_clause_count"] == 0
    assert fidelity["clauses_in_order"] is True
    assert fidelity["clause_positions"] == sorted(
        fidelity["clause_positions"]
    )


def test_exact_asr_can_reroute_join_to_better_script_line() -> None:
    actions = [
        {
            "start_index": 10,
            "count": 3,
            "line_index": 0,
            "primary_line_index": 0,
            "match_score": 94.0,
            "confidence_margin": 5.0,
            "segment_match_rank": 1,
            "is_primary_match": True,
            "fragment_join": True,
            "fragment_source_count": 3,
            "fragment_join_provisional": True,
        }
    ]

    rerouted = _reroute_actions_to_exact_asr_primary_matches(
        actions,
        materialized_by_span={
            (10, 3): {"segment_id": "session__m00011_00013"}
        },
        exact_scores_by_segment={
            "session__m00011_00013": [94.0, 100.0]
        },
        minimum_score=45.0,
    )

    corrected = next(
        action for action in rerouted if action["line_index"] == 1
    )
    assert corrected["is_primary_match"] is True
    assert corrected["primary_line_index"] == 1
    assert corrected["fragment_join"] is True
    assert corrected["match_score"] == 100.0


def test_exact_line_scores_exclude_nonverbal_and_vocalization_lines() -> None:
    lines = [
        {"line": "(cough)"},
        {"line": "oof"},
        {"line": "Oof, that hurt."},
    ]
    evaluator = TranscriptEvaluator(lines, {})

    scores = _exact_line_scores(lines, evaluator, "oof")

    assert scores[:2] == [0.0, 0.0]
    assert scores[2] > 0.0


def test_grouped_alignment_settings_support_legacy_overrides() -> None:
    configured = default_alignment_config()
    configured["span_search"]["max_segments"] = 9
    configured["max_merge_segments"] = 7

    settings = AlignmentSettings.from_value(configured)

    assert settings["max_merge_segments"] == 7
    assert settings["fragment_join_max_segments"] == 8
    assert settings["duplicate_line_policy"] == "weak_order"
    assert settings["auto_reject_clipping"] is True


def test_grouped_alignment_settings_expose_every_effective_default() -> None:
    configured = default_alignment_config()

    settings = AlignmentSettings.from_value(configured)

    assert dict(settings) == ALIGNMENT_DEFAULTS
    assert settings["max_merge_segments"] == 8
    assert settings["fragment_join_max_segments"] == 8
    assert (
        configured["recovery"]["trim_candidates_per_segment"]
        == ALIGNMENT_DEFAULTS["intra_segment_trim_max_actions_per_segment"]
    )
    assert (
        configured["recovery"]["complete"]["maximum_length_ratio"]
        == ALIGNMENT_DEFAULTS["fragment_join_complete_max_length_ratio"]
    )


def test_grouped_v1_defaults_migrate_back_to_historical_limits() -> None:
    project = {
        "schema_version": 1,
        "segmentation": {
            "silence_noise_db": -45.0,
            "silence_detection_min_seconds": 0.35,
            "split_gap_seconds": 0.35,
        },
        "alignment": {
            "span_search": {"max_segments": 10},
            "recovery": {"max_segments": 10},
        },
    }

    migrated = migrate_project_config(project)
    settings = AlignmentSettings.from_value(migrated["alignment"])

    assert migrated["settings_version"] == 2
    assert settings["max_merge_segments"] == 8
    assert settings["fragment_join_max_segments"] == 8
    assert len(settings) == len(ALIGNMENT_DEFAULTS)
    assert migrated["segmentation"] == {
        "silence_noise_db": -40.0,
        "silence_detection_min_seconds": 0.20,
        "split_gap_seconds": 0.20,
    }


def test_generic_and_bandit_recordings_infer_narrow_sheet_mappings(
    tmp_path: Path,
) -> None:
    source_data = {
        "sheets": [
            {
                "name": "Any NPC using voice type ARG1RM",
                "line_count": 97,
                "voice_header": "You are voicing ARG1RMElfSly",
            },
            {
                "name": "Лист1",
                "line_count": 150,
                "voice_header": "You are voicing ARG1RMElfSly (Combat)",
            },
            {
                "name": "Member of faction ARGBanditFact",
                "line_count": 17,
                "voice_header": "You are voicing ARGBanditFaction",
            },
            {
                "name": "Лист2",
                "line_count": 64,
                "voice_header": "You are voicing ARGBanditFaction (Combat)",
            },
        ]
    }
    names = [
        "ARG_MaleElfSly_Generic_POST-PROC.wav",
        "ARG_MaleElfSly_Combat_POST-PROC.wav",
        "ARG_MaleElfSly_BanditGeneric_POST-PROC.wav",
        "ARG_MaleElfSly_BanditCombat_POST-PROC.wav",
    ]
    audio_files = [tmp_path / name for name in names]

    sessions = infer_sessions(audio_files, source_data, tmp_path)
    mappings = {session["id"]: session for session in sessions}

    assert mappings["arg_maleelfsly_generic_post_proc"]["sheets"] == [
        "Any NPC using voice type ARG1RM"
    ]
    assert mappings["arg_maleelfsly_combat_post_proc"]["sheets"] == ["Лист1"]
    assert mappings["arg_maleelfsly_banditgeneric_post_proc"]["sheets"] == [
        "Member of faction ARGBanditFact"
    ]
    assert mappings["arg_maleelfsly_banditcombat_post_proc"]["sheets"] == [
        "Лист2"
    ]
    assert not any(session["needs_mapping_review"] for session in sessions)


def test_exact_short_match_still_requires_resolved_ambiguity() -> None:
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
    ) == (False, "SHORT_LINE_AMBIGUOUS")
    assert _candidate_reliability(
        line=line,
        match_score=100.0,
        margin=5.0,
        settings=settings,
        observed="Goodbye",
        ambiguity_resolved=True,
    ) == (True, "")


def test_clipping_blocks_automatic_reliability() -> None:
    assert _candidate_reliability(
        line={"line": "This transcript is otherwise complete."},
        match_score=100.0,
        margin=100.0,
        settings={},
        observed="This transcript is otherwise complete.",
        clipping_samples=1,
        technical_score=65.0,
    ) == (False, "TECHNICAL_CLIPPING")


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
    limited = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_segments": 2,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
            "fragment_join_max_actions": 1,
        },
    )
    assert len(limited) == 1


def test_boundary_completion_can_join_repeated_short_clauses() -> None:
    line_text = "No! Don't be dead... Don't be dead..."
    lines = [{"line_id": "line", "line": line_text}]
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 0.8,
            "transcript": "No!",
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 0.9,
            "end_seconds": 2.9,
            "transcript": "Don't be dead! Don't be dead!",
            "asr_probability": 0.95,
        },
    ]
    actions = [
        {
            "start_index": index,
            "count": 1,
            "line_index": 0,
            "match_score": text_similarity(
                line_text,
                segment["transcript"],
            ),
            "transcript": segment["transcript"],
            "duration_plausibility": 70.0,
            "order_hint": 0.0,
            "top_matches": [],
        }
        for index, segment in enumerate(segments)
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
    )

    assert len(joined) == 1
    assert joined[0]["start_index"] == 0
    assert joined[0]["count"] == 2
    assert joined[0]["match_score"] == 100.0


def test_fragment_join_keeps_bounded_high_score_fallbacks() -> None:
    lines = [
        {
            "line_id": "target",
            "line": "You win... I... I surrender!",
        },
        {"line_id": "win", "line": "You win."},
        {"line_id": "aye", "line": "Aye."},
    ]
    transcripts = [
        "I surrender.",
        "You win.",
        "Aye.",
        "I surrender.",
    ]
    segments = [
        {
            "start_seconds": index * 1.1,
            "end_seconds": index * 1.1 + 1.0,
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
            "line_index": line_index,
            "match_score": text_similarity(
                lines[line_index]["line"],
                transcript,
            ),
            "transcript": transcript,
            "duration_plausibility": 80.0,
            "order_hint": 0.0,
            "top_matches": [],
        }
        for index, (line_index, transcript) in enumerate(
            zip([2, 1, 2, 0], transcripts)
        )
    ]
    words = [
        {"word": " Win", "start": 1.2, "end": 1.8},
        {"word": " I", "start": 2.3, "end": 2.5},
        {"word": " I", "start": 2.5, "end": 2.7},
        {"word": " surrender", "start": 3.4, "end": 3.9},
    ]
    transcription = {
        "segments": [{"start": 1.2, "end": 3.9, "words": words}]
    }

    without_fallback = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_segments": 4,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
            "fragment_join_fallback_max_actions": 0,
        },
        transcription=transcription,
    )
    with_fallback = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_segments": 4,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
            "fragment_join_fallback_max_actions": 2,
        },
        transcription=transcription,
    )

    assert not any(
        action["start_index"] == 1 and action["count"] == 3
        for action in without_fallback
    )
    restored = next(
        action
        for action in with_fallback
        if action["start_index"] == 1 and action["count"] == 3
    )
    assert restored["fragment_join_fallback"] is True
    assert restored["match_score"] >= 90.0
    assert (
        sum(
            bool(action.get("fragment_join_fallback"))
            for action in with_fallback
        )
        == 2
    )


def test_oversized_base_segment_can_recover_pause_bounded_trim() -> None:
    line_text = (
        "Oh! Is that all? Not that I ever say anything true while drunk."
    )
    lines = [{"line_id": "target", "line": line_text}]
    word_text = [
        "Oh",
        "is",
        "that",
        "all",
        "Not",
        "that",
        "I",
        "ever",
        "say",
        "anything",
        "true",
        "while",
        "drunk",
    ]
    words = [
        {
            "word": f" {word}",
            "start": index * 0.2,
            "end": index * 0.2 + 0.18,
        }
        for index, word in enumerate(word_text)
    ]
    words.extend(
        [
            {"word": " Yes", "start": 3.2, "end": 3.5},
            {"word": " appended", "start": 3.5, "end": 3.9},
            {"word": " sentence", "start": 3.9, "end": 4.3},
        ]
    )
    full_transcript = " ".join(
        str(word["word"]).strip() for word in words
    )
    base_start_sample = 1000
    segment = {
        "segment_id": "session__s00001",
        "start_sample": base_start_sample,
        "end_sample": base_start_sample + 5 * 48000,
        "start_seconds": 10.0,
        "end_seconds": 15.0,
        "transcript": full_transcript,
        "asr_probability": 0.95,
        "segment_asr": {
            "primary": {
                "transcript": full_transcript,
                "words": words,
            }
        },
    }
    action = {
        "start_index": 0,
        "count": 1,
        "line_index": 0,
        "match_score": text_similarity(line_text, full_transcript),
        "transcript": full_transcript,
        "duration_plausibility": 60.0,
        "order_hint": 0.0,
        "top_matches": [],
    }

    trimmed = _intra_segment_trim_actions(
        [action],
        lines=lines,
        base_segments=[segment],
        sample_rate=48000,
        settings={},
        evaluator=TranscriptEvaluator(lines, {}),
    )

    assert len(trimmed) == 1
    assert trimmed[0]["intra_segment_trim"] is True
    assert trimmed[0]["trim_start_sample"] == base_start_sample
    assert trimmed[0]["trim_end_sample"] < segment["end_sample"]
    assert trimmed[0]["transcript"].endswith("drunk")
    assert "appended" not in trimmed[0]["transcript"]
    assert trimmed[0]["match_score"] == 100.0
    expanded = _expand_alignment_actions(
        trimmed,
        lines=lines,
        settings={},
    )
    assert expanded[0]["intra_segment_trim"] is True
    assert (
        expanded[0]["trim_end_sample"]
        == trimmed[0]["trim_end_sample"]
    )


def test_secondary_near_complete_match_can_seed_fragment_join() -> None:
    lines = [
        {"line_id": "short", "line": "You should leave."},
        {"line_id": "target", "line": "You should leave. Now."},
        {"line_id": "now", "line": "Now."},
    ]
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 1.2,
            "transcript": "You should leave.",
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 1.2,
            "end_seconds": 2.0,
            "transcript": "Now.",
            "asr_probability": 0.95,
        },
    ]
    actions = [
        {
            "start_index": 0,
            "count": 1,
            "line_index": 0,
            "match_score": 100.0,
            "transcript": segments[0]["transcript"],
            "duration_plausibility": 80.0,
            "order_hint": 0.0,
            "top_matches": [
                {"line_index": 0, "match_score": 100.0},
                {
                    "line_index": 1,
                    "match_score": text_similarity(
                        lines[1]["line"],
                        segments[0]["transcript"],
                    ),
                },
            ],
        },
        {
            "start_index": 1,
            "count": 1,
            "line_index": 2,
            "match_score": 100.0,
            "transcript": segments[1]["transcript"],
            "duration_plausibility": 80.0,
            "order_hint": 0.0,
            "top_matches": [
                {"line_index": 2, "match_score": 100.0}
            ],
        },
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
    )

    recovered = next(action for action in joined if action["line_index"] == 1)
    assert recovered["transcript"] == "You should leave. Now."
    assert recovered["fragment_join_provisional"] is False


def test_uncertain_short_boundary_audio_is_kept_for_review() -> None:
    line_text = "No... escape..."
    lines = [
        {"line_id": "target", "line": line_text},
        {"line_id": "other", "line": "Oh, come on! I need it more!"},
    ]
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 0.9,
            "transcript": "Oh",
            "asr_probability": 0.08,
        },
        {
            "start_seconds": 0.9,
            "end_seconds": 2.7,
            "transcript": "Escape.",
            "asr_probability": 0.43,
        },
    ]
    actions = [
        {
            "start_index": 0,
            "count": 1,
            "line_index": 1,
            "match_score": text_similarity(
                lines[1]["line"],
                segments[0]["transcript"],
            ),
            "transcript": segments[0]["transcript"],
            "duration_plausibility": 50.0,
            "order_hint": 0.0,
            "top_matches": [],
        },
        {
            "start_index": 1,
            "count": 1,
            "line_index": 0,
            "match_score": text_similarity(
                line_text,
                segments[1]["transcript"],
            ),
            "transcript": segments[1]["transcript"],
            "duration_plausibility": 50.0,
            "order_hint": 0.0,
            "top_matches": [],
        },
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
    )

    recovered = next(action for action in joined if action["line_index"] == 0)
    assert recovered["start_index"] == 0
    assert recovered["count"] == 2
    assert recovered["forced_review_reason"] == "UNCERTAIN_BOUNDARY_AUDIO"


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


def test_fragment_join_prefers_complete_segment_asr_over_stale_session_words() -> None:
    line_text = "Maybe over here... No, nothing."
    lines = [{"line_id": "line", "line": line_text}]
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "transcript": "Maybe over here",
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 1.2,
            "end_seconds": 2.0,
            "transcript": "No nothing",
            "asr_probability": 0.95,
        },
    ]
    transcription = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "words": [
                    {
                        "start": 0.0,
                        "end": 0.3,
                        "word": " Maybe",
                        "probability": 0.99,
                    },
                    {
                        "start": 0.3,
                        "end": 0.6,
                        "word": " over",
                        "probability": 0.99,
                    },
                    {
                        "start": 0.6,
                        "end": 0.9,
                        "word": " here",
                        "probability": 0.99,
                    },
                ],
            }
        ]
    }
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
            "duration_plausibility": 60.0,
            "order_hint": 0.0,
            "top_matches": [],
        }
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
        transcription=transcription,
    )

    assert len(joined) == 1
    assert joined[0]["start_index"] == 0
    assert joined[0]["count"] == 2
    assert joined[0]["transcript"] == "Maybe over here No nothing"
    assert joined[0]["transcript_source"] == "segment_asr_span"


def test_fragment_join_recovers_single_clause_split_at_internal_pause() -> None:
    line_text = "Only my consciousness remains, trapped in this dream."
    lines = [{"line_id": "line", "line": line_text}]
    transcripts = [
        "Only my consciousness remains",
        "trapped in this dream",
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
            "start_index": 0,
            "count": 1,
            "line_index": 0,
            "match_score": text_similarity(line_text, transcripts[0]),
            "transcript": transcripts[0],
            "duration_plausibility": 60.0,
            "order_hint": 0.0,
            "top_matches": [],
        }
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
    )

    assert [(action["start_index"], action["count"]) for action in joined] == [
        (0, 2)
    ]


def test_fragment_join_can_recover_four_segments_around_partial_seed() -> None:
    line_text = (
        "The tree here is an illusion. What point would there be? "
        "No. He's swayed you!"
    )
    lines = [{"line_id": "line", "line": line_text}]
    transcripts = [
        "The tree here is an illusion",
        "What point would there be",
        "No",
        "He's swayed you",
    ]
    segments = [
        {
            "start_seconds": index * 1.2,
            "end_seconds": index * 1.2 + 1.0,
            "transcript": transcript,
            "asr_probability": 0.95,
        }
        for index, transcript in enumerate(transcripts)
    ]
    partial = f"{transcripts[1]} {transcripts[2]}"
    actions = [
        {
            "type": "assigned",
            "start_index": 1,
            "count": 2,
            "line_index": 0,
            "match_score": text_similarity(line_text, partial),
            "transcript": partial,
            "duration_plausibility": 70.0,
            "order_hint": 0.0,
            "top_matches": [],
        }
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_segments": 4,
            "fragment_join_max_segments": 2,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
    )

    assert any(
        action["start_index"] == 0 and action["count"] == 4
        for action in joined
    )


def test_fragment_join_sends_plausible_incomplete_span_to_exact_asr() -> None:
    line_text = "Bah. Fine. I can see you're no fool. What do you want?"
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 0.8,
            "transcript": "Ugh, fine",
            "asr_probability": 0.9,
        },
        {
            "start_seconds": 1.0,
            "end_seconds": 2.2,
            "transcript": "I can see you're no fool",
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 2.4,
            "end_seconds": 3.2,
            "transcript": "What do you want",
            "asr_probability": 0.95,
        },
    ]
    partial = " ".join(segment["transcript"] for segment in segments[:2])
    actions = [
        {
            "type": "assigned",
            "start_index": 0,
            "count": 2,
            "line_index": 0,
            "match_score": text_similarity(line_text, partial),
            "transcript": partial,
            "duration_plausibility": 70.0,
            "order_hint": 0.0,
            "top_matches": [],
        }
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=[{"line_id": "line", "line": line_text}],
        base_segments=segments,
        settings={
            "max_merge_segments": 3,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
    )

    recovered = next(
        action
        for action in joined
        if action["start_index"] == 0 and action["count"] == 3
    )
    assert recovered["fragment_join_provisional"] is True


def test_fragment_join_recovers_single_sentence_missing_end() -> None:
    line_text = (
        "I live with that woman, and there is no way anyone who's heard "
        "her speak would want to bed her, much less live with her."
    )
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 6.0,
            "transcript": (
                "I live with that woman, and there's no way anyone who's "
                "heard her speak would want to bed her."
            ),
            "asr_probability": 0.98,
        },
        {
            "start_seconds": 6.2,
            "end_seconds": 8.0,
            "transcript": "Much less live with her.",
            "asr_probability": 0.96,
        },
    ]
    partial = segments[0]["transcript"]
    actions = [
        {
            "type": "assigned",
            "start_index": 0,
            "count": 1,
            "line_index": 0,
            "match_score": text_similarity(line_text, partial),
            "transcript": partial,
            "duration_plausibility": 75.0,
            "order_hint": 0.0,
            "top_matches": [],
        }
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=[{"line_id": "Palioth::R53", "line": line_text}],
        base_segments=segments,
        settings={
            "max_merge_segments": 10,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 20.0,
        },
    )

    recovered = next(
        action
        for action in joined
        if action["start_index"] == 0 and action["count"] == 2
    )
    fidelity = transcript_fidelity(line_text, recovered["transcript"])
    assert fidelity["prefix_token_coverage"] == 1.0
    assert fidelity["suffix_token_coverage"] == 1.0


def test_fragment_join_recovers_seven_fragments_with_ten_segment_limit() -> None:
    line_text = (
        "I live with that woman, and there is no way anyone who's heard "
        "her speak would want to bed her, much less live with her."
    )
    transcripts = [
        "I live with that woman.",
        "And there is no way.",
        "Anyone",
        "who's heard her speak",
        "would want to bed her",
        "much less",
        "live with her",
    ]
    segments = [
        {
            "start_seconds": index * 1.2,
            "end_seconds": index * 1.2 + 1.0,
            "transcript": transcript,
            "asr_probability": 0.95,
        }
        for index, transcript in enumerate(transcripts)
    ]
    partial = " ".join(transcripts[1:])
    actions = [
        {
            "type": "assigned",
            "start_index": 1,
            "count": 6,
            "line_index": 0,
            "match_score": text_similarity(line_text, partial),
            "transcript": partial,
            "duration_plausibility": 75.0,
            "order_hint": 0.0,
            "top_matches": [],
        }
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=[{"line_id": "Palioth::R53", "line": line_text}],
        base_segments=segments,
        settings={
            "max_merge_segments": 10,
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 20.0,
        },
    )

    assert any(
        action["start_index"] == 0 and action["count"] == 7
        for action in joined
    )


def test_inline_nonverbal_cues_do_not_make_spoken_line_incomplete() -> None:
    line_text = "(laugh) See! What did I tell you?"
    observed = "See! What did I tell you?"

    assert text_similarity(line_text, observed) == pytest.approx(100.0)
    assert sentence_fidelity(line_text, observed)["missing_clause_count"] == 0
    assert _candidate_reliability(
        line={"line": line_text},
        match_score=100.0,
        margin=20.0,
        settings={},
        observed=observed,
    ) == (True, "")


def test_fragment_join_searches_neighbor_before_worse_selected_candidate() -> None:
    line_text = "All right, okay... Sounds good."
    lines = [{"line_id": "line", "line": line_text}]
    transcripts = [
        "Oh all right okay",
        "Sounds good",
        "Alright, okay. Sounds good. Alright, okay.",
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
            "start_index": 2,
            "count": 1,
            "line_index": 0,
            "match_score": text_similarity(line_text, transcripts[2]),
            "transcript": transcripts[2],
            "duration_plausibility": 45.0,
            "order_hint": 0.0,
            "top_matches": [],
        }
    ]

    joined = _multisentence_fragment_join_actions(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
    )

    assert any(
        action["start_index"] == 0 and action["count"] == 2
        for action in joined
    )


def test_fragment_join_skips_line_with_complete_contraction_variant() -> None:
    line_text = "Could've been worse."
    lines = [{"line_id": "line", "line": line_text}]
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "transcript": "Could have been worse",
            "asr_probability": 0.95,
        },
        {
            "start_seconds": 1.2,
            "end_seconds": 1.8,
            "transcript": "Seems",
            "asr_probability": 0.95,
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
            "max_merge_gap_seconds": 1.0,
            "max_span_seconds": 10.0,
        },
    )

    assert joined == []


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


def test_exact_asr_boundary_gap_rejects_complete_fallback_join() -> None:
    unsafe = _has_unsafe_untranscribed_merge(
        action={"start_index": 0, "count": 2},
        base_segments=[
            {
                "transcript": "Thank you.",
                "metrics": {"duration_seconds": 1.2, "rms_dbfs": -48.0},
                "segment_asr": {"silence_rejected": True},
            },
            {
                "transcript": "Hello there.",
                "metrics": {"duration_seconds": 1.5, "rms_dbfs": -20.0},
            },
        ],
        segment={
            "start_seconds": 0.0,
            "end_seconds": 2.7,
            "metrics": {"duration_seconds": 2.7},
            "words": [
                {"word": "Hello", "start": 1.2, "end": 1.6},
                {"word": "there", "start": 1.6, "end": 2.2},
            ],
        },
        settings={},
    )

    assert unsafe is True


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


def test_segment_padding_is_clamped_to_a_shared_silence_boundary() -> None:
    regions = [
        {
            "speech_start": 0.0,
            "speech_end": 1.0,
            "start": 0.0,
            "end": 1.4,
        },
        {
            "speech_start": 1.2,
            "speech_end": 2.0,
            "start": 0.9,
            "end": 2.0,
        },
    ]

    result = prevent_region_overlaps(regions)

    assert result[0]["end"] == pytest.approx(1.1)
    assert result[1]["start"] == pytest.approx(1.1)


def test_word_gap_splitting_adapts_beyond_soft_boundary_cap() -> None:
    words = []
    for index in range(8):
        start = index * 4.0 + 0.2
        words.append(
            {
                "start": start,
                "end": start + 0.5,
                "word": f" take{index}",
            }
        )
    transcription = {
        "segments": [
            {
                "start": 0.0,
                "end": 32.0,
                "text": "takes",
                "words": words,
            }
        ]
    }

    refined = split_regions_on_word_gaps(
        [
            {
                "speech_start": 0.0,
                "speech_end": 32.0,
                "start": 0.0,
                "end": 32.0,
            }
        ],
        transcription,
        duration_seconds=32.0,
        settings={
            "word_split_enabled": True,
            "word_split_gap_seconds": 0.3,
            "word_split_min_region_seconds": 1.5,
            "word_split_max_boundaries": 2,
            "word_split_max_segment_seconds": 8.0,
            "minimum_segment_seconds": 0.15,
            "pre_padding_seconds": 0.1,
            "post_padding_seconds": 0.1,
        },
    )

    assert len(refined) > 3
    assert max(
        region["speech_end"] - region["speech_start"] for region in refined
    ) <= 8.0


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


def test_duplicate_weak_order_distributes_nearby_distinct_takes() -> None:
    lines = [
        {"line_id": "a", "line": "I'll bury you!"},
        {"line_id": "b", "line": "I'll bury you!"},
    ]
    segments = [
        {"start_seconds": 0.0, "end_seconds": 1.0},
        {"start_seconds": 1.2, "end_seconds": 2.2},
    ]
    actions = [
        {
            "start_index": index,
            "count": 1,
            "line_index": 0,
            "match_score": 100.0,
            "top_matches": [
                {"line_index": 0, "match_score": 100.0},
                {"line_index": 1, "match_score": 100.0},
            ],
        }
        for index in range(2)
    ]

    _apply_duplicate_line_policy(
        actions,
        lines=lines,
        base_segments=segments,
        settings={
            "duplicate_line_policy": "weak_order",
            "take_group_gap_seconds": 12.0,
        },
    )

    assert [action["line_index"] for action in actions] == [0, 1]
    assert all(action["duplicate_resolved"] for action in actions)


def test_duplicate_review_and_reuse_expand_only_identical_targets() -> None:
    lines = [
        {"line": "Same words."},
        {"line": "Same words!"},
        {"line": "Different words."},
    ]
    action = {
        "type": "assigned",
        "start_index": 0,
        "count": 1,
        "line_index": 0,
        "match_score": 100.0,
        "confidence_margin": 0.0,
        "transcript": "Same words",
        "duration_plausibility": 100.0,
        "order_hint": 0.0,
    }

    review_actions = _expand_alignment_actions(
        [action],
        lines=lines,
        settings={"duplicate_line_policy": "review"},
    )
    reuse_actions = _expand_alignment_actions(
        [action],
        lines=lines,
        settings={"duplicate_line_policy": "reuse"},
    )

    assert [item["line_index"] for item in review_actions] == [0, 1]
    assert all(item["is_primary_match"] for item in review_actions)
    assert not any(item["duplicate_resolved"] for item in review_actions)
    assert [item["line_index"] for item in reuse_actions] == [0, 1]
    assert all(item["duplicate_resolved"] for item in reuse_actions)


def test_line_review_types_nonverbal_lines_and_uses_audible_unmatched_pool(
    tmp_path: Path,
) -> None:
    line = {
        "line_id": "Sheet::R3",
        "sheet": "Sheet",
        "sheet_index": 0,
        "excel_row": 3,
        "quest": "",
        "context": "Standing near the market.",
        "line": "(cough)",
        "acting_note": "A restrained cough.",
        "emotion": "",
        "target_filename": "cough_target",
    }
    review_path = tmp_path / "line_review.json"
    review = build_line_review(
        source_lines=[line],
        candidates_by_line={},
        unmatched_segments=[
            {
                "segment_id": "audible",
                "segment_file": "audible.wav",
                "transcript": "",
                "technical_score": 80.0,
                "audible": True,
            },
            {
                "segment_id": "quiet",
                "segment_file": "quiet.wav",
                "transcript": "",
                "technical_score": 90.0,
                "audible": False,
            },
        ],
    )
    save_line_review(review_path, review)
    loaded = load_line_review(review_path)

    assert loaded["lines"][0]["type"] == "nonverbal"
    assert loaded["lines"][0]["context"] == "Standing near the market."
    assert loaded["lines"][0]["acting_note"] == "A restrained cough."
    assert loaded["lines"][0]["status"] == "REVIEW"
    assert loaded["lines"][0]["selected_segment_id"] is None
    assert loaded["lines"][0]["candidates"] == []
    assert [
        segment["segment_id"] for segment in loaded["unmatched_segments"]
    ] == ["audible"]


def test_finalize_omits_unselected_lines(tmp_path: Path) -> None:
    line = {
        "line_id": "Sheet::R3",
        "sheet": "Sheet",
        "sheet_index": 0,
        "excel_row": 3,
        "line": "No take was selected.",
        "target_filename": "unselected",
    }
    review_path = tmp_path / "line_review.json"
    save_line_review(
        review_path,
        build_line_review(
            source_lines=[line],
            candidates_by_line={},
            unmatched_segments=[],
        ),
    )
    write_json(tmp_path / "segments_manifest.json", {"sessions": []})

    result = finalize_review(
        project_dir=tmp_path,
        project={"export": {}},
        review_path=review_path,
        output_dir=tmp_path / "final",
    )

    assert result["export_count"] == 0
    assert result["error_count"] == 0


def test_review_candidates_keep_primary_top_score_cluster() -> None:
    def candidate(
        segment_id: str,
        score: float,
        *,
        primary: bool = True,
    ) -> dict:
        return {
            "segment_id": segment_id,
            "session_id": "session",
            "base_indices": [int(segment_id[-1])],
            "match_score": score,
            "selection_score": score,
            "is_primary_match": primary,
            "reliable": score >= 90.0,
            "reliability_reason": "",
        }

    retained = prune_line_candidates(
        [
            candidate("segment1", 99.0),
            candidate("segment2", 96.0),
            candidate("segment3", 88.0),
            candidate("segment4", 70.0),
            candidate("segment5", 98.0, primary=False),
        ]
    )

    assert [item["segment_id"] for item in retained] == [
        "segment1",
        "segment2",
        "segment3",
    ]


def test_review_auto_selects_best_quality_within_reliable_cluster() -> None:
    line = {
        "line_id": "Sheet::R3",
        "sheet": "Sheet",
        "excel_row": 3,
        "line": "Choose the clean take.",
        "target_filename": "clean_take",
    }
    candidates = [
        {
            "segment_id": "highest_text",
            "segment_file": "highest_text.wav",
            "session_id": "session",
            "base_indices": [0],
            "match_score": 100.0,
            "selection_score": 105.0,
            "technical_score": 50.0,
            "is_primary_match": True,
            "reliable": True,
            "reliability_reason": "",
        },
        {
            "segment_id": "cleanest",
            "segment_file": "cleanest.wav",
            "session_id": "session",
            "base_indices": [1],
            "match_score": 99.0,
            "selection_score": 116.0,
            "technical_score": 100.0,
            "is_primary_match": True,
            "reliable": True,
            "reliability_reason": "",
        },
    ]

    review = build_line_review(
        source_lines=[line],
        candidates_by_line={line["line_id"]: candidates},
        unmatched_segments=[],
    )

    assert review["lines"][0]["status"] == "AUTO_OK"
    assert review["lines"][0]["selected_segment_id"] == "cleanest"


def test_review_candidates_keep_best_fragment_join_when_none_are_reliable() -> None:
    retained = prune_line_candidates(
        [
            {
                "segment_id": "partial",
                "session_id": "session",
                "base_indices": [1],
                "match_score": 91.0,
                "selection_score": 91.0,
                "is_primary_match": True,
                "reliable": False,
                "reliability_reason": "MISSING_SENTENCE",
                "fragment_join": False,
            },
            {
                "segment_id": "joined",
                "session_id": "session",
                "base_indices": [0, 1],
                "match_score": 73.0,
                "selection_score": 73.0,
                "is_primary_match": True,
                "reliable": False,
                "reliability_reason": "UNCERTAIN_BOUNDARY_AUDIO",
                "fragment_join": True,
            },
        ]
    )

    assert [candidate["segment_id"] for candidate in retained] == [
        "partial",
        "joined",
    ]


def test_review_regeneration_preserves_manual_selection() -> None:
    line = {
        "line_id": "Sheet::R3",
        "sheet": "Sheet",
        "excel_row": 3,
        "line": "Hello there.",
        "target_filename": "hello",
    }
    candidate = {
        "segment_id": "session__s00001",
        "segment_file": "segment.wav",
        "session_id": "session",
        "base_indices": [0],
        "transcript": "Hello there.",
        "match_score": 100.0,
        "selection_score": 100.0,
        "reliable": True,
    }
    previous = build_line_review(
        source_lines=[line],
        candidates_by_line={line["line_id"]: [candidate]},
        unmatched_segments=[],
    )
    previous["lines"][0]["status"] = "MANUALLY_REVIEWED"
    new = build_line_review(
        source_lines=[line],
        candidates_by_line={},
        unmatched_segments=[],
    )

    merged = preserve_manual_selections(new, previous)

    assert merged["lines"][0]["status"] == "MANUALLY_REVIEWED"
    assert merged["lines"][0]["selected_segment_id"] == "session__s00001"
    assert merged["lines"][0]["candidates"][0]["segment_id"] == (
        "session__s00001"
    )


def test_missing_line_can_select_and_preserve_unmatched_segment() -> None:
    line = {
        "line_id": "Sheet::R3",
        "sheet": "Sheet",
        "excel_row": 3,
        "line": "No automatic candidate.",
        "target_filename": "missing",
    }
    unmatched = {
        "segment_id": "session__s00001",
        "segment_file": "segment.wav",
        "session_id": "session",
        "base_index": 0,
        "transcript": "Maybe this is it",
        "technical_score": 82.0,
        "audible": True,
    }
    previous = build_line_review(
        source_lines=[line],
        candidates_by_line={},
        unmatched_segments=[unmatched],
    )
    previous_line = previous["lines"][0]
    assert previous_line["status"] == "MISSING"
    assert _uses_unmatched_candidates(previous_line)
    previous_line["selected_segment_id"] = unmatched["segment_id"]
    previous_line["status"] = "MANUALLY_REVIEWED"
    assert _uses_unmatched_candidates(previous_line)
    assert _selected_segment_score(previous, previous_line) == 82.0

    regenerated = build_line_review(
        source_lines=[line],
        candidates_by_line={},
        unmatched_segments=[],
    )
    merged = preserve_manual_selections(regenerated, previous)

    assert merged["lines"][0]["status"] == "MANUALLY_REVIEWED"
    assert merged["lines"][0]["selected_segment_id"] == unmatched["segment_id"]
    assert merged["unmatched_segments"][0]["segment_id"] == unmatched["segment_id"]


def test_text_aligner_excludes_nonverbal_lines() -> None:
    actions = order_independent_align(
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
            "candidate_min_score": 35.0,
        },
    )

    assert actions
    assert {action["line_index"] for action in actions} == {1}


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


def test_high_no_speech_low_rms_transcript_is_rejected_as_hallucination() -> None:
    segment = {
        "metrics": {
            "duration_seconds": 1.1,
            "rms_dbfs": -48.0,
        }
    }
    primary = {
        "transcript": "Thank you.",
        "segments": [{"no_speech_prob": 0.91}],
    }

    assert _likely_silence_hallucination(segment, primary, {}) is True
    assert (
        _likely_silence_hallucination(
            {**segment, "metrics": {"rms_dbfs": -20.0}},
            primary,
            {},
        )
        is False
    )


def test_segment_asr_batches_independent_clips_with_configured_size(
    tmp_path: Path,
) -> None:
    segment_records = []
    for index in range(3):
        segment_file = tmp_path / f"segment_{index}.wav"
        _write_tone(segment_file, duration_seconds=0.5)
        segment_records.append(
            {
                "segment_id": f"session__s{index + 1:05d}",
                "kind": "base",
                "file": segment_file.name,
                "source_sha256": "source-hash",
                "start_sample": index * 24000,
                "end_sample": (index + 1) * 24000,
                "start_seconds": index * 0.5,
                "end_seconds": (index + 1) * 0.5,
                "transcript": "",
                "words": [],
                "word_count": 0,
                "asr_probability": None,
                "metrics": {"duration_seconds": 0.5},
            }
        )
    line = {
        "line_id": "Actor::R2",
        "sheet": "Actor",
        "sheet_index": 0,
        "excel_row": 2,
        "line": "Hello.",
    }
    write_json(tmp_path / "source_lines.json", {"lines": [line]})
    write_json(
        tmp_path / "segments_manifest.json",
        {
            "sessions": [
                {
                    "session_id": "session",
                    "segments": segment_records,
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
            "batch_size": 2,
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
    batch_sizes = []

    class FakeBatchedModel:
        def transcribe(self, _audio, **kwargs):
            clips = kwargs["clip_timestamps"]
            batch_sizes.append(kwargs["batch_size"])
            decoded = []
            for index, clip in enumerate(clips):
                start = int(clip["start"] * 16000) / 16000
                decoded.append(
                    SimpleNamespace(
                        start=start + 0.05,
                        end=start + 0.35,
                        text=f"Hello {index + 1}",
                        avg_logprob=-0.1,
                        no_speech_prob=0.01,
                        words=[
                            SimpleNamespace(
                                start=start + 0.05,
                                end=start + 0.25,
                                word=" Hello",
                                probability=0.95,
                            )
                        ],
                    )
                )
            return iter(decoded), SimpleNamespace(language="en")

    runtime = {
        "model": SimpleNamespace(
            feature_extractor=SimpleNamespace(sampling_rate=16000)
        ),
        "batched_model": FakeBatchedModel(),
    }
    transcribe_segments_project(
        project_dir=tmp_path,
        project=project,
        runtime=runtime,
    )

    manifest = read_json(tmp_path / "segments_manifest.json")
    segments = manifest["sessions"][0]["segments"]
    assert batch_sizes == [2, 1]
    assert manifest["segment_transcription"]["batch_size"] == 2
    assert all(segment["transcript_source"] == "segment_asr" for segment in segments)
    assert segments[1]["segment_asr"]["primary"]["words"][0]["start"] == pytest.approx(
        0.05
    )


def test_segment_asr_batches_prompted_fallbacks_that_share_a_prompt(
    tmp_path: Path,
) -> None:
    segment_records = []
    for index in range(3):
        segment_file = tmp_path / f"prompt_segment_{index}.wav"
        _write_tone(segment_file, duration_seconds=0.5)
        segment_records.append(
            {
                "segment_id": f"session__s{index + 1:05d}",
                "kind": "base",
                "file": segment_file.name,
                "source_sha256": "source-hash",
                "start_sample": index * 24000,
                "end_sample": (index + 1) * 24000,
                "start_seconds": index * 0.5,
                "end_seconds": (index + 1) * 0.5,
                "transcript": "",
                "words": [],
                "word_count": 0,
                "asr_probability": None,
                "metrics": {"duration_seconds": 0.5},
            }
        )
    line = {
        "line_id": "Actor::R2",
        "sheet": "Actor",
        "sheet_index": 0,
        "excel_row": 2,
        "line": "Hello.",
    }
    write_json(tmp_path / "source_lines.json", {"lines": [line]})
    write_json(
        tmp_path / "segments_manifest.json",
        {
            "sessions": [
                {
                    "session_id": "session",
                    "segments": segment_records,
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
            "batch_size": 3,
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
    calls = []

    class FakeBatchedModel:
        def transcribe(self, _audio, **kwargs):
            calls.append(kwargs)
            decoded = []
            for clip in kwargs["clip_timestamps"]:
                start = int(clip["start"] * 16000) / 16000
                prompted = bool(kwargs.get("initial_prompt"))
                decoded.append(
                    SimpleNamespace(
                        start=start,
                        end=start + 0.4,
                        text="Hello" if prompted else "",
                        avg_logprob=-0.1 if prompted else -1.0,
                        no_speech_prob=0.01 if prompted else 0.8,
                        words=(
                            [
                                SimpleNamespace(
                                    start=start + 0.05,
                                    end=start + 0.25,
                                    word=" Hello",
                                    probability=0.95,
                                )
                            ]
                            if prompted
                            else []
                        ),
                    )
                )
            return iter(decoded), SimpleNamespace(language="en")

    runtime = {
        "model": SimpleNamespace(
            feature_extractor=SimpleNamespace(
                sampling_rate=16000,
                chunk_length=30,
            )
        ),
        "batched_model": FakeBatchedModel(),
    }
    transcribe_segments_project(
        project_dir=tmp_path,
        project=project,
        runtime=runtime,
    )

    manifest = read_json(tmp_path / "segments_manifest.json")
    assert [call["batch_size"] for call in calls] == [3, 3]
    assert "initial_prompt" not in calls[0]
    assert calls[1]["initial_prompt"] == "Hello."
    assert manifest["segment_transcription"]["primary_inference_batch_count"] == 1
    assert manifest["segment_transcription"]["prompted_fallback_count"] == 3
    assert manifest["segment_transcription"]["prompted_inference_batch_count"] == 1
    assert all(
        segment["transcript_source"] == "segment_asr_prompted"
        for segment in manifest["sessions"][0]["segments"]
    )


def test_long_clip_avoids_batched_pipeline_truncation(tmp_path: Path) -> None:
    audio_path = tmp_path / "long.wav"
    _write_tone(audio_path, duration_seconds=30.1)
    direct_calls = []
    batched_calls = []
    decoded_segment = SimpleNamespace(
        start=0.0,
        end=30.1,
        text="Complete long clip",
        avg_logprob=-0.1,
        no_speech_prob=0.01,
        words=[],
    )

    class FakeModel:
        feature_extractor = SimpleNamespace(
            sampling_rate=16000,
            chunk_length=30,
        )

        def transcribe(self, audio, **kwargs):
            direct_calls.append((audio, kwargs))
            return iter([decoded_segment]), SimpleNamespace(language="en")

    class FakeBatchedModel:
        def transcribe(self, audio, **kwargs):
            batched_calls.append((audio, kwargs))
            raise AssertionError("Long clips must not use batched clip timestamps")

    profile = {
        "model": "large-v3",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 5,
        "batch_size": 8,
        "batch_size_max": 32,
    }
    result = _decode_clips_batched(
        audio_paths=[audio_path],
        project={"language": "en"},
        profile=profile,
        runtime={
            "model": FakeModel(),
            "batched_model": FakeBatchedModel(),
            "device": "cpu",
            "compute_type": "int8",
            "model_name": "large-v3",
        },
    )

    assert not batched_calls
    assert len(direct_calls) == 1
    assert result[0]["transcript"] == "Complete long clip"
    assert result[0]["segments"][0]["end"] == pytest.approx(30.1)


def test_automatic_batch_size_uses_free_gpu_memory_conservatively() -> None:
    batch_size, source, memory = _automatic_batch_size(
        device="cuda",
        model_name="large-v3",
        compute_type="float16",
        memory_info={
            "index": 0,
            "name": "Test GPU",
            "total_mib": 16384,
            "free_mib": 10000,
        },
    )
    assert batch_size == 16
    assert "10000 MiB free" in source
    assert memory and memory["name"] == "Test GPU"

    assert _automatic_batch_size(
        device="cuda",
        model_name="large-v3",
        compute_type="float16",
        maximum=8,
        memory_info={
            "index": 0,
            "name": "Test GPU",
            "total_mib": 16384,
            "free_mib": 10000,
        },
    )[0] == 8
    assert _automatic_batch_size(
        device="cpu",
        model_name="large-v3",
        compute_type="int8",
    )[0] == 1


def test_recording_asr_auto_sizes_cpu_and_batch_changes_reuse_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "source.wav"
    _write_tone(audio_path)
    line = {
        "line_id": "Actor::R2",
        "sheet": "Actor",
        "sheet_index": 0,
        "excel_row": 2,
        "line": "Hello.",
    }
    write_json(tmp_path / "source_lines.json", {"lines": [line]})
    write_json(
        tmp_path / "audio_inventory.json",
        {
            "files": [
                {
                    "path": str(audio_path.resolve()),
                    "sha256": "source-hash",
                }
            ]
        },
    )
    project = {
        "source_lines": "source_lines.json",
        "audio_inventory": "audio_inventory.json",
        "language": "en",
        "transcription": {
            "model": "large-v3",
            "device": "cpu",
            "compute_type": "int8",
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
    calls = []
    fake_segment = SimpleNamespace(
        id=1,
        start=0.0,
        end=0.5,
        text="Hello",
        avg_logprob=-0.1,
        no_speech_prob=0.01,
        words=[],
    )

    class FakePipeline:
        def __init__(self, model):
            self.model = model

        def transcribe(self, _audio, **kwargs):
            calls.append(kwargs)
            return iter([fake_segment]), SimpleNamespace(
                language="en",
                language_probability=1.0,
                duration=1.0,
                duration_after_vad=1.0,
            )

    import dialogue_pipeline.transcription as transcription_module
    import faster_whisper

    monkeypatch.setattr(
        transcription_module,
        "_make_model",
        lambda *_args, **_kwargs: (object(), "cpu", "int8"),
    )
    monkeypatch.setattr(
        faster_whisper,
        "BatchedInferencePipeline",
        FakePipeline,
    )
    outputs = transcribe_project(
        project_dir=tmp_path,
        project=project,
    )

    assert len(outputs) == 1
    assert calls[0]["batch_size"] == 1
    assert calls[0]["without_timestamps"] is False
    payload = read_json(outputs[0])
    assert payload["batched_inference"] is True
    assert payload["batch_size_requested"] == "auto"
    assert payload["batch_size"] == 1

    stable_cache_key = payload["cache_key"]
    project["transcription"]["batch_size"] = 8
    cached_outputs = transcribe_project(
        project_dir=tmp_path,
        project=project,
    )
    assert len(calls) == 1
    assert read_json(cached_outputs[0])["cache_key"] == stable_cache_key

    # A cache written by the previous schema is also reusable when only the
    # batch size changes.
    legacy_settings = dict(project["transcription"])
    legacy_settings.pop("batch_size")
    legacy_batch_size = 16
    payload.pop("cache_identity")
    payload.pop("batch_size_requested")
    payload.pop("batch_size_source")
    payload["batch_size"] = legacy_batch_size
    payload["cache_key"] = stable_hash(
        {
            "audio_sha256": "source-hash",
            "model": "large-v3",
            "device_request": "cpu",
            "compute_type_request": "int8",
            "language": "en",
            "batched_inference": True,
            "batch_size": legacy_batch_size,
            "settings": legacy_settings,
        }
    )
    write_json(outputs[0], payload)
    project["transcription"]["batch_size"] = 4
    transcribe_project(project_dir=tmp_path, project=project)
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


def test_candidate_span_asr_batches_unique_uncached_spans(
    tmp_path: Path,
) -> None:
    segments = []
    for index in range(3):
        segment_file = tmp_path / f"merged_{index}.wav"
        _write_tone(segment_file, duration_seconds=0.5)
        segments.append(
            {
                "segment_id": f"session__m{index + 1:05d}_{index + 2:05d}",
                "kind": "merged",
                "file": segment_file.name,
                "source_sha256": "source-hash",
                "start_sample": index * 24000,
                "end_sample": (index + 1) * 24000,
            }
        )
    project = {
        "language": "en",
        "transcription": {
            "model": "large-v3",
            "device": "cpu",
            "compute_type": "int8",
        },
        "segment_transcription": {
            "enabled": True,
            "batch_size": 2,
        },
        "segmentation": {"fade_ms": 5.0},
        "export": {
            "sample_rate": 48000,
            "channels": 1,
            "bits_per_sample": 16,
        },
    }
    batch_sizes = []

    class FakeBatchedModel:
        def transcribe(self, _audio, **kwargs):
            clips = kwargs["clip_timestamps"]
            batch_sizes.append(kwargs["batch_size"])
            decoded = []
            for index, clip in enumerate(clips):
                start = int(clip["start"] * 16000) / 16000
                decoded.append(
                    SimpleNamespace(
                        start=start + 0.05,
                        end=start + 0.35,
                        text=f"Line {index + 1}",
                        avg_logprob=-0.1,
                        no_speech_prob=0.01,
                        words=[
                            SimpleNamespace(
                                start=start + 0.05,
                                end=start + 0.25,
                                word=" Line",
                                probability=0.95,
                            )
                        ],
                    )
                )
            return iter(decoded), SimpleNamespace(language="en")

    runtime = {
        "model": SimpleNamespace(
            feature_extractor=SimpleNamespace(sampling_rate=16000)
        ),
        "batched_model": FakeBatchedModel(),
    }
    results = transcribe_candidate_spans(
        project_dir=tmp_path,
        project=project,
        segments=[segments[0], segments[1], segments[0], segments[2]],
        runtime=runtime,
    )
    project["segment_transcription"]["batch_size"] = 8
    cached = transcribe_candidate_spans(
        project_dir=tmp_path,
        project=project,
        segments=segments,
        runtime={},
    )

    assert batch_sizes == [2, 1]
    assert set(results) == {segment["segment_id"] for segment in segments}
    assert {
        segment_id: payload["cache_key"]
        for segment_id, payload in cached.items()
    } == {
        segment_id: payload["cache_key"]
        for segment_id, payload in results.items()
    }


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
    review_path = tmp_path / "line_review.json"
    review_data = build_line_review(
        source_lines=[line],
        candidates_by_line={line["line_id"]: [candidate]},
        unmatched_segments=[
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
                "technical_score": 80.0,
                "audible": True,
            }
        ],
    )
    save_line_review(review_path, review_data)
    loaded_review = load_line_review(review_path)
    assert loaded_review["lines"][0]["type"] == "normal"
    assert loaded_review["lines"][0]["status"] == "AUTO_OK"
    assert loaded_review["lines"][0]["suggested_segment_id"] == (
        "session__s00001"
    )
    assert loaded_review["lines"][0]["selected_segment_id"] == (
        "session__s00001"
    )
    assert loaded_review["lines"][0]["candidates"][0]["score"] == 100.0
    assert loaded_review["unmatched_segments"] == []

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
    review_path = tmp_path / "line_review.json"
    save_line_review(
        review_path,
        build_line_review(
            source_lines=lines,
            candidates_by_line=candidates,
            unmatched_segments=[],
        ),
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
            "candidate_min_score": 45.0,
            "reliable_min_score": 72.0,
            "reliable_min_margin": 8.0,
            "short_line_min_score": 88.0,
            "short_line_min_margin": 15.0,
            "noise_penalty": 2.2,
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
    assert outputs["review"].name == "line_review.json"
    assert "unmatched" not in outputs
    assert not stale_unmatched.exists()
    review = load_line_review(outputs["review"])
    assert review["lines"][0]["selected_segment_id"]
    assert review["lines"][1]["selected_segment_id"]
    assert review["lines"][0]["candidates"]
    assert "unmatched_segments" in review
