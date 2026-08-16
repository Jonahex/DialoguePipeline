from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .cancellation import check_processing_cancelled
from .audio import (
    acoustic_regions,
    cut_pcm_wav,
    detect_silences,
    pcm_voice_regions,
    prepare_pcm_segmentation_source,
    probe_audio,
    quietest_pcm_boundary,
    seconds_to_sample,
    transcript_for_region,
)
from .project import inventory_by_path
from .util import (
    normalize_spoken_text,
    read_json,
    relpath_for_config,
    resolve_project_path,
    stable_hash,
    write_json,
)


def split_regions_on_word_gaps(
    regions: list[dict[str, Any]],
    transcription: dict[str, Any],
    *,
    duration_seconds: float,
    settings: dict[str, Any],
    audio_path: Path | None = None,
    sample_rate: int | None = None,
) -> list[dict[str, Any]]:
    if not bool(settings.get("word_split_enabled", True)):
        return regions

    minimum_gap = float(settings.get("word_split_gap_seconds", 0.3))
    minimum_region = float(settings.get("word_split_min_region_seconds", 1.5))
    maximum_boundaries = max(
        0,
        int(settings.get("word_split_max_boundaries", 2)),
    )
    maximum_piece_seconds = max(
        minimum_region,
        float(settings.get("word_split_max_segment_seconds", 8.0)),
    )
    pre_padding = float(settings.get("pre_padding_seconds", 0.15))
    post_padding = float(settings.get("post_padding_seconds", 0.25))
    minimum_segment = float(settings.get("minimum_segment_seconds", 0.15))
    if maximum_boundaries == 0:
        return regions

    refined = []
    for region in regions:
        speech_start = float(region["speech_start"])
        speech_end = float(region["speech_end"])
        if speech_end - speech_start < minimum_region:
            refined.append(region)
            continue
        _, words, _ = transcript_for_region(
            transcription,
            speech_start,
            speech_end,
        )
        if len(words) < 2:
            refined.append(region)
            continue
        gap_candidates = []
        for left, right in zip(words, words[1:]):
            left_end = float(left.get("end", speech_start))
            right_start = float(right.get("start", speech_end))
            gap = right_start - left_end
            if gap >= minimum_gap:
                boundary = (left_end + right_start) / 2.0
                if (
                    audio_path is not None
                    and sample_rate is not None
                    and bool(settings.get("word_split_snap_enabled", True))
                ):
                    boundary = quietest_pcm_boundary(
                        audio_path,
                        proposed_sample=round(boundary * sample_rate),
                        minimum_sample=round(left_end * sample_rate),
                        maximum_sample=round(right_start * sample_rate),
                        search_seconds=min(
                            float(
                                settings.get(
                                    "word_split_snap_search_seconds",
                                    0.20,
                                )
                            ),
                            gap / 2.0,
                        ),
                        window_seconds=float(
                            settings.get(
                                "word_split_snap_window_seconds",
                                0.02,
                            )
                        ),
                        maximum_rms_dbfs=float(
                            settings.get(
                                "word_split_snap_max_rms_dbfs",
                                -42.0,
                            )
                        ),
                    ) / sample_rate
                gap_candidates.append((gap, boundary))
        ranked_gaps = sorted(gap_candidates, reverse=True)
        selected = {
            boundary for _, boundary in ranked_gaps[:maximum_boundaries]
        }
        # The configured boundary count remains the normal cap, but it must
        # not strand many takes inside one exceptionally long region. Add the
        # strongest remaining gap inside each oversized piece until no
        # splittable piece exceeds the target duration.
        while True:
            edges = [speech_start, *sorted(selected), speech_end]
            oversized = [
                (left, right)
                for left, right in zip(edges, edges[1:])
                if right - left > maximum_piece_seconds
            ]
            if not oversized:
                break
            added = False
            for left, right in oversized:
                extra = next(
                    (
                        boundary
                        for _gap, boundary in ranked_gaps
                        if boundary not in selected and left < boundary < right
                    ),
                    None,
                )
                if extra is not None:
                    selected.add(extra)
                    added = True
            if not added:
                break
        boundaries = sorted(selected)
        if not boundaries:
            refined.append(region)
            continue

        speech_boundaries = [speech_start, *boundaries, speech_end]
        pieces = []
        for left, right in zip(speech_boundaries, speech_boundaries[1:]):
            if right - left < minimum_segment:
                continue
            pieces.append(
                {
                    "speech_start": left,
                    "speech_end": right,
                    "start": max(0.0, left - pre_padding),
                    "end": min(duration_seconds, right + post_padding),
                    "split_source": "word_gap",
                }
            )
        if len(pieces) >= 2:
            refined.extend(pieces)
        else:
            refined.append(region)
    return refined


def prevent_region_overlaps(
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep padding without duplicating audio across adjacent base clips."""

    ordered = sorted(regions, key=lambda region: float(region["speech_start"]))
    for left, right in zip(ordered, ordered[1:]):
        if float(left["end"]) <= float(right["start"]):
            continue
        boundary = (
            float(left["speech_end"]) + float(right["speech_start"])
        ) / 2.0
        left["end"] = min(float(left["end"]), boundary)
        right["start"] = max(float(right["start"]), boundary)
    return [
        region
        for region in ordered
        if float(region["end"]) > float(region["start"])
    ]


def _segment_voice_bounds(
    audio_path: Path,
    *,
    start_sample: int,
    end_sample: int,
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(settings.get("enabled", True)):
        return None
    speech_regions = pcm_voice_regions(
        audio_path,
        start_sample=start_sample,
        end_sample=end_sample,
        threshold=float(settings.get("vad_threshold", 0.50)),
    )
    breath_regions = pcm_voice_regions(
        audio_path,
        start_sample=start_sample,
        end_sample=end_sample,
        threshold=float(settings.get("breath_vad_threshold", 0.70)),
    )
    return {
        "source": "silero_vad",
        "speech_regions": [
            {"start_sample": start, "end_sample": end}
            for start, end in speech_regions
        ],
        "strict_speech_regions": [
            {"start_sample": start, "end_sample": end}
            for start, end in breath_regions
        ],
        "speech": (
            {
                "start_sample": speech_regions[0][0],
                "end_sample": speech_regions[-1][1],
            }
            if speech_regions
            else None
        ),
        "strict_speech": (
            {
                "start_sample": breath_regions[0][0],
                "end_sample": breath_regions[-1][1],
            }
            if breath_regions
            else None
        ),
    }


def segment_project(
    *,
    project_dir: Path,
    project: dict[str, Any],
    session_filter: set[str] | None = None,
    force: bool = False,
) -> Path:
    settings = dict(project.get("segmentation") or {})
    export_settings = dict(project.get("export") or {})
    output_sample_rate = int(export_settings.get("sample_rate", 48000))
    output_channels = int(export_settings.get("channels", 1))
    output_bits = int(export_settings.get("bits_per_sample", 16))
    voice_boundary_settings = {
        "enabled": bool(
            settings.get("voice_boundary_detection_enabled", True)
        ),
        "vad_threshold": float(
            settings.get("voice_boundary_vad_threshold", 0.50)
        ),
        "breath_vad_threshold": float(
            settings.get("voice_boundary_breath_vad_threshold", 0.70)
        ),
    }
    for key, default in (
        ("word_split_snap_search_seconds", 0.20),
        ("word_split_snap_window_seconds", 0.02),
    ):
        if float(settings.get(key, default)) < 0.0:
            raise ValueError(f"segmentation.{key} cannot be negative")
    snap_maximum_rms = float(
        settings.get("word_split_snap_max_rms_dbfs", -42.0)
    )
    if not -120.0 <= snap_maximum_rms <= 0.0:
        raise ValueError(
            "segmentation.word_split_snap_max_rms_dbfs must be between "
            "-120 and 0"
        )
    if not 0.0 <= voice_boundary_settings["vad_threshold"] <= 1.0:
        raise ValueError(
            "segmentation.voice_boundary_vad_threshold must be between 0 and 1"
        )
    if not 0.0 <= voice_boundary_settings["breath_vad_threshold"] <= 1.0:
        raise ValueError(
            "segmentation.voice_boundary_breath_vad_threshold must be "
            "between 0 and 1"
        )
    if (
        voice_boundary_settings["breath_vad_threshold"]
        < voice_boundary_settings["vad_threshold"]
    ):
        raise ValueError(
            "segmentation.voice_boundary_breath_vad_threshold must be at "
            "least segmentation.voice_boundary_vad_threshold"
        )
    inventory = inventory_by_path(project_dir, project)
    transcript_dir = project_dir / "transcripts"
    segment_root = project_dir / "segments"
    manifest_path = project_dir / "segments_manifest.json"

    previous_manifest = read_json(manifest_path) if manifest_path.exists() else {}
    previous_by_session = {
        entry["session_id"]: entry
        for entry in previous_manifest.get("sessions", [])
    }
    session_outputs = []

    sessions = [
        session
        for session in project["sessions"]
        if session.get("enabled", True)
        and (not session_filter or session["id"] in session_filter)
    ]
    if not sessions:
        raise ValueError("No enabled sessions matched the requested filter.")

    for index, session in enumerate(sessions, start=1):
        check_processing_cancelled()
        transcript_path = transcript_dir / f"{session['id']}.json"
        if not transcript_path.is_file():
            raise FileNotFoundError(
                f"Missing transcript for {session['id']}: {transcript_path}. "
                "Run the transcribe command first."
            )
        transcription = read_json(transcript_path)
        audio_path = resolve_project_path(project_dir, session["audio"])
        inventory_item = inventory.get(audio_path.resolve())
        if not inventory_item:
            raise KeyError(f"Audio is not present in inventory: {audio_path}")
        cache_key = stable_hash(
            {
                "audio_sha256": inventory_item["sha256"],
                "transcription_cache_key": transcription.get("cache_key"),
                "settings": settings,
                "voice_boundary_detection": voice_boundary_settings,
                "export": export_settings,
            }
        )
        previous = previous_by_session.get(session["id"])
        if (
            not force
            and previous
            and previous.get("cache_key") == cache_key
            and all(
                resolve_project_path(project_dir, segment["file"]).is_file()
                for segment in previous.get("segments", [])
            )
            and (
                not previous.get("working_audio")
                or resolve_project_path(
                    project_dir, previous["working_audio"]
                ).is_file()
            )
        ):
            print(
                f"[segment {index}/{len(sessions)}] Cached: {session['id']}",
                flush=True,
            )
            session_outputs.append(previous)
            continue

        normalization_name = (
            f"{session['id']}__{inventory_item['sha256'][:16]}__"
            f"{output_sample_rate}hz_{output_channels}ch_s{output_bits}.wav"
        )
        working_audio, working_shape, normalized = prepare_pcm_segmentation_source(
            audio_path,
            project_dir / "normalized_sources" / normalization_name,
            sample_rate=output_sample_rate,
            channels=output_channels,
            bits_per_sample=output_bits,
        )
        check_processing_cancelled()
        if normalized:
            print(
                f"[segment {index}/{len(sessions)}] Using normalized PCM source: "
                f"{session['id']}",
                flush=True,
            )
        print(
            f"[segment {index}/{len(sessions)}] Detecting gaps: {session['id']}",
            flush=True,
        )
        silences = detect_silences(
            working_audio,
            noise_db=float(settings.get("silence_noise_db", -40.0)),
            minimum_duration_seconds=float(
                settings.get("silence_detection_min_seconds", 0.20)
            ),
        )
        check_processing_cancelled()
        sample_rate = working_shape["sample_rate"]
        source_frames = working_shape["frame_count"]
        duration_seconds = source_frames / sample_rate
        regions = acoustic_regions(
            duration_seconds,
            silences,
            split_gap_seconds=float(settings.get("split_gap_seconds", 0.20)),
            minimum_segment_seconds=float(
                settings.get("minimum_segment_seconds", 0.15)
            ),
            pre_padding_seconds=float(settings.get("pre_padding_seconds", 0.15)),
            post_padding_seconds=float(
                settings.get("post_padding_seconds", 0.25)
            ),
        )
        regions = split_regions_on_word_gaps(
            regions,
            transcription,
            duration_seconds=duration_seconds,
            settings=settings,
            audio_path=working_audio,
            sample_rate=sample_rate,
        )
        regions = prevent_region_overlaps(regions)
        session_dir = segment_root / session["id"]
        segment_records = []

        print(
            f"[segment {index}/{len(sessions)}] Writing {len(regions)} segments",
            flush=True,
        )
        for segment_index, region in enumerate(regions, start=1):
            check_processing_cancelled()
            segment_id = f"{session['id']}__s{segment_index:05d}"
            output_path = session_dir / f"{segment_id}.wav"
            start_sample = seconds_to_sample(region["start"], sample_rate)
            end_sample = seconds_to_sample(region["end"], sample_rate)
            start_sample = max(0, min(start_sample, source_frames))
            end_sample = max(start_sample, min(end_sample, source_frames))
            metrics = cut_pcm_wav(
                working_audio,
                output_path,
                start_sample=start_sample,
                end_sample=end_sample,
                fade_ms=float(settings.get("fade_ms", 5.0)),
            )
            voice_bounds = _segment_voice_bounds(
                working_audio,
                start_sample=start_sample,
                end_sample=end_sample,
                settings=voice_boundary_settings,
            )
            transcript, words, probability = transcript_for_region(
                transcription,
                start_sample / sample_rate,
                end_sample / sample_rate,
            )
            segment_records.append(
                {
                    "segment_id": segment_id,
                    "kind": "base",
                    "session_id": session["id"],
                    "source_audio": session["audio"],
                    "source_sha256": inventory_item["sha256"],
                    "base_indices": [segment_index - 1],
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_seconds": start_sample / sample_rate,
                    "end_seconds": end_sample / sample_rate,
                    "file": output_path.relative_to(project_dir).as_posix(),
                    "transcript": transcript,
                    "word_count": len(words),
                    "words": words,
                    "asr_probability": probability,
                    "split_source": region.get("split_source", "acoustic"),
                    "voice_bounds": voice_bounds,
                    "metrics": metrics,
                }
            )

        gap_durations = [
            silence["duration"]
            for silence in silences
            if silence["duration"] >= float(settings.get("split_gap_seconds", 0.20))
        ]
        session_outputs.append(
            {
                "session_id": session["id"],
                "cache_key": cache_key,
                "audio": session["audio"],
                "working_audio": relpath_for_config(working_audio, project_dir),
                "normalized_source": normalized,
                "source_sha256": inventory_item["sha256"],
                "sample_rate": sample_rate,
                "source_frames": source_frames,
                "silence_count": len(silences),
                "split_gap_count": len(gap_durations),
                "median_split_gap_seconds": (
                    statistics.median(gap_durations) if gap_durations else None
                ),
                "voice_boundary_detection": voice_boundary_settings,
                "segments": segment_records,
                "derived_segments": [],
            }
        )

    if session_filter:
        untouched = [
            value
            for key, value in previous_by_session.items()
            if key not in {session["id"] for session in sessions}
        ]
        session_outputs.extend(untouched)
    session_outputs.sort(key=lambda entry: entry["session_id"])
    manifest = {
        "schema_version": 1,
        "settings": settings,
        "sessions": session_outputs,
    }
    check_processing_cancelled()
    write_json(manifest_path, manifest)
    return manifest_path


def refresh_project_audio(
    *,
    project_dir: Path,
    project: dict[str, Any],
) -> dict[str, Any]:
    """Re-cut stored segment spans after source audio-only edits.

    The caller is asserting that the spoken content and its timing did not
    change. Existing transcripts, segment boundaries, alignment results, and
    review selections are therefore preserved. Updated sources must normalize
    to the exact sample rate and frame count recorded by the manifest.
    """

    project_dir = project_dir.resolve()
    manifest_path = project_dir / "segments_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing segment manifest: {manifest_path}. Run segment first."
        )
    inventory_path = resolve_project_path(
        project_dir,
        project["audio_inventory"],
    )
    previous_inventory = (
        read_json(inventory_path) if inventory_path.is_file() else {"files": []}
    )
    previous_inventory_by_path = {
        Path(item["path"]).resolve(): item
        for item in previous_inventory.get("files", [])
    }

    configured_sessions = project.get("sessions") or []
    session_by_id = {
        str(session["id"]): session for session in configured_sessions
    }
    if len(session_by_id) != len(configured_sessions):
        raise ValueError("Project contains duplicate session IDs.")

    audio_paths = {
        resolve_project_path(project_dir, str(session["audio"])).resolve()
        for session in configured_sessions
    }
    configured_audio_dir = project.get("audio_dir")
    if configured_audio_dir:
        audio_dir = resolve_project_path(project_dir, str(configured_audio_dir))
        if audio_dir.is_dir():
            audio_paths.update(path.resolve() for path in audio_dir.glob("*.wav"))
    if not audio_paths:
        raise ValueError("Project has no configured source recordings.")

    ordered_audio_paths = sorted(
        audio_paths,
        key=lambda path: (path.name.casefold(), str(path).casefold()),
    )
    refreshed_inventory = []
    for index, audio_path in enumerate(ordered_audio_paths, start=1):
        check_processing_cancelled()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Source recording not found: {audio_path}")
        print(
            f"[refresh inventory {index}/{len(ordered_audio_paths)}] "
            f"{audio_path.name}",
            flush=True,
        )
        refreshed_inventory.append(probe_audio(audio_path, include_hash=True))
    refreshed_inventory_by_path = {
        Path(item["path"]).resolve(): item for item in refreshed_inventory
    }

    manifest = read_json(manifest_path)
    manifest_sessions = manifest.get("sessions")
    if not isinstance(manifest_sessions, list) or not manifest_sessions:
        raise ValueError("Segment manifest has no sessions to refresh.")

    export_settings = dict(project.get("export") or {})
    segmentation_settings = dict(
        manifest.get("settings") or project.get("segmentation") or {}
    )
    fade_ms = float(segmentation_settings.get("fade_ms", 5.0))
    prepared_sessions: dict[str, dict[str, Any]] = {}
    total_segment_count = 0

    # Prepare and validate every source before overwriting any segment file.
    for session_entry in manifest_sessions:
        check_processing_cancelled()
        session_id = str(session_entry.get("session_id") or "")
        session = session_by_id.get(session_id)
        if session is None:
            raise KeyError(
                f"Manifest session is not configured in project.json: {session_id}"
            )
        audio_path = resolve_project_path(
            project_dir,
            str(session["audio"]),
        ).resolve()
        inventory_item = refreshed_inventory_by_path.get(audio_path)
        if inventory_item is None:
            raise KeyError(f"Audio is not present in refreshed inventory: {audio_path}")

        base_segments = session_entry.get("segments") or []
        derived_segments = session_entry.get("derived_segments") or []
        all_segments = [*base_segments, *derived_segments]
        total_segment_count += len(all_segments)

        sample_rate = int(
            session_entry.get(
                "sample_rate",
                export_settings.get("sample_rate", 48000),
            )
        )
        first_metrics = (
            dict((all_segments[0].get("metrics") or {}))
            if all_segments
            else {}
        )
        channels = int(
            first_metrics.get(
                "channels",
                export_settings.get("channels", 1),
            )
        )
        bits_per_sample = int(
            first_metrics.get(
                "bits_per_sample",
                export_settings.get("bits_per_sample", 16),
            )
        )
        expected_frames = int(session_entry.get("source_frames", -1))
        if sample_rate <= 0 or channels <= 0 or bits_per_sample <= 0:
            raise ValueError(
                f"Invalid stored audio format for session {session_id}."
            )
        if expected_frames < 0:
            raise ValueError(
                f"Session {session_id} has no stored source frame count; "
                "run segmentation once before refreshing audio."
            )

        normalization_name = (
            f"{session_id}__{inventory_item['sha256'][:16]}__"
            f"{sample_rate}hz_{channels}ch_s{bits_per_sample}.wav"
        )
        working_audio, working_shape, normalized = (
            prepare_pcm_segmentation_source(
                audio_path,
                project_dir / "normalized_sources" / normalization_name,
                sample_rate=sample_rate,
                channels=channels,
                bits_per_sample=bits_per_sample,
            )
        )
        actual_shape = {
            "sample_rate": int(working_shape["sample_rate"]),
            "channels": int(working_shape["channels"]),
            "bits_per_sample": int(working_shape["bits_per_sample"]),
        }
        expected_shape = {
            "sample_rate": sample_rate,
            "channels": channels,
            "bits_per_sample": bits_per_sample,
        }
        if actual_shape != expected_shape:
            raise ValueError(
                f"Updated source for {session_id} normalizes to "
                f"{actual_shape!r}, expected {expected_shape!r}."
            )
        actual_frames = int(working_shape["frame_count"])
        if actual_frames != expected_frames:
            raise ValueError(
                f"Updated source duration changed for {session_id}: "
                f"{actual_frames} frames instead of {expected_frames}. "
                "This refresh is only safe when spoken timing is unchanged."
            )

        for segment in all_segments:
            segment_id = str(segment.get("segment_id") or "")
            if not segment_id or not segment.get("file"):
                raise ValueError(
                    f"Session {session_id} contains a segment without an ID or file."
                )
            try:
                start_sample = int(segment["start_sample"])
                end_sample = int(segment["end_sample"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Segment {segment_id} has invalid stored sample bounds."
                ) from error
            if not 0 <= start_sample < end_sample <= actual_frames:
                raise ValueError(
                    f"Segment {segment_id} bounds {start_sample}..{end_sample} "
                    f"are outside the updated {actual_frames}-frame source."
                )

        stored_voice_settings = session_entry.get("voice_boundary_detection")
        voice_settings = (
            dict(stored_voice_settings)
            if isinstance(stored_voice_settings, dict)
            else {
                "enabled": bool(
                    segmentation_settings.get(
                        "voice_boundary_detection_enabled",
                        True,
                    )
                ),
                "vad_threshold": float(
                    segmentation_settings.get(
                        "voice_boundary_vad_threshold",
                        0.50,
                    )
                ),
                "breath_vad_threshold": float(
                    segmentation_settings.get(
                        "voice_boundary_breath_vad_threshold",
                        0.70,
                    )
                ),
            }
        )
        prepared_sessions[session_id] = {
            "session": session,
            "inventory": inventory_item,
            "working_audio": working_audio,
            "working_shape": working_shape,
            "normalized": normalized,
            "voice_settings": voice_settings,
        }

    refreshed_segments = 0
    refreshed_derived_segments = 0
    for session_index, session_entry in enumerate(manifest_sessions, start=1):
        check_processing_cancelled()
        session_id = str(session_entry["session_id"])
        prepared = prepared_sessions[session_id]
        session = prepared["session"]
        inventory_item = prepared["inventory"]
        working_audio = prepared["working_audio"]
        working_shape = prepared["working_shape"]
        sample_rate = int(working_shape["sample_rate"])
        base_segments = session_entry.get("segments") or []
        derived_segments = session_entry.get("derived_segments") or []
        all_segments = [*base_segments, *derived_segments]
        base_segment_objects = {id(segment) for segment in base_segments}
        print(
            f"[refresh segments {session_index}/{len(manifest_sessions)}] "
            f"{session_id}: {len(all_segments)} file(s)",
            flush=True,
        )

        for segment in all_segments:
            check_processing_cancelled()
            start_sample = int(segment["start_sample"])
            end_sample = int(segment["end_sample"])
            output_path = resolve_project_path(project_dir, str(segment["file"]))
            segment["metrics"] = cut_pcm_wav(
                working_audio,
                output_path,
                start_sample=start_sample,
                end_sample=end_sample,
                fade_ms=fade_ms,
            )
            segment["session_id"] = session_id
            segment["source_audio"] = session["audio"]
            segment["source_sha256"] = inventory_item["sha256"]
            segment["start_seconds"] = start_sample / sample_rate
            segment["end_seconds"] = end_sample / sample_rate
            if id(segment) in base_segment_objects:
                segment["voice_bounds"] = _segment_voice_bounds(
                    working_audio,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    settings=prepared["voice_settings"],
                )

        refreshed_segments += len(base_segments)
        refreshed_derived_segments += len(derived_segments)
        session_entry["audio"] = session["audio"]
        session_entry["working_audio"] = relpath_for_config(
            working_audio,
            project_dir,
        )
        session_entry["normalized_source"] = bool(prepared["normalized"])
        session_entry["source_sha256"] = inventory_item["sha256"]
        session_entry["sample_rate"] = sample_rate
        session_entry["source_frames"] = int(working_shape["frame_count"])
        session_entry["voice_boundary_detection"] = prepared["voice_settings"]

    check_processing_cancelled()
    write_json(inventory_path, {"files": refreshed_inventory})
    write_json(manifest_path, manifest)

    changed_source_count = sum(
        1
        for path, item in refreshed_inventory_by_path.items()
        if (previous_inventory_by_path.get(path) or {}).get("sha256")
        != item.get("sha256")
    )
    return {
        "audio_inventory": inventory_path,
        "segments_manifest": manifest_path,
        "source_count": len(refreshed_inventory),
        "changed_source_count": changed_source_count,
        "segment_count": refreshed_segments,
        "derived_segment_count": refreshed_derived_segments,
        "total_segment_count": total_segment_count,
    }


def materialize_derived_segment(
    *,
    project_dir: Path,
    project: dict[str, Any],
    session_entry: dict[str, Any],
    base_segments: list[dict[str, Any]],
    start_index: int,
    count: int,
) -> dict[str, Any]:
    if count == 1:
        return base_segments[start_index]

    selected = base_segments[start_index : start_index + count]
    end_index = start_index + count - 1
    segment_id = (
        f"{session_entry['session_id']}__m{start_index + 1:05d}_{end_index + 1:05d}"
    )
    existing = next(
        (
            segment
            for segment in session_entry.get("derived_segments", [])
            if segment["segment_id"] == segment_id
        ),
        None,
    )
    if existing and resolve_project_path(project_dir, existing["file"]).is_file():
        return existing

    audio_path = resolve_project_path(
        project_dir,
        session_entry.get("working_audio", session_entry["audio"]),
    )
    output_path = (
        project_dir
        / "segments"
        / session_entry["session_id"]
        / f"{segment_id}.wav"
    )
    start_sample = selected[0]["start_sample"]
    end_sample = selected[-1]["end_sample"]
    metrics = cut_pcm_wav(
        audio_path,
        output_path,
        start_sample=start_sample,
        end_sample=end_sample,
        fade_ms=float(project["segmentation"].get("fade_ms", 5.0)),
    )
    probabilities = [
        segment["asr_probability"]
        for segment in selected
        if segment.get("asr_probability") is not None
    ]
    derived = {
        "segment_id": segment_id,
        "kind": "merged",
        "session_id": session_entry["session_id"],
        "source_audio": session_entry["audio"],
        "source_sha256": session_entry["source_sha256"],
        "base_indices": list(range(start_index, start_index + count)),
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_seconds": start_sample / session_entry["sample_rate"],
        "end_seconds": end_sample / session_entry["sample_rate"],
        "file": output_path.relative_to(project_dir).as_posix(),
        "transcript": " ".join(
            segment.get("transcript", "").strip()
            for segment in selected
            if segment.get("transcript", "").strip()
        ),
        "word_count": sum(int(segment.get("word_count", 0)) for segment in selected),
        "asr_probability": (
            sum(probabilities) / len(probabilities) if probabilities else None
        ),
        "metrics": metrics,
    }
    session_entry.setdefault("derived_segments", []).append(derived)
    return derived


def materialize_trimmed_segment(
    *,
    project_dir: Path,
    project: dict[str, Any],
    session_entry: dict[str, Any],
    base_segments: list[dict[str, Any]],
    base_index: int,
    count: int = 1,
    start_sample: int,
    end_sample: int,
    transcript: str,
    asr_probability: float | None,
) -> dict[str, Any]:
    """Materialize sample-bounded audio across contiguous base segments."""

    selected = base_segments[base_index : base_index + count]
    if len(selected) != count or count < 1:
        raise ValueError(
            f"Invalid bounded segment range: {base_index} + {count}"
        )
    base_start = int(selected[0]["start_sample"])
    base_end = int(selected[-1]["end_sample"])
    start_sample = max(base_start, min(int(start_sample), base_end))
    end_sample = max(start_sample, min(int(end_sample), base_end))
    if start_sample == base_start and end_sample == base_end:
        return materialize_derived_segment(
            project_dir=project_dir,
            project=project,
            session_entry=session_entry,
            base_segments=base_segments,
            start_index=base_index,
            count=count,
        )
    if end_sample <= start_sample:
        raise ValueError(
            f"Invalid trimmed segment bounds: {start_sample}..{end_sample}"
        )

    if count == 1:
        segment_id = (
            f"{session_entry['session_id']}__t{base_index + 1:05d}_"
            f"{start_sample:010d}_{end_sample:010d}"
        )
    else:
        end_index = base_index + count - 1
        segment_id = (
            f"{session_entry['session_id']}__b{base_index + 1:05d}_"
            f"{end_index + 1:05d}_{start_sample:010d}_{end_sample:010d}"
        )
    existing = next(
        (
            segment
            for segment in session_entry.get("derived_segments", [])
            if segment["segment_id"] == segment_id
        ),
        None,
    )
    if existing and resolve_project_path(project_dir, existing["file"]).is_file():
        return existing

    audio_path = resolve_project_path(
        project_dir,
        session_entry.get("working_audio", session_entry["audio"]),
    )
    output_path = (
        project_dir
        / "segments"
        / session_entry["session_id"]
        / f"{segment_id}.wav"
    )
    metrics = cut_pcm_wav(
        audio_path,
        output_path,
        start_sample=start_sample,
        end_sample=end_sample,
        fade_ms=float(
            (project.get("segmentation") or {}).get("fade_ms", 5.0)
        ),
    )
    sample_rate = int(session_entry["sample_rate"])
    probabilities = [
        float(segment["asr_probability"])
        for segment in selected
        if segment.get("asr_probability") is not None
    ]
    derived = {
        "segment_id": segment_id,
        "kind": "trimmed" if count == 1 else "bounded",
        "session_id": session_entry["session_id"],
        "source_audio": session_entry["audio"],
        "source_sha256": session_entry["source_sha256"],
        "base_indices": list(range(base_index, base_index + count)),
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_seconds": start_sample / sample_rate,
        "end_seconds": end_sample / sample_rate,
        "file": output_path.relative_to(project_dir).as_posix(),
        "transcript": transcript,
        "word_count": len(normalize_spoken_text(transcript).split()),
        "asr_probability": (
            asr_probability
            if asr_probability is not None
            else (
                sum(probabilities) / len(probabilities)
                if probabilities
                else None
            )
        ),
        "transcript_source": (
            "segment_asr_word_trim"
            if count == 1
            else "segment_asr_trimmed_edge_span"
        ),
        "metrics": metrics,
    }
    session_entry.setdefault("derived_segments", []).append(derived)
    return derived


def materialize_manual_segment(
    *,
    project_dir: Path,
    project: dict[str, Any],
    session_entry: dict[str, Any],
    source_segment: dict[str, Any],
    line_id: str,
    start_sample: int,
    end_sample: int,
    transcript: str,
) -> dict[str, Any]:
    """Copy a review candidate with user-selected sample boundaries."""

    sample_rate = int(session_entry["sample_rate"])
    source_frames = int(session_entry["source_frames"])
    start_sample = max(0, min(int(start_sample), source_frames))
    end_sample = max(start_sample, min(int(end_sample), source_frames))
    if end_sample <= start_sample:
        raise ValueError("An edited segment must contain at least one sample.")

    edit_key = stable_hash(
        {
            "line_id": line_id,
            "source_segment_id": str(source_segment["segment_id"]),
            "start_sample": start_sample,
            "end_sample": end_sample,
        }
    )
    segment_id = f"{session_entry['session_id']}__e{edit_key[:12]}"
    existing = next(
        (
            segment
            for segment in session_entry.get("derived_segments", [])
            if str(segment.get("segment_id")) == segment_id
        ),
        None,
    )
    if existing is not None:
        line_ids = existing.setdefault("manual_line_ids", [])
        if line_id not in line_ids:
            line_ids.append(line_id)
            line_ids.sort()
        if resolve_project_path(project_dir, str(existing["file"])).is_file():
            return existing

    base_segments = session_entry.get("segments") or []
    base_indices = [
        index
        for index, segment in enumerate(base_segments)
        if (
            start_sample < int(segment["end_sample"])
            and end_sample > int(segment["start_sample"])
        )
    ]
    working_audio = resolve_project_path(
        project_dir,
        str(session_entry.get("working_audio") or session_entry["audio"]),
    )
    output_path = (
        project_dir
        / "segments"
        / str(session_entry["session_id"])
        / f"{segment_id}.wav"
    )
    metrics = cut_pcm_wav(
        working_audio,
        output_path,
        start_sample=start_sample,
        end_sample=end_sample,
        fade_ms=float(
            (project.get("segmentation") or {}).get("fade_ms", 5.0)
        ),
    )
    manual_line_ids = sorted(
        {
            line_id,
            *(
                str(value)
                for value in (
                    existing.get("manual_line_ids", [])
                    if existing is not None
                    else []
                )
            ),
        }
    )
    manual_segment = {
        "segment_id": segment_id,
        "kind": "manual_edit",
        "session_id": str(session_entry["session_id"]),
        "source_audio": str(
            source_segment.get("source_audio") or session_entry["audio"]
        ),
        "source_sha256": str(
            source_segment.get("source_sha256")
            or session_entry.get("source_sha256")
            or ""
        ),
        "base_indices": base_indices,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_seconds": start_sample / sample_rate,
        "end_seconds": end_sample / sample_rate,
        "file": output_path.relative_to(project_dir).as_posix(),
        "transcript": transcript,
        "word_count": len(normalize_spoken_text(transcript).split()),
        "asr_probability": source_segment.get("asr_probability"),
        "transcript_source": "manual_copy_edit",
        "edited_from_segment_id": str(source_segment["segment_id"]),
        "manual_line_ids": manual_line_ids,
        "metrics": metrics,
    }
    if existing is None:
        session_entry.setdefault("derived_segments", []).append(manual_segment)
    else:
        existing.clear()
        existing.update(manual_segment)
    return manual_segment
