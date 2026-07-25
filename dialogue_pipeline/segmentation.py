from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .audio import (
    acoustic_regions,
    cut_pcm_wav,
    detect_silences,
    prepare_pcm_segmentation_source,
    seconds_to_sample,
    transcript_for_region,
)
from .project import inventory_by_path
from .util import (
    read_json,
    relpath_for_config,
    resolve_project_path,
    stable_hash,
    write_json,
)


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
            noise_db=float(settings.get("silence_noise_db", -45.0)),
            minimum_duration_seconds=float(
                settings.get("silence_detection_min_seconds", 0.35)
            ),
        )
        sample_rate = working_shape["sample_rate"]
        source_frames = working_shape["frame_count"]
        duration_seconds = source_frames / sample_rate
        regions = acoustic_regions(
            duration_seconds,
            silences,
            split_gap_seconds=float(settings.get("split_gap_seconds", 0.8)),
            minimum_segment_seconds=float(
                settings.get("minimum_segment_seconds", 0.15)
            ),
            pre_padding_seconds=float(settings.get("pre_padding_seconds", 0.15)),
            post_padding_seconds=float(
                settings.get("post_padding_seconds", 0.25)
            ),
        )
        session_dir = segment_root / session["id"]
        segment_records = []

        print(
            f"[segment {index}/{len(sessions)}] Writing {len(regions)} segments",
            flush=True,
        )
        for segment_index, region in enumerate(regions, start=1):
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
                    "asr_probability": probability,
                    "metrics": metrics,
                }
            )

        gap_durations = [
            silence["duration"]
            for silence in silences
            if silence["duration"] >= float(settings.get("split_gap_seconds", 0.8))
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
    write_json(manifest_path, manifest)
    return manifest_path


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
