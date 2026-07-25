from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .project import inventory_by_path, load_source_data
from .util import (
    resolve_model_cache_root,
    resolve_project_path,
    stable_hash,
    write_json,
)
from .workbook_io import lines_for_session


def _session_hotwords(lines: list[dict[str, Any]], max_characters: int = 1200) -> str:
    common = {
        "I",
        "The",
        "A",
        "An",
        "And",
        "But",
        "You",
        "Your",
        "He",
        "She",
        "It",
        "We",
        "They",
        "What",
        "Where",
        "When",
        "Why",
        "How",
    }
    words = []
    seen = set()
    for line in lines:
        for word in re.findall(r"\b[A-Z][A-Za-z'-]{2,}\b", line["line"]):
            if word in common or word.lower() in seen:
                continue
            seen.add(word.lower())
            words.append(word)
    output = " ".join(words)
    return output[:max_characters]


def _make_model(
    model_name: str,
    *,
    device: str,
    compute_type: str,
    download_root: Path,
):
    from faster_whisper import WhisperModel

    if device != "auto":
        resolved_compute = compute_type
        if resolved_compute == "auto":
            resolved_compute = "float16" if device == "cuda" else "int8"
        return (
            WhisperModel(
                model_name,
                device=device,
                compute_type=resolved_compute,
                download_root=str(download_root),
            ),
            device,
            resolved_compute,
        )

    try:
        model = WhisperModel(
            model_name,
            device="cuda",
            compute_type="float16" if compute_type == "auto" else compute_type,
            download_root=str(download_root),
        )
        return model, "cuda", "float16" if compute_type == "auto" else compute_type
    except Exception as error:
        print(
            f"CUDA model initialization failed ({error}). Falling back to CPU INT8.",
            flush=True,
        )
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8" if compute_type == "auto" else compute_type,
            download_root=str(download_root),
        )
        return model, "cpu", "int8" if compute_type == "auto" else compute_type


def transcribe_project(
    *,
    project_dir: Path,
    project: dict[str, Any],
    session_filter: set[str] | None = None,
    force: bool = False,
    model_override: str | None = None,
    device_override: str | None = None,
) -> list[Path]:
    source_data = load_source_data(project_dir, project)
    inventory = inventory_by_path(project_dir, project)
    settings = dict(project.get("transcription") or {})
    model_name = model_override or settings.get("model", "large-v3")
    device = device_override or settings.get("device", "auto")
    compute_type = settings.get("compute_type", "auto")
    model_root = resolve_model_cache_root(project_dir, settings)
    model_root.mkdir(parents=True, exist_ok=True)
    transcript_dir = project_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    sessions = [
        session
        for session in project["sessions"]
        if session.get("enabled", True)
        and (not session_filter or session["id"] in session_filter)
    ]
    if not sessions:
        raise ValueError("No enabled sessions matched the requested filter.")

    model = None
    resolved_device = device
    resolved_compute = compute_type
    written = []

    for index, session in enumerate(sessions, start=1):
        audio_path = resolve_project_path(project_dir, session["audio"])
        inventory_item = inventory.get(audio_path.resolve())
        if not inventory_item:
            raise KeyError(f"Audio is not present in inventory: {audio_path}")
        session_lines = lines_for_session(source_data, session)
        if not session_lines:
            print(
                f"[transcribe {index}/{len(sessions)}] Skipping {session['id']}: "
                "no mapped script lines",
                flush=True,
            )
            continue

        cache_key_data = {
            "audio_sha256": inventory_item["sha256"],
            "model": model_name,
            "device_request": device,
            "compute_type_request": compute_type,
            "language": project.get("language", "en"),
            "settings": settings,
        }
        cache_key = stable_hash(cache_key_data)
        output_path = transcript_dir / f"{session['id']}.json"
        if output_path.exists() and not force:
            from .util import read_json

            existing = read_json(output_path)
            if existing.get("cache_key") == cache_key:
                print(
                    f"[transcribe {index}/{len(sessions)}] Cached: {session['id']}",
                    flush=True,
                )
                written.append(output_path)
                continue

        if model is None:
            print(f"[model cache] {model_root}", flush=True)
            model, resolved_device, resolved_compute = _make_model(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=model_root,
            )

        print(
            f"[transcribe {index}/{len(sessions)}] {session['id']} "
            f"({resolved_device}/{resolved_compute})",
            flush=True,
        )
        kwargs = {
            "language": project.get("language", "en"),
            "beam_size": int(settings.get("beam_size", 5)),
            "word_timestamps": True,
            "condition_on_previous_text": bool(
                settings.get("condition_on_previous_text", False)
            ),
            "vad_filter": bool(settings.get("vad_filter", True)),
            "vad_parameters": {
                "min_silence_duration_ms": int(
                    settings.get("vad_min_silence_ms", 500)
                )
            },
        }
        hotwords = _session_hotwords(session_lines)
        if hotwords:
            kwargs["hotwords"] = hotwords

        try:
            segments_iter, info = model.transcribe(str(audio_path), **kwargs)
            segment_records = []
            for segment in segments_iter:
                words = [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                    for word in (segment.words or [])
                ]
                segment_records.append(
                    {
                        "id": segment.id,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "avg_logprob": segment.avg_logprob,
                        "no_speech_prob": segment.no_speech_prob,
                        "words": words,
                    }
                )
        except Exception as error:
            if resolved_device != "cuda" or device != "auto":
                raise
            print(
                f"CUDA transcription failed ({error}). Retrying on CPU INT8.",
                flush=True,
            )
            model, resolved_device, resolved_compute = _make_model(
                model_name,
                device="cpu",
                compute_type="int8",
                download_root=model_root,
            )
            segments_iter, info = model.transcribe(str(audio_path), **kwargs)
            segment_records = [
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "words": [
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability,
                        }
                        for word in (segment.words or [])
                    ],
                }
                for segment in segments_iter
            ]

        payload = {
            "schema_version": 1,
            "session_id": session["id"],
            "audio": session["audio"],
            "audio_sha256": inventory_item["sha256"],
            "cache_key": cache_key,
            "model": model_name,
            "device": resolved_device,
            "compute_type": resolved_compute,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_seconds": info.duration,
            "duration_after_vad_seconds": info.duration_after_vad,
            "segments": segment_records,
        }
        write_json(output_path, payload)
        written.append(output_path)
    return written
