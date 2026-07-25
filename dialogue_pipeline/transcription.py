from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from .project import inventory_by_path, load_source_data
from .util import (
    is_vocalization_script,
    normalize_text,
    read_json,
    resolve_model_cache_root,
    resolve_project_path,
    stable_hash,
    word_count,
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


def _segment_transcription_profile(
    project: dict[str, Any],
    *,
    model_override: str | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    source_settings = dict(project.get("transcription") or {})
    settings = dict(project.get("segment_transcription") or {})
    return {
        **settings,
        "model": (
            model_override
            or settings.get("model")
            or source_settings.get("model")
            or "large-v3"
        ),
        "device": (
            device_override
            or settings.get("device")
            or source_settings.get("device")
            or "auto"
        ),
        "compute_type": (
            settings.get("compute_type")
            or source_settings.get("compute_type")
            or "auto"
        ),
        "beam_size": int(
            settings.get(
                "beam_size",
                source_settings.get("beam_size", 5),
            )
        ),
        "condition_on_previous_text": False,
        "vad_filter": False,
    }


def _ensure_clip_model(
    *,
    project_dir: Path,
    project: dict[str, Any],
    profile: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    identity = (
        str(profile["model"]),
        str(profile["device"]),
        str(profile["compute_type"]),
    )
    if runtime.get("model") and runtime.get("requested_identity") is None:
        runtime.setdefault("model_name", str(profile["model"]))
        runtime.setdefault("device", str(profile["device"]))
        runtime.setdefault("compute_type", str(profile["compute_type"]))
        runtime["requested_identity"] = identity
        return
    if runtime.get("requested_identity") == identity and runtime.get("model"):
        return

    source_settings = dict(project.get("transcription") or {})
    model_root = resolve_model_cache_root(project_dir, source_settings)
    model_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[segment ASR] Loading {profile['model']} from {model_root}",
        flush=True,
    )
    model, resolved_device, resolved_compute = _make_model(
        str(profile["model"]),
        device=str(profile["device"]),
        compute_type=str(profile["compute_type"]),
        download_root=model_root,
    )
    runtime.update(
        {
            "model": model,
            "device": resolved_device,
            "compute_type": resolved_compute,
            "model_name": str(profile["model"]),
            "requested_identity": identity,
            "model_root": model_root,
        }
    )


def _decode_clip(
    *,
    audio_path: Path,
    project: dict[str, Any],
    profile: dict[str, Any],
    runtime: dict[str, Any],
    prompt: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "language": project.get("language", "en"),
        "beam_size": int(profile.get("beam_size", 5)),
        "word_timestamps": True,
        # A clip must be decoded without context leaking from another take.
        "condition_on_previous_text": False,
        # The segmenter already found the speech region. Whisper VAD commonly
        # removes very short clips such as "Yes?" and "No!".
        "vad_filter": False,
    }
    if prompt:
        kwargs["initial_prompt"] = prompt
        kwargs["hotwords"] = prompt

    def run_decode() -> tuple[list[Any], Any]:
        segments_iter, info = runtime["model"].transcribe(
            str(audio_path),
            **kwargs,
        )
        return list(segments_iter), info

    try:
        decoded, info = run_decode()
    except Exception as error:
        if (
            runtime.get("device") != "cuda"
            or str(profile.get("device", "auto")) != "auto"
        ):
            raise
        print(
            f"[segment ASR] CUDA inference failed ({error}); "
            "continuing on CPU INT8.",
            flush=True,
        )
        model, resolved_device, resolved_compute = _make_model(
            str(profile["model"]),
            device="cpu",
            compute_type="int8",
            download_root=Path(runtime["model_root"]),
        )
        runtime.update(
            {
                "model": model,
                "device": resolved_device,
                "compute_type": resolved_compute,
            }
        )
        decoded, info = run_decode()

    transcript_parts: list[str] = []
    words: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    probabilities: list[float] = []
    for decoded_segment in decoded:
        text = str(getattr(decoded_segment, "text", "") or "").strip()
        if text:
            transcript_parts.append(text)
        decoded_words = []
        for decoded_word in getattr(decoded_segment, "words", None) or []:
            probability = getattr(decoded_word, "probability", None)
            word_record = {
                "start": getattr(decoded_word, "start", None),
                "end": getattr(decoded_word, "end", None),
                "word": str(getattr(decoded_word, "word", "") or ""),
                "probability": probability,
            }
            decoded_words.append(word_record)
            words.append(word_record)
            if probability is not None:
                probabilities.append(float(probability))
        segment_records.append(
            {
                "start": getattr(decoded_segment, "start", None),
                "end": getattr(decoded_segment, "end", None),
                "text": text,
                "avg_logprob": getattr(decoded_segment, "avg_logprob", None),
                "no_speech_prob": getattr(
                    decoded_segment,
                    "no_speech_prob",
                    None,
                ),
                "words": decoded_words,
            }
        )
    transcript = " ".join(transcript_parts).strip()
    return {
        "transcript": transcript,
        "word_count": word_count(transcript),
        "words": words,
        "asr_probability": (
            sum(probabilities) / len(probabilities)
            if probabilities
            else None
        ),
        "segments": segment_records,
        "language": getattr(info, "language", project.get("language", "en")),
        "language_probability": getattr(info, "language_probability", None),
        "model": runtime.get("model_name"),
        "device": runtime.get("device"),
        "compute_type": runtime.get("compute_type"),
        "prompted": bool(prompt),
    }


def transcribe_clip_cached(
    *,
    project_dir: Path,
    project: dict[str, Any],
    audio_path: Path,
    cache_path: Path,
    cache_identity: dict[str, Any],
    profile: dict[str, Any],
    runtime: dict[str, Any],
    prompt: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    decoding_identity = {
        "model": profile["model"],
        "device_request": profile["device"],
        "compute_type_request": profile["compute_type"],
        "language": project.get("language", "en"),
        "beam_size": int(profile.get("beam_size", 5)),
        "condition_on_previous_text": False,
        "vad_filter": False,
        "prompt": prompt or "",
    }
    cache_key = stable_hash(
        {
            **cache_identity,
            "decoding": decoding_identity,
        }
    )
    if cache_path.is_file() and not force:
        cached = read_json(cache_path)
        if cached.get("cache_key") == cache_key:
            return cached

    _ensure_clip_model(
        project_dir=project_dir,
        project=project,
        profile=profile,
        runtime=runtime,
    )
    decoded = _decode_clip(
        audio_path=audio_path,
        project=project,
        profile=profile,
        runtime=runtime,
        prompt=prompt,
    )
    payload = {
        "schema_version": 1,
        "cache_key": cache_key,
        "cache_identity": cache_identity,
        "decoding": decoding_identity,
        **decoded,
    }
    write_json(cache_path, payload)
    return payload


def _prompt_candidates(
    lines: list[dict[str, Any]],
    transcript: str,
    *,
    maximum_words: int,
    top_k: int,
) -> list[dict[str, Any]]:
    eligible = [
        line
        for line in lines
        if (
            not is_vocalization_script(line["line"])
            and 0 < word_count(line["line"]) <= maximum_words
        )
    ]
    if not transcript:
        return eligible[:top_k]
    return sorted(
        eligible,
        key=lambda line: fuzz.WRatio(
            normalize_text(line["line"]),
            normalize_text(transcript),
        ),
        reverse=True,
    )[:top_k]


def _best_ordered_script_score(
    lines: list[dict[str, Any]],
    transcript: str,
) -> float:
    observed = normalize_text(transcript)
    if not observed:
        return 0.0
    return max(
        (
            fuzz.ratio(normalize_text(line["line"]), observed)
            for line in lines
            if normalize_text(line["line"])
        ),
        default=0.0,
    )


def _apply_segment_asr_result(
    segment: dict[str, Any],
    *,
    primary: dict[str, Any],
    prompted: dict[str, Any] | None,
) -> None:
    if "session_transcript" not in segment:
        segment["session_transcript"] = str(segment.get("transcript") or "")
        segment["session_words"] = list(segment.get("words") or [])
        segment["session_asr_probability"] = segment.get("asr_probability")

    primary_text = str(primary.get("transcript") or "").strip()
    prompted_text = str((prompted or {}).get("transcript") or "").strip()
    session_text = str(segment.get("session_transcript") or "").strip()
    if primary_text:
        chosen = primary
        source = "segment_asr"
    elif prompted_text:
        chosen = prompted or {}
        source = "segment_asr_prompted"
    else:
        chosen = {
            "transcript": session_text,
            "word_count": word_count(session_text),
            "words": segment.get("session_words") or [],
            "asr_probability": segment.get("session_asr_probability"),
        }
        source = "session_asr_fallback"

    segment["transcript"] = str(chosen.get("transcript") or "").strip()
    segment["word_count"] = int(
        chosen.get("word_count")
        if chosen.get("word_count") is not None
        else word_count(segment["transcript"])
    )
    segment["words"] = list(chosen.get("words") or [])
    segment["asr_probability"] = chosen.get("asr_probability")
    segment["transcript_source"] = source
    segment["segment_asr"] = {
        "primary": primary,
        "prompted_fallback": prompted,
        "canonical_source": source,
    }


def transcribe_segments_project(
    *,
    project_dir: Path,
    project: dict[str, Any],
    session_filter: set[str] | None = None,
    segment_filter: set[str] | None = None,
    force: bool = False,
    model_override: str | None = None,
    device_override: str | None = None,
    runtime: dict[str, Any] | None = None,
) -> Path:
    """Independently transcribe every base clip and update the manifest."""
    manifest_path = project_dir / "segments_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing segment manifest: {manifest_path}. Run segment first."
        )
    settings = dict(project.get("segment_transcription") or {})
    if not bool(settings.get("enabled", False)):
        return manifest_path

    manifest = read_json(manifest_path)
    source_data = load_source_data(project_dir, project)
    session_config = {
        item["id"]: item
        for item in project["sessions"]
        if item.get("enabled", True)
        and (not session_filter or item["id"] in session_filter)
    }
    if not session_config:
        raise ValueError("No enabled sessions matched the requested filter.")

    profile = _segment_transcription_profile(
        project,
        model_override=model_override,
        device_override=device_override,
    )
    runtime = runtime if runtime is not None else {}
    cache_root = project_dir / "segment_transcripts" / "base"
    prompt_enabled = bool(settings.get("prompt_fallback_enabled", True))
    prompt_max_duration = float(
        settings.get("prompt_fallback_max_segment_seconds", 6.0)
    )
    prompt_max_words = int(settings.get("prompt_fallback_max_script_words", 8))
    prompt_top_k = max(1, int(settings.get("prompt_fallback_top_k", 8)))
    prompt_max_characters = max(
        100,
        int(settings.get("prompt_fallback_max_characters", 800)),
    )
    trigger_probability = float(
        settings.get("prompt_fallback_trigger_probability", 0.55)
    )
    trigger_ordered = float(
        settings.get("prompt_fallback_trigger_ordered_score", 70.0)
    )

    processed = 0
    cached = 0
    prompted_count = 0
    for session_entry in manifest.get("sessions", []):
        session = session_config.get(session_entry["session_id"])
        if not session:
            continue
        lines = lines_for_session(source_data, session)
        base_segments = [
            segment
            for segment in session_entry.get("segments", [])
            if (
                not segment_filter
                or segment["segment_id"] in segment_filter
            )
        ]
        print(
            f"[segment ASR] {session['id']}: "
            f"{len(base_segments)} independent clips",
            flush=True,
        )
        for index, segment in enumerate(base_segments, start=1):
            audio_path = resolve_project_path(project_dir, segment["file"])
            cache_identity = {
                "kind": "base_segment",
                "segment_id": segment["segment_id"],
                "source_sha256": segment.get("source_sha256"),
                "start_sample": segment.get("start_sample"),
                "end_sample": segment.get("end_sample"),
                "fade_ms": (project.get("segmentation") or {}).get("fade_ms"),
                "export": project.get("export") or {},
            }
            cache_path = cache_root / f"{segment['segment_id']}.json"
            previous_key = (
                (segment.get("segment_asr") or {})
                .get("primary", {})
                .get("cache_key")
            )
            primary = transcribe_clip_cached(
                project_dir=project_dir,
                project=project,
                audio_path=audio_path,
                cache_path=cache_path,
                cache_identity=cache_identity,
                profile=profile,
                runtime=runtime,
                force=force,
            )
            if previous_key and previous_key == primary.get("cache_key") and not force:
                cached += 1

            transcript = str(primary.get("transcript") or "").strip()
            probability = primary.get("asr_probability")
            duration = float(
                (segment.get("metrics") or {}).get("duration_seconds")
                or (
                    float(segment.get("end_seconds", 0.0))
                    - float(segment.get("start_seconds", 0.0))
                )
            )
            ordered_score = _best_ordered_script_score(lines, transcript)
            needs_prompt = bool(
                prompt_enabled
                and duration <= prompt_max_duration
                and (
                    not transcript
                    or probability is None
                    or float(probability) < trigger_probability
                    or ordered_score < trigger_ordered
                )
            )
            prompted = None
            if needs_prompt:
                candidates = _prompt_candidates(
                    lines,
                    transcript,
                    maximum_words=prompt_max_words,
                    top_k=prompt_top_k,
                )
                prompt = " | ".join(line["line"] for line in candidates)[
                    :prompt_max_characters
                ]
                if prompt:
                    prompted_count += 1
                    prompted = transcribe_clip_cached(
                        project_dir=project_dir,
                        project=project,
                        audio_path=audio_path,
                        cache_path=(
                            cache_root
                            / f"{segment['segment_id']}__prompted.json"
                        ),
                        cache_identity={
                            **cache_identity,
                            "kind": "base_segment_prompted_fallback",
                            "candidate_line_ids": [
                                line["line_id"] for line in candidates
                            ],
                        },
                        profile=profile,
                        runtime=runtime,
                        prompt=prompt,
                        force=force,
                    )
            _apply_segment_asr_result(
                segment,
                primary=primary,
                prompted=prompted,
            )
            processed += 1
            if index % 25 == 0 or index == len(base_segments):
                print(
                    f"[segment ASR] {session['id']}: "
                    f"{index}/{len(base_segments)}",
                    flush=True,
                )

    manifest["segment_transcription"] = {
        "enabled": True,
        "model": profile["model"],
        "device": runtime.get("device", profile["device"]),
        "compute_type": runtime.get("compute_type", profile["compute_type"]),
        "processed_segment_count": processed,
        "manifest_cache_hits": cached,
        "prompted_fallback_count": prompted_count,
    }
    if segment_filter and processed == 0:
        raise ValueError("No base segments matched the requested filter.")
    write_json(manifest_path, manifest)
    return manifest_path


def transcribe_candidate_span(
    *,
    project_dir: Path,
    project: dict[str, Any],
    segment: dict[str, Any],
    runtime: dict[str, Any],
    force: bool = False,
    model_override: str | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Transcribe the exact candidate WAV without script prompting."""
    profile = _segment_transcription_profile(
        project,
        model_override=model_override,
        device_override=device_override,
    )
    if segment.get("kind") == "base":
        primary = (segment.get("segment_asr") or {}).get("primary")
        if primary:
            return primary

    cache_identity = {
        "kind": "candidate_span",
        "segment_id": segment["segment_id"],
        "source_sha256": segment.get("source_sha256"),
        "start_sample": segment.get("start_sample"),
        "end_sample": segment.get("end_sample"),
        "fade_ms": (project.get("segmentation") or {}).get("fade_ms"),
        "export": project.get("export") or {},
    }
    result = transcribe_clip_cached(
        project_dir=project_dir,
        project=project,
        audio_path=resolve_project_path(project_dir, segment["file"]),
        cache_path=(
            project_dir
            / "segment_transcripts"
            / "candidates"
            / f"{segment['segment_id']}.json"
        ),
        cache_identity=cache_identity,
        profile=profile,
        runtime=runtime,
        force=force,
    )
    segment["candidate_asr"] = result
    return result


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
