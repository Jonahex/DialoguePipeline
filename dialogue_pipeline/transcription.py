from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from .cancellation import check_processing_cancelled
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


_AUTOMATIC_BATCH_STEPS = (1, 2, 4, 8, 12, 16, 24, 32)


def _batch_size_request(value: Any) -> int | str:
    """Normalize a configured batch size to a positive integer or ``auto``."""
    if value is None:
        return "auto"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized or normalized == "auto":
            return "auto"
        value = normalized
    if isinstance(value, bool):
        raise ValueError("batch_size must be 'auto' or a positive integer")
    try:
        batch_size = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "batch_size must be 'auto' or a positive integer"
        ) from error
    if batch_size < 1:
        raise ValueError("batch_size must be 'auto' or a positive integer")
    return batch_size


def _gpu_memory_info() -> dict[str, Any] | None:
    """Return memory telemetry for the CUDA device used by default."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None

    first_line = next(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()),
        "",
    )
    fields = [field.strip() for field in first_line.split(",", 3)]
    if len(fields) != 4:
        return None
    try:
        return {
            "index": int(fields[0]),
            "name": fields[1],
            "total_mib": int(fields[2]),
            "free_mib": int(fields[3]),
        }
    except ValueError:
        return None


def _automatic_batch_size(
    *,
    device: str,
    model_name: str,
    compute_type: str,
    maximum: int = 32,
    memory_info: dict[str, Any] | None = None,
) -> tuple[int, str, dict[str, Any] | None]:
    """Choose a conservative inference batch from currently free GPU memory."""
    maximum = max(1, int(maximum))
    if str(device).lower() != "cuda":
        return 1, "automatic CPU fallback", None

    memory_info = memory_info if memory_info is not None else _gpu_memory_info()
    if not memory_info:
        return 1, "automatic fallback (GPU memory telemetry unavailable)", None

    total_mib = max(0, int(memory_info.get("total_mib", 0)))
    free_mib = max(0, int(memory_info.get("free_mib", 0)))
    # The model has already been loaded when this is called, so memory.free
    # reflects its resident weights. Keep headroom for CUDA workspaces, audio
    # features, and other applications before estimating per-item capacity.
    reserve_mib = max(1024, math.ceil(total_mib * 0.12))

    normalized_model = str(model_name).lower()
    if "tiny" in normalized_model:
        per_item_mib = 140
    elif "base" in normalized_model:
        per_item_mib = 180
    elif "small" in normalized_model:
        per_item_mib = 240
    elif "medium" in normalized_model:
        per_item_mib = 320
    elif "turbo" in normalized_model or "distil" in normalized_model:
        per_item_mib = 340
    else:
        per_item_mib = 400

    normalized_compute = str(compute_type).lower()
    if normalized_compute == "float32":
        per_item_mib = math.ceil(per_item_mib * 1.7)
    elif normalized_compute.startswith("int8"):
        per_item_mib = math.ceil(per_item_mib * 0.75)

    estimated_capacity = max(
        1,
        (free_mib - reserve_mib) // max(1, per_item_mib),
    )
    upper_bound = min(maximum, estimated_capacity)
    choices = [
        size
        for size in _AUTOMATIC_BATCH_STEPS
        if size <= upper_bound
    ]
    batch_size = choices[-1] if choices else 1
    reason = (
        f"automatic from {free_mib} MiB free/{total_mib} MiB total "
        f"on {memory_info.get('name') or 'CUDA GPU'}"
    )
    return batch_size, reason, memory_info


def _resolve_profile_batch_size(
    *,
    profile: dict[str, Any],
    runtime: dict[str, Any],
) -> int:
    request = _batch_size_request(profile.get("batch_size", "auto"))
    maximum = max(1, int(profile.get("batch_size_max", 32)))
    identity = (
        request,
        maximum,
        str(profile.get("model") or runtime.get("model_name") or ""),
        str(runtime.get("device") or profile.get("device") or ""),
        str(runtime.get("compute_type") or profile.get("compute_type") or ""),
    )
    if runtime.get("batch_size_identity") == identity:
        return int(runtime["batch_size"])

    if isinstance(request, int):
        batch_size = request
        source = "configured"
        memory_info = None
    else:
        batch_size, source, memory_info = _automatic_batch_size(
            device=str(runtime.get("device") or profile.get("device") or ""),
            model_name=str(
                profile.get("model") or runtime.get("model_name") or ""
            ),
            compute_type=str(
                runtime.get("compute_type")
                or profile.get("compute_type")
                or ""
            ),
            maximum=maximum,
        )

    runtime.update(
        {
            "batch_size": batch_size,
            "batch_size_request": request,
            "batch_size_source": source,
            "batch_size_memory": memory_info,
            "batch_size_identity": identity,
        }
    )
    profile["resolved_batch_size"] = batch_size
    return batch_size


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
        "batch_size": _batch_size_request(
            settings.get(
                "batch_size",
                source_settings.get("batch_size", "auto"),
            )
        ),
        "batch_size_max": max(
            1,
            int(
                settings.get(
                    "batch_size_max",
                    source_settings.get("batch_size_max", 32),
                )
            ),
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
    check_processing_cancelled()
    model, resolved_device, resolved_compute = _make_model(
        str(profile["model"]),
        device=str(profile["device"]),
        compute_type=str(profile["compute_type"]),
        download_root=model_root,
    )
    check_processing_cancelled()
    from faster_whisper import BatchedInferencePipeline

    runtime.update(
        {
            "model": model,
            "batched_model": BatchedInferencePipeline(model=model),
            "device": resolved_device,
            "compute_type": resolved_compute,
            "model_name": str(profile["model"]),
            "requested_identity": identity,
            "model_root": model_root,
        }
    )


def _decoded_segments_payload(
    decoded: list[Any],
    *,
    info: Any,
    project: dict[str, Any],
    runtime: dict[str, Any],
    prompted: bool,
    time_offset: float = 0.0,
) -> dict[str, Any]:
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
            start = getattr(decoded_word, "start", None)
            end = getattr(decoded_word, "end", None)
            word_record = {
                "start": (
                    max(0.0, float(start) - time_offset)
                    if start is not None
                    else None
                ),
                "end": (
                    max(0.0, float(end) - time_offset)
                    if end is not None
                    else None
                ),
                "word": str(getattr(decoded_word, "word", "") or ""),
                "probability": probability,
            }
            decoded_words.append(word_record)
            words.append(word_record)
            if probability is not None:
                probabilities.append(float(probability))
        segment_start = getattr(decoded_segment, "start", None)
        segment_end = getattr(decoded_segment, "end", None)
        segment_records.append(
            {
                "start": (
                    max(0.0, float(segment_start) - time_offset)
                    if segment_start is not None
                    else None
                ),
                "end": (
                    max(0.0, float(segment_end) - time_offset)
                    if segment_end is not None
                    else None
                ),
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
        "prompted": prompted,
    }


def _fall_back_clip_runtime_to_cpu(
    *,
    profile: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    check_processing_cancelled()
    model, resolved_device, resolved_compute = _make_model(
        str(profile["model"]),
        device="cpu",
        compute_type="int8",
        download_root=Path(runtime["model_root"]),
    )
    check_processing_cancelled()
    from faster_whisper import BatchedInferencePipeline

    runtime.update(
        {
            "model": model,
            "batched_model": BatchedInferencePipeline(model=model),
            "device": resolved_device,
            "compute_type": resolved_compute,
        }
    )
    _resolve_profile_batch_size(profile=profile, runtime=runtime)


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
        decoded = []
        for segment in segments_iter:
            check_processing_cancelled()
            decoded.append(segment)
        return decoded, info

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
        _fall_back_clip_runtime_to_cpu(
            profile=profile,
            runtime=runtime,
        )
        decoded, info = run_decode()

    return _decoded_segments_payload(
        decoded,
        info=info,
        project=project,
        runtime=runtime,
        prompted=bool(prompt),
    )


def _clip_decoding_identity(
    *,
    project: dict[str, Any],
    profile: dict[str, Any],
    prompt: str | None,
) -> dict[str, Any]:
    return {
        "model": profile["model"],
        "device_request": profile["device"],
        "compute_type_request": profile["compute_type"],
        "language": project.get("language", "en"),
        "beam_size": int(profile.get("beam_size", 5)),
        "condition_on_previous_text": False,
        "vad_filter": False,
        "prompt": prompt or "",
    }


def _clip_cache_key(
    *,
    cache_identity: dict[str, Any],
    decoding_identity: dict[str, Any],
) -> str:
    return stable_hash(
        {
            **cache_identity,
            "decoding": decoding_identity,
        }
    )


def _cached_clip_payload(
    *,
    cache_path: Path,
    cache_key: str,
    cache_identity: dict[str, Any],
    decoding_identity: dict[str, Any],
    force: bool,
) -> dict[str, Any] | None:
    if not cache_path.is_file() or force:
        return None
    cached = read_json(cache_path)
    if cached.get("cache_key") == cache_key:
        return cached

    # Batch size changes only how clips are grouped for inference; it does not
    # change the requested decoding of an individual clip. Accept caches from
    # versions that recorded this execution detail in the decoding identity.
    cached_decoding = dict(cached.get("decoding") or {})
    cached_decoding.pop("batch_size", None)
    normalized_decoding = dict(decoding_identity)
    normalized_decoding.pop("batch_size", None)
    if (
        cached.get("cache_identity") == cache_identity
        and cached_decoding == normalized_decoding
    ):
        return cached
    return None


def _clip_payload(
    *,
    cache_key: str,
    cache_identity: dict[str, Any],
    decoding_identity: dict[str, Any],
    decoded: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cache_key": cache_key,
        "cache_identity": cache_identity,
        "decoding": decoding_identity,
        **decoded,
    }


def _decode_clips_batched(
    *,
    audio_paths: list[Path],
    project: dict[str, Any],
    profile: dict[str, Any],
    runtime: dict[str, Any],
    prompt: str | None = None,
) -> list[dict[str, Any]]:
    if not audio_paths:
        return []
    # Injected test/custom runtimes can expose only WhisperModel.transcribe.
    # The production runtime always has BatchedInferencePipeline.
    if runtime.get("batched_model") is None:
        return [
            _decode_clip(
                audio_path=audio_path,
                project=project,
                profile=profile,
                runtime=runtime,
                prompt=prompt,
            )
            for audio_path in audio_paths
        ]

    from faster_whisper.audio import decode_audio

    sampling_rate = int(runtime["model"].feature_extractor.sampling_rate)
    waveforms = []
    for audio_path in audio_paths:
        check_processing_cancelled()
        waveforms.append(
            decode_audio(str(audio_path), sampling_rate=sampling_rate)
        )
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for waveform in waveforms:
        start = cursor
        cursor += int(waveform.shape[0])
        offsets.append((start, cursor))
    combined = (
        np.concatenate(waveforms)
        if waveforms
        else np.empty((0,), dtype=np.float32)
    )
    clip_timestamps = [
        {
            "start": start / sampling_rate,
            "end": end / sampling_rate,
        }
        for start, end in offsets
    ]
    kwargs: dict[str, Any] = {
        "language": project.get("language", "en"),
        "beam_size": int(profile.get("beam_size", 5)),
        "batch_size": min(
            len(audio_paths),
            _resolve_profile_batch_size(profile=profile, runtime=runtime),
        ),
        "word_timestamps": True,
        "without_timestamps": False,
        "condition_on_previous_text": False,
        "vad_filter": False,
        "clip_timestamps": clip_timestamps,
    }
    if prompt:
        kwargs["initial_prompt"] = prompt
        kwargs["hotwords"] = prompt

    def run_decode() -> tuple[list[Any], Any]:
        segments_iter, info = runtime["batched_model"].transcribe(
            combined,
            **kwargs,
        )
        decoded = []
        for segment in segments_iter:
            check_processing_cancelled()
            decoded.append(segment)
        return decoded, info

    try:
        decoded, info = run_decode()
    except Exception as error:
        if (
            runtime.get("device") != "cuda"
            or str(profile.get("device", "auto")) != "auto"
        ):
            raise
        print(
            f"[segment ASR] CUDA batched inference failed ({error}); "
            "continuing on CPU INT8.",
            flush=True,
        )
        _fall_back_clip_runtime_to_cpu(
            profile=profile,
            runtime=runtime,
        )
        kwargs["batch_size"] = min(
            len(audio_paths),
            _resolve_profile_batch_size(profile=profile, runtime=runtime),
        )
        decoded, info = run_decode()

    decoded_by_clip: list[list[Any]] = [[] for _ in audio_paths]
    offset_seconds = [
        (start / sampling_rate, end / sampling_rate)
        for start, end in offsets
    ]
    for decoded_segment in decoded:
        start = float(getattr(decoded_segment, "start", 0.0) or 0.0)
        end = float(getattr(decoded_segment, "end", start) or start)
        midpoint = (start + end) / 2.0
        clip_index = next(
            (
                index
                for index, (clip_start, clip_end) in enumerate(offset_seconds)
                if (
                    clip_start <= midpoint < clip_end
                    or (
                        index == len(offset_seconds) - 1
                        and midpoint == clip_end
                    )
                )
            ),
            None,
        )
        if clip_index is not None:
            decoded_by_clip[clip_index].append(decoded_segment)

    return [
        _decoded_segments_payload(
            clip_decoded,
            info=info,
            project=project,
            runtime=runtime,
            prompted=bool(prompt),
            time_offset=offset_seconds[index][0],
        )
        for index, clip_decoded in enumerate(decoded_by_clip)
    ]


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
    decoding_identity = _clip_decoding_identity(
        project=project,
        profile=profile,
        prompt=prompt,
    )
    cache_key = _clip_cache_key(
        cache_identity=cache_identity,
        decoding_identity=decoding_identity,
    )
    cached = _cached_clip_payload(
        cache_path=cache_path,
        cache_key=cache_key,
        cache_identity=cache_identity,
        decoding_identity=decoding_identity,
        force=force,
    )
    if cached:
        return cached

    _ensure_clip_model(
        project_dir=project_dir,
        project=project,
        profile=profile,
        runtime=runtime,
    )
    decoded = _decode_clips_batched(
        audio_paths=[audio_path],
        project=project,
        profile=profile,
        runtime=runtime,
        prompt=prompt,
    )[0]
    payload = _clip_payload(
        cache_key=cache_key,
        cache_identity=cache_identity,
        decoding_identity=decoding_identity,
        decoded=decoded,
    )
    write_json(cache_path, payload)
    return payload


def transcribe_candidate_spans(
    *,
    project_dir: Path,
    project: dict[str, Any],
    segments: list[dict[str, Any]],
    runtime: dict[str, Any],
    force: bool = False,
    model_override: str | None = None,
    device_override: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Transcribe unique exact candidate WAVs in cache-aware batches."""

    profile = _segment_transcription_profile(
        project,
        model_override=model_override,
        device_override=device_override,
    )
    results: dict[str, dict[str, Any]] = {}
    pending = []
    seen = set()
    cache_dir = project_dir / "segment_transcripts" / "candidates"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        check_processing_cancelled()
        segment_id = str(segment["segment_id"])
        if segment_id in seen:
            continue
        seen.add(segment_id)
        if segment.get("kind") == "base":
            primary = (segment.get("segment_asr") or {}).get("primary")
            if primary:
                results[segment_id] = primary
                continue

        cache_identity = {
            "kind": "candidate_span",
            "segment_id": segment_id,
            "source_sha256": segment.get("source_sha256"),
            "start_sample": segment.get("start_sample"),
            "end_sample": segment.get("end_sample"),
            "fade_ms": (project.get("segmentation") or {}).get("fade_ms"),
            "export": project.get("export") or {},
        }
        decoding_identity = _clip_decoding_identity(
            project=project,
            profile=profile,
            prompt=None,
        )
        cache_key = _clip_cache_key(
            cache_identity=cache_identity,
            decoding_identity=decoding_identity,
        )
        cache_path = cache_dir / f"{segment_id}.json"
        cached = _cached_clip_payload(
            cache_path=cache_path,
            cache_key=cache_key,
            cache_identity=cache_identity,
            decoding_identity=decoding_identity,
            force=force,
        )
        if cached:
            segment["candidate_asr"] = cached
            results[segment_id] = cached
            continue
        pending.append(
            {
                "segment": segment,
                "audio_path": resolve_project_path(
                    project_dir,
                    segment["file"],
                ),
                "cache_path": cache_path,
                "cache_identity": cache_identity,
                "decoding_identity": decoding_identity,
                "cache_key": cache_key,
            }
        )

    if not pending:
        return results

    _ensure_clip_model(
        project_dir=project_dir,
        project=project,
        profile=profile,
        runtime=runtime,
    )
    batch_size = _resolve_profile_batch_size(profile=profile, runtime=runtime)
    print(
        f"[candidate ASR] {len(results)} cached, {len(pending)} pending "
        f"(batch size {batch_size}, {runtime['batch_size_source']})",
        flush=True,
    )
    for batch_start in range(0, len(pending), batch_size):
        check_processing_cancelled()
        batch = pending[batch_start : batch_start + batch_size]
        print(
            f"[candidate ASR] batch "
            f"{batch_start // batch_size + 1}/"
            f"{math.ceil(len(pending) / batch_size)}",
            flush=True,
        )
        try:
            decoded_batch = _decode_clips_batched(
                audio_paths=[entry["audio_path"] for entry in batch],
                project=project,
                profile=profile,
                runtime=runtime,
            )
            decoded_or_errors = list(decoded_batch)
        except Exception as batch_error:
            decoded_or_errors = []
            for entry in batch:
                check_processing_cancelled()
                try:
                    decoded_or_errors.append(
                        _decode_clips_batched(
                            audio_paths=[entry["audio_path"]],
                            project=project,
                            profile=profile,
                            runtime=runtime,
                        )[0]
                    )
                except Exception as error:
                    decoded_or_errors.append(
                        {
                            "error": str(error),
                            "batch_error": str(batch_error),
                        }
                    )

        for entry, decoded in zip(batch, decoded_or_errors):
            check_processing_cancelled()
            segment = entry["segment"]
            segment_id = str(segment["segment_id"])
            if decoded.get("error"):
                results[segment_id] = decoded
                continue
            payload = _clip_payload(
                cache_key=entry["cache_key"],
                cache_identity=entry["cache_identity"],
                decoding_identity=entry["decoding_identity"],
                decoded=decoded,
            )
            write_json(entry["cache_path"], payload)
            segment["candidate_asr"] = payload
            results[segment_id] = payload
    return results


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
        check_processing_cancelled()
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
        if segment_filter and not base_segments:
            continue
        print(
            f"[segment ASR] {session['id']}: "
            f"{len(base_segments)} independent clips "
            f"(batch size request {profile['batch_size']})",
            flush=True,
        )
        primary_entries: list[dict[str, Any]] = []
        pending_entries: list[dict[str, Any]] = []
        for segment in base_segments:
            check_processing_cancelled()
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
            decoding_identity = _clip_decoding_identity(
                project=project,
                profile=profile,
                prompt=None,
            )
            cache_key = _clip_cache_key(
                cache_identity=cache_identity,
                decoding_identity=decoding_identity,
            )
            primary = _cached_clip_payload(
                cache_path=cache_path,
                cache_key=cache_key,
                cache_identity=cache_identity,
                decoding_identity=decoding_identity,
                force=force,
            )
            entry = {
                "segment": segment,
                "audio_path": audio_path,
                "cache_path": cache_path,
                "cache_identity": cache_identity,
                "decoding_identity": decoding_identity,
                "cache_key": cache_key,
                "primary": primary,
            }
            primary_entries.append(entry)
            if primary:
                cached += 1
            else:
                pending_entries.append(entry)

        if pending_entries:
            _ensure_clip_model(
                project_dir=project_dir,
                project=project,
                profile=profile,
                runtime=runtime,
            )
            batch_size = _resolve_profile_batch_size(
                profile=profile,
                runtime=runtime,
            )
            print(
                f"[segment ASR] {session['id']}: "
                f"{len(pending_entries)} pending "
                f"(batch size {batch_size}, {runtime['batch_size_source']})",
                flush=True,
            )
            for batch_start in range(0, len(pending_entries), batch_size):
                check_processing_cancelled()
                batch = pending_entries[batch_start : batch_start + batch_size]
                decoded_batch = _decode_clips_batched(
                    audio_paths=[entry["audio_path"] for entry in batch],
                    project=project,
                    profile=profile,
                    runtime=runtime,
                )
                for entry, decoded in zip(batch, decoded_batch):
                    check_processing_cancelled()
                    primary = _clip_payload(
                        cache_key=entry["cache_key"],
                        cache_identity=entry["cache_identity"],
                        decoding_identity=entry["decoding_identity"],
                        decoded=decoded,
                    )
                    write_json(entry["cache_path"], primary)
                    entry["primary"] = primary

        for index, entry in enumerate(primary_entries, start=1):
            check_processing_cancelled()
            segment = entry["segment"]
            audio_path = entry["audio_path"]
            cache_identity = entry["cache_identity"]
            primary = entry["primary"]
            if primary is None:
                raise RuntimeError(
                    f"Missing batched transcript for {segment['segment_id']}"
                )

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
        "batch_size_requested": profile["batch_size"],
        "batch_size": runtime.get("batch_size", profile["batch_size"]),
        "batch_size_source": runtime.get("batch_size_source"),
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
    result = transcribe_candidate_spans(
        project_dir=project_dir,
        project=project,
        segments=[segment],
        runtime=runtime,
        force=force,
        model_override=model_override,
        device_override=device_override,
    )
    payload = result[str(segment["segment_id"])]
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def _recording_cache_identity(
    *,
    audio_sha256: str,
    project: dict[str, Any],
    settings: dict[str, Any],
    model_name: str,
    device: str,
    compute_type: str,
    hotwords: str,
) -> dict[str, Any]:
    """Describe only inputs that can change an individual ASR result."""
    return {
        "kind": "recording_transcript",
        "audio_sha256": audio_sha256,
        "decoding": {
            "model": model_name,
            "device_request": device,
            "compute_type_request": compute_type,
            "language": project.get("language", "en"),
            "beam_size": int(settings.get("beam_size", 5)),
            "word_timestamps": True,
            "without_timestamps": False,
            "condition_on_previous_text": bool(
                settings.get("condition_on_previous_text", False)
            ),
            "vad_filter": bool(settings.get("vad_filter", True)),
            "vad_min_silence_ms": int(
                settings.get("vad_min_silence_ms", 500)
            ),
            "hotwords": hotwords,
        },
    }


def _recording_cache_matches(
    *,
    existing: dict[str, Any],
    cache_key: str,
    cache_identity: dict[str, Any],
    settings: dict[str, Any],
    audio_sha256: str,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
) -> bool:
    if existing.get("cache_key") == cache_key:
        return True
    if existing.get("cache_identity") == cache_identity:
        return True

    # Compatibility with schema-v1 recording caches, whose hash included
    # batch_size twice (directly and inside the complete settings object).
    # Reconstruct both possible old settings shapes so changing only this
    # execution knob does not discard an already valid transcript.
    legacy_batch_size = existing.get("batch_size")
    try:
        legacy_batch_size = int(legacy_batch_size)
    except (TypeError, ValueError):
        return False

    legacy_settings_variants = []
    for retain_maximum in (True, False):
        legacy_settings_with_batch = dict(settings)
        legacy_settings_with_batch["batch_size"] = legacy_batch_size
        if not retain_maximum:
            legacy_settings_with_batch.pop("batch_size_max", None)
        legacy_settings_without_batch = dict(legacy_settings_with_batch)
        legacy_settings_without_batch.pop("batch_size", None)
        legacy_settings_variants.extend(
            (legacy_settings_with_batch, legacy_settings_without_batch)
        )
    for legacy_settings in legacy_settings_variants:
        legacy_key = stable_hash(
            {
                "audio_sha256": audio_sha256,
                "model": model_name,
                "device_request": device,
                "compute_type_request": compute_type,
                "language": language,
                "batched_inference": True,
                "batch_size": legacy_batch_size,
                "settings": legacy_settings,
            }
        )
        if existing.get("cache_key") == legacy_key:
            return True
    return False


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
    batch_request = _batch_size_request(settings.get("batch_size", "auto"))
    batch_profile = {
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "batch_size": batch_request,
        "batch_size_max": max(1, int(settings.get("batch_size_max", 32))),
    }
    batch_runtime: dict[str, Any] = {}
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
    batched_model = None
    resolved_device = device
    resolved_compute = compute_type
    batch_size: int | None = None
    written = []

    for index, session in enumerate(sessions, start=1):
        check_processing_cancelled()
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

        hotwords = _session_hotwords(session_lines)
        cache_identity = _recording_cache_identity(
            audio_sha256=inventory_item["sha256"],
            project=project,
            settings=settings,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            hotwords=hotwords,
        )
        cache_key = stable_hash(cache_identity)
        output_path = transcript_dir / f"{session['id']}.json"
        if output_path.exists() and not force:
            existing = read_json(output_path)
            if _recording_cache_matches(
                existing=existing,
                cache_key=cache_key,
                cache_identity=cache_identity,
                settings=settings,
                audio_sha256=inventory_item["sha256"],
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                language=project.get("language", "en"),
            ):
                print(
                    f"[transcribe {index}/{len(sessions)}] Cached: {session['id']}",
                    flush=True,
                )
                written.append(output_path)
                continue

        if model is None:
            check_processing_cancelled()
            print(f"[model cache] {model_root}", flush=True)
            model, resolved_device, resolved_compute = _make_model(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=model_root,
            )
            check_processing_cancelled()
            check_processing_cancelled()
            from faster_whisper import BatchedInferencePipeline

            batched_model = BatchedInferencePipeline(model=model)
            batch_runtime.update(
                {
                    "model_name": model_name,
                    "device": resolved_device,
                    "compute_type": resolved_compute,
                }
            )
            batch_size = _resolve_profile_batch_size(
                profile=batch_profile,
                runtime=batch_runtime,
            )

        print(
            f"[transcribe {index}/{len(sessions)}] {session['id']} "
            f"({resolved_device}/{resolved_compute}, batch size {batch_size}, "
            f"{batch_runtime['batch_size_source']})",
            flush=True,
        )
        kwargs = {
            "language": project.get("language", "en"),
            "beam_size": int(settings.get("beam_size", 5)),
            "word_timestamps": True,
            "without_timestamps": False,
            "batch_size": int(batch_size),
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
        if hotwords:
            kwargs["hotwords"] = hotwords

        try:
            segments_iter, info = batched_model.transcribe(
                str(audio_path),
                **kwargs,
            )
            segment_records = []
            for segment in segments_iter:
                check_processing_cancelled()
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
            from faster_whisper import BatchedInferencePipeline

            batched_model = BatchedInferencePipeline(model=model)
            batch_runtime.update(
                {
                    "model_name": model_name,
                    "device": resolved_device,
                    "compute_type": resolved_compute,
                }
            )
            batch_size = _resolve_profile_batch_size(
                profile=batch_profile,
                runtime=batch_runtime,
            )
            kwargs["batch_size"] = batch_size
            segments_iter, info = batched_model.transcribe(
                str(audio_path),
                **kwargs,
            )
            segment_records = []
            for segment in segments_iter:
                check_processing_cancelled()
                segment_records.append(
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
                )

        check_processing_cancelled()
        payload = {
            "schema_version": 1,
            "session_id": session["id"],
            "audio": session["audio"],
            "audio_sha256": inventory_item["sha256"],
            "cache_key": cache_key,
            "cache_identity": cache_identity,
            "model": model_name,
            "device": resolved_device,
            "compute_type": resolved_compute,
            "batched_inference": True,
            "batch_size_requested": batch_request,
            "batch_size": int(batch_size),
            "batch_size_source": batch_runtime["batch_size_source"],
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_seconds": info.duration,
            "duration_after_vad_seconds": info.duration_after_vad,
            "segments": segment_records,
        }
        write_json(output_path, payload)
        written.append(output_path)
    return written
