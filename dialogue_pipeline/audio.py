from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .util import sha256_file


SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(
    r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)"
)


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"Required executable is not on PATH: {name}")
    return executable


def probe_audio(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    ffprobe = require_executable("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration,size:"
                "stream=index,codec_name,codec_type,sample_fmt,sample_rate,"
                "channels,channel_layout,bits_per_sample"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result.stdout)
    audio_streams = [
        stream
        for stream in payload.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]
    if len(audio_streams) != 1:
        raise ValueError(f"Expected one audio stream in {path}, found {len(audio_streams)}")
    stream = audio_streams[0]
    output = {
        "path": str(path.resolve()),
        "size_bytes": int(payload["format"].get("size") or path.stat().st_size),
        "duration_seconds": float(payload["format"]["duration"]),
        "codec": stream.get("codec_name"),
        "sample_format": stream.get("sample_fmt"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "channel_layout": stream.get("channel_layout"),
        "bits_per_sample": int(stream.get("bits_per_sample") or 0),
    }
    if include_hash:
        output["sha256"] = sha256_file(path)
    return output


def pcm_wav_shape(path: Path) -> dict[str, int] | None:
    try:
        with wave.open(str(path), "rb") as reader:
            params = reader.getparams()
            if params.comptype != "NONE":
                return None
            return {
                "sample_rate": params.framerate,
                "channels": params.nchannels,
                "bits_per_sample": params.sampwidth * 8,
                "frame_count": params.nframes,
            }
    except (EOFError, wave.Error):
        return None


def quietest_pcm_boundary(
    path: Path,
    *,
    proposed_sample: int,
    minimum_sample: int,
    maximum_sample: int,
    search_seconds: float = 0.20,
    window_seconds: float = 0.02,
    maximum_rms_dbfs: float = -42.0,
) -> int:
    """Snap a proposed PCM cut to the quietest nearby analysis window."""

    try:
        with wave.open(str(path), "rb") as reader:
            if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
                return int(proposed_sample)
            sample_rate = int(reader.getframerate())
            channels = int(reader.getnchannels())
            frame_count = int(reader.getnframes())
            lower = max(
                0,
                int(minimum_sample),
                int(proposed_sample)
                - round(max(0.0, float(search_seconds)) * sample_rate),
            )
            upper = min(
                frame_count,
                int(maximum_sample),
                int(proposed_sample)
                + round(max(0.0, float(search_seconds)) * sample_rate),
            )
            window_frames = max(
                1,
                round(max(0.001, float(window_seconds)) * sample_rate),
            )
            half_window = window_frames // 2
            first_center = lower + half_window
            last_center = upper - (window_frames - half_window)
            if last_center < first_center:
                return int(proposed_sample)
            reader.setpos(lower)
            raw = reader.readframes(upper - lower)
    except (EOFError, OSError, wave.Error):
        return int(proposed_sample)

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples /= 32768.0
    hop_frames = max(1, round(0.005 * sample_rate))
    candidates: list[tuple[float, int]] = []
    for center in range(first_center, last_center + 1, hop_frames):
        local_start = center - half_window - lower
        local_end = local_start + window_frames
        window = samples[local_start:local_end]
        if window.shape[0] != window_frames:
            continue
        rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
        candidates.append((rms, center))
    if not candidates:
        return int(proposed_sample)

    best_rms, best_sample = min(
        candidates,
        key=lambda item: (
            item[0],
            abs(item[1] - int(proposed_sample)),
        ),
    )
    best_dbfs = 20.0 * math.log10(max(best_rms, 1e-12))
    if best_dbfs > float(maximum_rms_dbfs):
        return int(proposed_sample)
    return int(best_sample)


def pcm_voice_bounds(
    path: Path,
    *,
    start_sample: int,
    end_sample: int,
    threshold: float = 0.5,
) -> tuple[int, int] | None:
    """Return absolute PCM bounds classified as speech by Silero VAD."""

    try:
        with wave.open(str(path), "rb") as reader:
            if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
                return None
            sample_rate = int(reader.getframerate())
            channels = int(reader.getnchannels())
            frame_count = int(reader.getnframes())
            start_sample = max(0, min(int(start_sample), frame_count))
            end_sample = max(
                start_sample,
                min(int(end_sample), frame_count),
            )
            if end_sample <= start_sample:
                return None
            reader.setpos(start_sample)
            raw = reader.readframes(end_sample - start_sample)
    except (EOFError, OSError, wave.Error):
        return None

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples /= 32768.0
    vad_rate = 16000
    if sample_rate != vad_rate:
        target_count = max(
            1,
            round(samples.shape[0] * vad_rate / sample_rate),
        )
        source_positions = np.linspace(
            0.0,
            max(0.0, float(samples.shape[0] - 1)),
            num=target_count,
        )
        samples = np.interp(
            source_positions,
            np.arange(samples.shape[0], dtype=np.float64),
            samples,
        ).astype(np.float32)

    from faster_whisper.vad import VadOptions, get_speech_timestamps

    timestamps = get_speech_timestamps(
        samples,
        VadOptions(
            threshold=float(threshold),
            min_speech_duration_ms=80,
            min_silence_duration_ms=100,
            speech_pad_ms=0,
        ),
    )
    if not timestamps:
        return None
    first = int(timestamps[0]["start"])
    last = int(timestamps[-1]["end"])
    voiced_start = start_sample + round(first * sample_rate / vad_rate)
    voiced_end = start_sample + round(last * sample_rate / vad_rate)
    return (
        max(start_sample, min(voiced_start, end_sample)),
        max(start_sample, min(voiced_end, end_sample)),
    )


def prepare_pcm_segmentation_source(
    source: Path,
    destination: Path,
    *,
    sample_rate: int,
    channels: int,
    bits_per_sample: int,
) -> tuple[Path, dict[str, int], bool]:
    source_shape = pcm_wav_shape(source)
    expected = {
        "sample_rate": sample_rate,
        "channels": channels,
        "bits_per_sample": bits_per_sample,
    }
    if source_shape and all(
        source_shape[key] == value for key, value in expected.items()
    ):
        return source, source_shape, False

    if bits_per_sample != 16:
        raise ValueError(
            "Segmentation normalization currently supports 16-bit PCM output only; "
            f"requested {bits_per_sample}-bit."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_shape = pcm_wav_shape(destination) if destination.exists() else None
    if destination_shape and all(
        destination_shape[key] == value for key, value in expected.items()
    ):
        return destination, destination_shape, True

    ffmpeg = require_executable("ffmpeg")
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".wav",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)

    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-map_metadata",
                "-1",
                "-ac",
                str(channels),
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg normalization failed for {source}:\n{result.stderr}"
            )
        normalized_shape = pcm_wav_shape(temporary)
        if not normalized_shape or any(
            normalized_shape[key] != value for key, value in expected.items()
        ):
            raise RuntimeError(
                f"FFmpeg produced an unexpected WAV format for {source}: "
                f"{normalized_shape!r}"
            )
        os.replace(temporary, destination)
        return destination, normalized_shape, True
    finally:
        if temporary.exists():
            temporary.unlink()


def detect_silences(
    path: Path, *, noise_db: float, minimum_duration_seconds: float
) -> list[dict[str, float]]:
    ffmpeg = require_executable("ffmpeg")
    filter_value = (
        f"silencedetect=noise={noise_db:g}dB:d={minimum_duration_seconds:g}"
    )
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            filter_value,
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg silence detection failed for {path}:\n{result.stderr}")

    silences: list[dict[str, float]] = []
    pending_start: float | None = None
    for line in result.stderr.splitlines():
        start_match = SILENCE_START_RE.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = SILENCE_END_RE.search(line)
        if end_match:
            end = float(end_match.group(1))
            duration = float(end_match.group(2))
            start = pending_start if pending_start is not None else end - duration
            silences.append({"start": start, "end": end, "duration": duration})
            pending_start = None
    return silences


def acoustic_regions(
    duration_seconds: float,
    silences: list[dict[str, float]],
    *,
    split_gap_seconds: float,
    minimum_segment_seconds: float,
    pre_padding_seconds: float,
    post_padding_seconds: float,
) -> list[dict[str, float]]:
    split_silences = [
        silence for silence in silences if silence["duration"] >= split_gap_seconds
    ]
    regions: list[dict[str, float]] = []
    speech_start = 0.0

    for silence in split_silences:
        speech_end = max(speech_start, silence["start"])
        if speech_end - speech_start >= minimum_segment_seconds:
            regions.append(
                {
                    "speech_start": speech_start,
                    "speech_end": speech_end,
                    "start": max(0.0, speech_start - pre_padding_seconds),
                    "end": min(
                        duration_seconds, speech_end + post_padding_seconds
                    ),
                }
            )
        speech_start = max(speech_start, silence["end"])

    if duration_seconds - speech_start >= minimum_segment_seconds:
        regions.append(
            {
                "speech_start": speech_start,
                "speech_end": duration_seconds,
                "start": max(0.0, speech_start - pre_padding_seconds),
                "end": duration_seconds,
            }
        )
    return regions


def transcript_for_region(
    transcription: dict[str, Any], start: float, end: float
) -> tuple[str, list[dict[str, Any]], float | None]:
    words = []
    for segment in transcription.get("segments", []):
        for word in segment.get("words") or []:
            word_start = float(word.get("start", segment["start"]))
            word_end = float(word.get("end", segment["end"]))
            midpoint = (word_start + word_end) / 2.0
            if start <= midpoint <= end:
                words.append(word)

    if words:
        text = "".join(str(word.get("word") or "") for word in words).strip()
        probabilities = [
            float(word["probability"])
            for word in words
            if word.get("probability") is not None
        ]
        probability = (
            sum(probabilities) / len(probabilities) if probabilities else None
        )
        return text, words, probability

    overlapping = []
    for segment in transcription.get("segments", []):
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])
        overlap = min(end, segment_end) - max(start, segment_start)
        segment_duration = max(0.001, segment_end - segment_start)
        if overlap > 0 and overlap / segment_duration >= 0.50:
            overlapping.append(str(segment.get("text") or "").strip())
    return " ".join(text for text in overlapping if text).strip(), [], None


def cut_pcm_wav(
    source: Path,
    destination: Path,
    *,
    start_sample: int,
    end_sample: int,
    fade_ms: float = 5.0,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        if params.comptype != "NONE":
            raise ValueError(f"Compressed WAV is not supported for exact cutting: {source}")
        if params.sampwidth != 2:
            raise ValueError(
                f"Only 16-bit PCM WAV is currently supported, got "
                f"{params.sampwidth * 8}-bit: {source}"
            )
        start_sample = max(0, min(start_sample, params.nframes))
        end_sample = max(start_sample, min(end_sample, params.nframes))
        reader.setpos(start_sample)
        raw = reader.readframes(end_sample - start_sample)

    samples = np.frombuffer(raw, dtype="<i2").copy()
    if params.nchannels > 1:
        samples = samples.reshape(-1, params.nchannels)
    frame_count = samples.shape[0]
    fade_frames = min(
        frame_count // 2,
        max(0, int(round(params.framerate * fade_ms / 1000.0))),
    )
    if fade_frames:
        fade_in = np.linspace(0.0, 1.0, fade_frames, endpoint=True)
        fade_out = np.linspace(1.0, 0.0, fade_frames, endpoint=True)
        if params.nchannels > 1:
            fade_in = fade_in[:, None]
            fade_out = fade_out[:, None]
        samples[:fade_frames] = np.rint(samples[:fade_frames] * fade_in).astype(
            np.int16
        )
        samples[-fade_frames:] = np.rint(
            samples[-fade_frames:] * fade_out
        ).astype(np.int16)

    flat_samples = samples.reshape(-1).astype("<i2", copy=False)
    peak = int(np.max(np.abs(flat_samples.astype(np.int32)))) if flat_samples.size else 0
    rms = (
        float(np.sqrt(np.mean(flat_samples.astype(np.float64) ** 2)))
        if flat_samples.size
        else 0.0
    )
    peak_dbfs = 20.0 * math.log10(peak / 32768.0) if peak else float("-inf")
    rms_dbfs = 20.0 * math.log10(rms / 32768.0) if rms else float("-inf")
    clipping_samples = int(np.count_nonzero(np.abs(flat_samples.astype(np.int32)) >= 32767))

    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(flat_samples.tobytes())

    return {
        "sample_rate": params.framerate,
        "channels": params.nchannels,
        "bits_per_sample": params.sampwidth * 8,
        "frame_count": frame_count,
        "duration_seconds": frame_count / params.framerate,
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "clipping_samples": clipping_samples,
    }


def seconds_to_sample(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))
