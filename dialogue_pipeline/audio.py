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
        overlap = min(end, float(segment["end"])) - max(
            start, float(segment["start"])
        )
        if overlap > 0:
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
