from __future__ import annotations

import contextlib
import copy
import queue
import re
import sys
import tempfile
import threading
import traceback
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .alignment import align_project
from .alignment_settings import AlignmentSettings
from .audio import cut_pcm_wav, open_pcm_wav
from .cancellation import (
    ProcessingCancelled,
    cancellation_scope,
    check_processing_cancelled,
)
from .finalize import finalize_review
from .project import (
    apply_project_settings,
    create_project,
    editable_project_settings,
    inventory_by_path,
    load_project,
    load_source_data,
)
from .retakes import export_retake_script
from .review import (
    add_base_segment_candidate,
    delete_edited_candidate,
    LINE_STATUSES,
    REVIEW_FILE_NAME,
    load_line_review,
    save_line_review,
    save_edited_candidate,
    segment_edit_source,
    segment_file_for_id,
    transcribe_edited_candidate,
)
from .segmentation import refresh_project_audio, segment_project
from .transcription import transcribe_project, transcribe_segments_project
from .util import project_file_from_arg, read_json, resolve_project_path, write_json
from .workbook_io import lines_for_session


APP_TITLE = "Dialogue VA Pipeline"
COMPUTE_TYPES = (
    "auto",
    "float16",
    "int8",
    "int8_float16",
    "int8_float32",
    "bfloat16",
    "float32",
)
STATUS_COLORS = {
    "AUTO_OK": "#dcfce7",
    "REVIEW": "#fef3c7",
    "MISSING": "#fee2e2",
    "MANUALLY_REVIEWED": "#dbeafe",
    "RETAKE": "#fce7f3",
}

MULTIPLE_CONTEXTS_HEADER = "used in multiple contexts:"


def _format_excel_rows(rows: Any) -> str:
    """Format an exact Excel-row filter compactly for the mapping editor."""

    values = sorted({int(row) for row in (rows or [])})
    if not values:
        return ""
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def _parse_excel_rows(value: str) -> list[int]:
    """Parse comma-separated Excel rows and inclusive ranges."""

    text = value.strip()
    if not text:
        return []
    rows: set[int] = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("Empty item in the Excel rows filter.")
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if not match:
            raise ValueError(
                f"Invalid Excel row item {part!r}. Use values such as 12, 18-24."
            )
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first < 1 or last < 1:
            raise ValueError("Excel row numbers must be positive.")
        if last < first:
            raise ValueError(f"Excel row range {part!r} ends before it starts.")
        rows.update(range(first, last + 1))
    return sorted(rows)


def _validate_session_mappings(
    sessions: list[dict[str, Any]],
    source_data: dict[str, Any],
) -> None:
    """Reject mappings that would make the downstream pipeline unusable."""

    valid_sheets = {str(sheet["name"]) for sheet in source_data.get("sheets", [])}
    enabled_count = 0
    for session in sessions:
        session_id = str(session.get("id") or "<unnamed recording>")
        sheets = [str(sheet) for sheet in session.get("sheets") or []]
        unknown = [sheet for sheet in sheets if sheet not in valid_sheets]
        if unknown:
            raise ValueError(
                f"{session_id} refers to missing script sheet(s): "
                + ", ".join(unknown)
            )
        if not session.get("enabled", True):
            continue
        enabled_count += 1
        if not sheets:
            raise ValueError(
                f"Choose at least one script sheet for enabled recording "
                f"{session_id}."
            )
        if not lines_for_session(source_data, session):
            raise ValueError(
                f"The mapping for {session_id} contains no script lines. "
                "Check its sheets and optional Excel row filter."
            )
    if not enabled_count:
        raise ValueError("Enable at least one audio recording before processing.")


def _mapping_sheet_action_names(
    *,
    mapped_sheets: list[str],
    available_sheets: list[str],
    selected_sheet: str | None,
) -> tuple[str, str]:
    selected_action = (
        "Remove" if selected_sheet in mapped_sheets else "Add"
    )
    remove_all = bool(mapped_sheets) and all(
        sheet in mapped_sheets for sheet in available_sheets
    )
    return selected_action, "Remove All" if remove_all else "Add All"


def _context_display_text(value: Any) -> str:
    """Remove repeated multi-context headings while preserving context text."""

    lines: list[str] = []
    header_seen = False
    for line in str(value or "").strip().splitlines():
        normalized = " ".join(line.strip().casefold().split())
        if normalized == MULTIPLE_CONTEXTS_HEADER:
            if header_seen:
                continue
            header_seen = True
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _uses_unmatched_candidates(line: dict[str, Any]) -> bool:
    has_alignment_candidates = any(
        not bool(candidate.get("manual_edit"))
        for candidate in line.get("candidates") or []
    )
    return bool(
        line["type"] == "nonverbal"
        or (
            not has_alignment_candidates
            and line["status"]
            in {"REVIEW", "MISSING", "MANUALLY_REVIEWED", "RETAKE"}
        )
    )


def _selected_segment_score(
    review_data: dict[str, Any],
    line: dict[str, Any],
) -> float | None:
    selected_id = line.get("selected_segment_id")
    if not selected_id:
        return None
    for candidate in line.get("candidates") or []:
        if candidate["segment_id"] == selected_id:
            return float(candidate.get("score", 0.0))
    for candidate in review_data["unmatched_segments"]:
        if candidate["segment_id"] == selected_id:
            return float(candidate.get("score", 0.0))
    return None


def _selected_line_ids_by_segment(
    review_data: dict[str, Any],
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for line in review_data["lines"]:
        segment_id = line.get("selected_segment_id")
        if segment_id:
            selected.setdefault(str(segment_id), []).append(str(line["line_id"]))
    return selected


def _candidate_selection_display(
    *,
    line: dict[str, Any],
    segment_id: str,
    selected_line_ids: dict[str, list[str]],
) -> tuple[str, tuple[str, ...]]:
    is_selected = str(line.get("selected_segment_id") or "") == segment_id
    other_selection_count = len(
        [
            selected_line_id
            for selected_line_id in selected_line_ids.get(segment_id, [])
            if selected_line_id != str(line["line_id"])
        ]
    )
    if is_selected:
        return (
            (
                f"Unselect (+{other_selection_count})"
                if other_selection_count
                else "Unselect"
            ),
            ("selected",),
        )
    if line["type"] == "nonverbal" and other_selection_count:
        return f"In use ({other_selection_count})", ("selected_elsewhere",)
    return "Select", ()


def _candidate_base_segment_keys(
    candidate: dict[str, Any],
) -> set[tuple[str, int]]:
    """Return every base segment represented by a candidate span."""

    session_id = str(candidate.get("session_id") or "")
    base_indices = candidate.get("base_indices") or []
    if session_id and base_indices:
        return {(session_id, int(base_index)) for base_index in base_indices}
    match = re.fullmatch(r"(.+)__s(\d+)", str(candidate.get("segment_id") or ""))
    if match:
        return {(session_id or match.group(1), int(match.group(2)) - 1)}
    return set()


def _base_segment_candidate_usage(
    review_data: dict[str, Any],
    line_id: str,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    """Return base keys used by this line and by every other line."""

    current: set[tuple[str, int]] = set()
    other: set[tuple[str, int]] = set()
    for line in review_data["lines"]:
        destination = current if str(line["line_id"]) == line_id else other
        for candidate in line.get("candidates") or []:
            destination.update(_candidate_base_segment_keys(candidate))
    return current, other


def _first_line_base_segment_key(
    line: dict[str, Any],
) -> tuple[str, int] | None:
    """Choose the first base segment of the earliest candidate in time."""

    def candidate_time(candidate: dict[str, Any], field: str) -> float:
        try:
            return float(candidate[field])
        except (KeyError, TypeError, ValueError):
            return float("inf")

    candidates = sorted(
        line.get("candidates") or [],
        key=lambda candidate: (
            candidate_time(candidate, "start_seconds"),
            candidate_time(candidate, "end_seconds"),
            str(candidate.get("session_id") or ""),
            min(
                (int(value) for value in candidate.get("base_indices") or []),
                default=sys.maxsize,
            ),
            str(candidate.get("segment_id") or ""),
        ),
    )
    for candidate in candidates:
        keys = _candidate_base_segment_keys(candidate)
        if keys:
            return min(keys, key=lambda key: key[1])
    return None


def _candidate_take_source(
    candidate: dict[str, Any],
    candidates_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Follow a custom edit chain to the candidate from which it was made."""

    current = candidate
    visited: set[str] = set()
    while current.get("edited_from_segment_id"):
        source_id = str(current["edited_from_segment_id"])
        if source_id in visited:
            break
        visited.add(source_id)
        source = candidates_by_id.get(source_id)
        if source is None:
            current = {"segment_id": source_id}
            break
        current = source
    return current


def _candidate_take_key(
    candidate: dict[str, Any],
    candidates_by_id: dict[str, dict[str, Any]],
    *,
    use_acoustic_span: bool = False,
) -> tuple[str, str, tuple[int, ...] | str]:
    """Return the take identity, following a custom edit to its source."""

    current = _candidate_take_source(candidate, candidates_by_id)

    session_id = str(current.get("session_id") or "")
    if use_acoustic_span:
        start_seconds = current.get("start_seconds")
        end_seconds = current.get("end_seconds")
        if start_seconds is not None and end_seconds is not None:
            start_micros = round(float(start_seconds) * 1_000_000)
            end_micros = round(float(end_seconds) * 1_000_000)
            if end_micros > start_micros:
                return "time", session_id, (start_micros, end_micros)
    base_indices = tuple(int(value) for value in current.get("base_indices") or [])
    if base_indices:
        return "base", session_id, base_indices

    segment_id = str(current.get("segment_id") or "")
    match = re.fullmatch(r"(.+)__s(\d+)", segment_id)
    if match:
        return "base", session_id or match.group(1), (int(match.group(2)) - 1,)
    return "segment", session_id, segment_id


def _candidate_group_sort_key(
    candidate: dict[str, Any],
    original_order: dict[str, int],
) -> tuple[float, float, bool, int]:
    return (
        -float(candidate.get("score", 0.0)),
        -float(candidate.get("selection_score", 0.0)),
        bool(candidate.get("manual_edit")),
        original_order[str(candidate["segment_id"])],
    )


def _segment_gap(
    segments: list[dict[str, Any]],
    left_index: int,
    right_index: int,
) -> float:
    return max(
        0.0,
        float(segments[right_index].get("start_seconds", 0.0))
        - float(segments[left_index].get("end_seconds", 0.0)),
    )


def _candidate_acoustic_isolation(
    base_indices: tuple[int, ...],
    segments: list[dict[str, Any]],
) -> float | None:
    """Score how clearly a span is separated from neighboring performances."""

    if not base_indices:
        return None
    start_index = min(base_indices)
    end_index = max(base_indices)
    if (
        start_index < 0
        or end_index >= len(segments)
        or tuple(range(start_index, end_index + 1)) != base_indices
    ):
        return None

    internal_gaps = [
        _segment_gap(segments, index, index + 1)
        for index in range(start_index, end_index)
    ]
    known_edge_gaps = []
    if start_index > 0:
        known_edge_gaps.append(
            _segment_gap(segments, start_index - 1, start_index)
        )
    if end_index + 1 < len(segments):
        known_edge_gaps.append(
            _segment_gap(segments, end_index, end_index + 1)
        )
    edge_default = max([0.0, *known_edge_gaps, *internal_gaps])
    leading_gap = (
        _segment_gap(segments, start_index - 1, start_index)
        if start_index > 0
        else edge_default
    )
    trailing_gap = (
        _segment_gap(segments, end_index, end_index + 1)
        if end_index + 1 < len(segments)
        else edge_default
    )
    # A candidate that crosses a large pause is less likely to be one take,
    # even if its two outside edges also happen to be well separated.
    return leading_gap + trailing_gap - 2.0 * max(internal_gaps, default=0.0)


def _time_interval_overlap(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    shorter = min(left[1] - left[0], right[1] - right[0])
    if shorter <= 0.0 or overlap / shorter < 0.20:
        return 0.0
    return overlap


def _candidate_take_groups(
    candidates: list[dict[str, Any]],
    base_segments_by_session: dict[str, list[dict[str, Any]]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Group exact and overlapping variants around acoustically clean takes."""

    candidates_by_id = {
        str(candidate["segment_id"]): candidate for candidate in candidates
    }
    original_order = {
        str(candidate["segment_id"]): index
        for index, candidate in enumerate(candidates)
    }
    exact_groups: dict[
        tuple[str, str, tuple[int, ...] | str],
        list[dict[str, Any]],
    ] = {}
    for candidate in candidates:
        key = _candidate_take_key(
            candidate,
            candidates_by_id,
            use_acoustic_span=bool(base_segments_by_session),
        )
        exact_groups.setdefault(key, []).append(candidate)
    for group in exact_groups.values():
        group.sort(
            key=lambda candidate: _candidate_group_sort_key(
                candidate,
                original_order,
            )
        )

    if not base_segments_by_session:
        return list(exact_groups.values())

    nodes_by_session: dict[str, list[dict[str, Any]]] = {}
    fallback_nodes: list[dict[str, Any]] = []
    for key, group in exact_groups.items():
        kind, session_id, identity = key
        segments = base_segments_by_session.get(session_id)
        if not isinstance(identity, tuple) or not segments:
            fallback_nodes.append({"group": group})
            continue
        source_candidates = [
            _candidate_take_source(candidate, candidates_by_id)
            for candidate in group
        ]
        base_indices = tuple(
            int(value)
            for value in source_candidates[0].get("base_indices") or []
        )
        if kind == "base":
            base_indices = identity
        isolation = _candidate_acoustic_isolation(base_indices, segments)
        if isolation is None:
            fallback_nodes.append({"group": group})
            continue
        if kind == "time":
            interval = (identity[0] / 1_000_000.0, identity[1] / 1_000_000.0)
        else:
            interval = (
                float(segments[min(base_indices)]["start_seconds"]),
                float(segments[max(base_indices)]["end_seconds"]),
            )
        root_candidate = group[0]
        node = {
            "group": group,
            "interval": interval,
            "isolation": isolation,
            "priority": (
                float(root_candidate.get("score", 0.0))
                + 4.0 * isolation
            ),
            "repeated_take": any(
                bool(source.get("repeated_take_trim"))
                for source in source_candidates
            ),
            "original_order": min(
                original_order[str(candidate["segment_id"])]
                for candidate in group
            ),
        }
        nodes_by_session.setdefault(session_id, []).append(node)

    clustered_nodes: list[dict[str, Any]] = []
    for session_nodes in nodes_by_session.values():
        roots: list[dict[str, Any]] = []
        for node in sorted(
            session_nodes,
            key=lambda value: (
                -int(bool(value["repeated_take"])),
                -float(value["priority"]),
                -float(value["isolation"]),
                int(value["original_order"]),
            ),
        ):
            if any(
                _time_interval_overlap(node["interval"], root["interval"])
                for root in roots
            ):
                continue
            roots.append(node)

        members_by_root = {id(root): [root] for root in roots}
        for node in session_nodes:
            if node in roots:
                continue
            node_interval = node["interval"]
            overlapping_roots = [
                root
                for root in roots
                if _time_interval_overlap(node_interval, root["interval"])
            ]
            if not overlapping_roots:
                roots.append(node)
                members_by_root[id(node)] = [node]
                continue
            node_length = node_interval[1] - node_interval[0]
            node_center = sum(node_interval) / 2.0
            owner = max(
                overlapping_roots,
                key=lambda root: (
                    _time_interval_overlap(node_interval, root["interval"])
                    / node_length,
                    _time_interval_overlap(node_interval, root["interval"]),
                    -abs(node_center - sum(root["interval"]) / 2.0),
                    float(root["priority"]),
                ),
            )
            members_by_root[id(owner)].append(node)

        for root in roots:
            alternatives = [
                candidate
                for node in members_by_root[id(root)]
                if node is not root
                for candidate in node["group"]
            ]
            alternatives.sort(
                key=lambda candidate: _candidate_group_sort_key(
                    candidate,
                    original_order,
                )
            )
            clustered_nodes.append(
                {
                    "group": [*root["group"], *alternatives],
                    "original_order": min(
                        int(node["original_order"])
                        for node in members_by_root[id(root)]
                    ),
                }
            )

    clustered_nodes.extend(
        {
            "group": node["group"],
            "original_order": min(
                original_order[str(candidate["segment_id"])]
                for candidate in node["group"]
            ),
        }
        for node in fallback_nodes
    )
    clustered_nodes.sort(key=lambda node: int(node["original_order"]))
    return [node["group"] for node in clustered_nodes]


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    with open_pcm_wav(path) as reader:
        if reader.getcomptype() != "NONE":
            raise ValueError(f"Unsupported compressed WAV: {path}")
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())

    if sample_width == 1:
        samples = np.frombuffer(frames, dtype=np.uint8)
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2")
    elif sample_width == 3:
        packed = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        samples = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        samples = ((samples ^ 0x800000) - 0x800000) << 8
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4")
    else:
        raise ValueError(f"Unsupported {sample_width * 8}-bit WAV: {path}")

    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples, sample_rate


def _sounddevice_output_options() -> dict[str, Any]:
    if sys.platform != "win32":
        return {}
    for host_api in sd.query_hostapis():
        if host_api["name"] != "Windows WASAPI":
            continue
        device = int(host_api["default_output_device"])
        if device < 0:
            break
        return {
            "device": device,
            "extra_settings": sd.WasapiSettings(
                exclusive=False,
                auto_convert=True,
            ),
        }
    raise RuntimeError("No default Windows WASAPI output device is available.")


def _playback_samples(
    samples: np.ndarray,
    sample_rate: int,
    *,
    output_sample_rate: int,
) -> np.ndarray:
    if samples.dtype == np.uint8:
        samples = (samples.astype(np.float32) - 128.0) / 128.0
    elif samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        samples = samples.astype(np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    if sample_rate == output_sample_rate or samples.size == 0:
        return np.asarray(samples, dtype=np.float32)
    output_length = max(1, round(samples.size * output_sample_rate / sample_rate))
    source_positions = np.arange(samples.size, dtype=np.float64)
    output_positions = (
        np.arange(output_length, dtype=np.float64) * sample_rate / output_sample_rate
    )
    return np.interp(output_positions, source_positions, samples).astype(np.float32)


class AudioPlayer:
    sample_rate = 48000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples = np.empty(0, dtype=np.float32)
        self._position = 0
        self._playing = False
        self._paused = False
        self._playback_id = 0
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._fill_output,
            **_sounddevice_output_options(),
        )
        self._stream.start()

    def _fill_output(
        self,
        output: np.ndarray,
        frames: int,
        _time: Any,
        _status: Any,
    ) -> None:
        output.fill(0)
        with self._lock:
            if not self._playing or self._paused:
                return
            remaining = self._samples.size - self._position
            copy_count = min(frames, max(0, remaining))
            if copy_count:
                output[:copy_count, 0] = self._samples[
                    self._position : self._position + copy_count
                ]
                self._position += copy_count
            if self._position >= self._samples.size:
                self._position = self._samples.size
                self._playing = False

    def stop(self) -> None:
        with self._lock:
            self._samples = np.empty(0, dtype=np.float32)
            self._position = 0
            self._playing = False
            self._paused = False
            self._playback_id += 1

    def close(self) -> None:
        self.stop()
        self._stream.stop()
        self._stream.close()

    def play(self, path: Path, *, start_seconds: float = 0.0) -> int:
        if not path.is_file():
            raise FileNotFoundError(path)
        samples, sample_rate = _read_pcm_wav(path)
        playback = _playback_samples(
            samples,
            sample_rate,
            output_sample_rate=self.sample_rate,
        )
        with self._lock:
            self._samples = playback
            self._position = max(
                0,
                min(
                    int(round(float(start_seconds) * self.sample_rate)),
                    playback.size,
                ),
            )
            self._playing = self._position < playback.size
            self._paused = False
            self._playback_id += 1
            return self._playback_id

    def pause(self, playback_id: int) -> bool:
        with self._lock:
            if (
                playback_id != self._playback_id
                or not self._playing
                or self._paused
            ):
                return False
            self._paused = True
            return True

    def seek(self, playback_id: int, position_seconds: float) -> bool:
        with self._lock:
            if playback_id != self._playback_id or not self._samples.size:
                return False
            self._position = max(
                0,
                min(
                    int(round(float(position_seconds) * self.sample_rate)),
                    self._samples.size,
                ),
            )
            self._playing = self._position < self._samples.size
            return True

    def resume(self, playback_id: int) -> bool:
        with self._lock:
            if (
                playback_id != self._playback_id
                or self._position >= self._samples.size
            ):
                return False
            self._playing = True
            self._paused = False
            return True

    def status(self, playback_id: int) -> tuple[float, float, bool, bool]:
        with self._lock:
            if playback_id != self._playback_id:
                return 0.0, 0.0, False, False
            return (
                self._position / self.sample_rate,
                self._samples.size / self.sample_rate,
                self._playing,
                self._paused,
            )


def _clock_time(sample: int, sample_rate: int) -> str:
    seconds = max(0.0, sample / sample_rate)
    minutes, seconds = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:06.3f}"
    return f"{minutes:02d}:{seconds:06.3f}"


def _zoomed_sample_window(
    *,
    view_start: int,
    view_end: int,
    context_start: int,
    context_end: int,
    selection_start: int,
    selection_end: int,
    anchor_sample: int,
    zoom_factor: float,
    sample_rate: int,
    maximum_span: int | None = None,
    keep_selection_visible: bool = True,
) -> tuple[int, int]:
    """Scale a waveform window around a sample while retaining both markers."""

    context_span = context_end - context_start
    view_span = view_end - view_start
    selection_span = selection_end - selection_start
    if context_span <= 0 or view_span <= 0 or selection_span <= 0:
        raise ValueError("Waveform zoom intervals must have positive durations.")
    if zoom_factor <= 0.0:
        raise ValueError("Waveform zoom factor must be positive.")

    maximum_span = (
        context_span
        if maximum_span is None
        else max(1, min(context_span, int(maximum_span)))
    )
    marker_margin = max(1, int(round(sample_rate * 0.05)))
    minimum_span = min(
        maximum_span,
        max(
            int(round(sample_rate * 0.25)),
            (
                selection_span + marker_margin * 2
                if keep_selection_visible
                else 0
            ),
        ),
    )
    target_span = max(
        minimum_span,
        min(maximum_span, int(round(view_span * zoom_factor))),
    )
    if target_span == view_span:
        return view_start, view_end

    anchor_sample = max(view_start, min(anchor_sample, view_end))
    anchor_fraction = (anchor_sample - view_start) / view_span
    new_start = int(round(anchor_sample - anchor_fraction * target_span))
    new_end = new_start + target_span

    if keep_selection_visible:
        required_start = selection_start - marker_margin
        required_end = selection_end + marker_margin
        if new_start > required_start:
            new_start = required_start
            new_end = new_start + target_span
        if new_end < required_end:
            new_end = required_end
            new_start = new_end - target_span

    if new_start < context_start:
        new_start = context_start
        new_end = new_start + target_span
    if new_end > context_end:
        new_end = context_end
        new_start = new_end - target_span
    return int(new_start), int(new_end)


def _initial_segment_window(
    *,
    context_start: int,
    context_end: int,
    selection_start: int,
    selection_end: int,
    sample_rate: int,
) -> tuple[int, int]:
    """Frame a segment prominently while retaining a little editing context."""

    context_span = context_end - context_start
    selection_span = selection_end - selection_start
    if context_span <= 0 or selection_span <= 0:
        raise ValueError("Editor intervals must have positive durations.")
    if not context_start <= selection_start < selection_end <= context_end:
        raise ValueError("The segment must be inside its editing context.")
    marker_margin = max(1, int(round(sample_rate * 0.05)))
    target_span = min(
        context_span,
        max(
            int(round(selection_span * 1.35)),
            selection_span + marker_margin * 2,
        ),
    )
    return _zoomed_sample_window(
        view_start=context_start,
        view_end=context_end,
        context_start=context_start,
        context_end=context_end,
        selection_start=selection_start,
        selection_end=selection_end,
        anchor_sample=(selection_start + selection_end) // 2,
        zoom_factor=target_span / context_span,
        sample_rate=sample_rate,
    )


def _panned_sample_window(
    *,
    view_start: int,
    view_end: int,
    context_start: int,
    context_end: int,
    drag_pixels: float,
    canvas_width: int,
) -> tuple[int, int]:
    """Move a fixed-width view as though the waveform were grabbed and dragged."""

    view_span = view_end - view_start
    context_span = context_end - context_start
    if view_span <= 0 or context_span < view_span or canvas_width <= 0:
        raise ValueError("Invalid waveform pan dimensions.")
    sample_delta = int(round(-drag_pixels * view_span / canvas_width))
    new_start = view_start + sample_delta
    maximum_start = context_end - view_span
    new_start = max(context_start, min(new_start, maximum_start))
    return new_start, new_start + view_span


class SegmentEditorDialog:
    WAVEFORM_BINS = 2400
    CONTEXT_SECONDS = 30.0

    def __init__(
        self,
        *,
        parent: tk.Tk,
        player: AudioPlayer,
        segment_id: str,
        audio_path: Path,
        sample_rate: int,
        source_frames: int,
        start_sample: int,
        end_sample: int,
        save_callback: Callable[[int, int], None],
    ) -> None:
        self.player = player
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.source_frames = source_frames
        self.save_callback = save_callback
        self.start_sample = start_sample
        self.end_sample = end_sample
        if not 0 <= start_sample < end_sample <= source_frames:
            raise ValueError("The candidate has invalid source sample boundaries.")
        context_frames = int(round(self.CONTEXT_SECONDS * sample_rate))
        self.context_start = max(0, start_sample - context_frames)
        self.context_end = min(source_frames, end_sample + context_frames)
        self.view_start, self.view_end = _initial_segment_window(
            context_start=self.context_start,
            context_end=self.context_end,
            selection_start=start_sample,
            selection_end=end_sample,
            sample_rate=sample_rate,
        )
        self._envelope_min, self._envelope_max = self._read_waveform()
        self._drag_boundary: str | None = None
        self._drag_changed = False
        self._pan_start_x: int | None = None
        self._pan_view_start = self.view_start
        self._pan_view_end = self.view_end
        self._pan_render_after_id: str | None = None
        self._playback_active = False
        self._playback_after_id: str | None = None
        self._playback_id: int | None = None
        self._playback_start_sample = start_sample
        self._playhead_sample: int | None = None
        self._drag_playhead = False
        self._resume_after_playhead_drag = False
        self._temporary_dir = tempfile.TemporaryDirectory(
            prefix="dialogue-va-segment-editor-"
        )
        self.playback_error_title = "Cannot play edited segment"
        self.saved = False

        self.window = tk.Toplevel(parent)
        self.window.title(f"Copy and edit segment — {segment_id}")
        self.window.geometry("1040x500")
        self.window.minsize(760, 420)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

        outer = ttk.Frame(self.window, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text=(
                "Drag the green start line and red end line to set the copy. "
                "Drag the blue playback line to seek. Mouse wheel: zoom. "
                "Right-drag: move along the timeline."
            ),
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 8))
        self.canvas = tk.Canvas(
            outer,
            height=310,
            background="#f8fafc",
            highlightthickness=1,
            highlightbackground="#94a3b8",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._canvas_resized)
        self.canvas.bind("<Button-1>", self._drag_started)
        self.canvas.bind("<B1-Motion>", self._drag_moved)
        self.canvas.bind("<ButtonRelease-1>", self._drag_finished)
        self.canvas.bind("<Button-3>", self._pan_started)
        self.canvas.bind("<B3-Motion>", self._pan_moved)
        self.canvas.bind("<ButtonRelease-3>", self._pan_finished)
        self.canvas.bind("<Motion>", self._canvas_motion)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", self._mouse_wheel_up)
        self.canvas.bind("<Button-5>", self._mouse_wheel_down)

        controls = ttk.Frame(outer, padding=(0, 12, 0, 0))
        controls.pack(fill="x")
        self.play_button = ttk.Button(
            controls,
            text="\u25b6 Play",
            command=self.play,
        )
        self.play_button.pack(side="left")
        self.bounds_text = tk.StringVar()
        ttk.Label(
            controls,
            textvariable=self.bounds_text,
            style="Muted.TLabel",
        ).pack(side="left", padx=16)
        ttk.Button(controls, text="Cancel", command=self.cancel).pack(
            side="right"
        )
        ttk.Button(
            controls,
            text="Save",
            style="Primary.TButton",
            command=self.save,
        ).pack(side="right", padx=(0, 8))

        self._update_bounds_text()
        self.window.grab_set()
        self.window.focus_set()
        self.window.after_idle(self._redraw)

    def _read_waveform(self) -> tuple[np.ndarray, np.ndarray]:
        with open_pcm_wav(self.audio_path) as reader:
            if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
                raise ValueError(
                    "Waveform editing requires an uncompressed 16-bit PCM source."
                )
            if reader.getframerate() != self.sample_rate:
                raise ValueError("The source sample rate does not match its manifest.")
            if reader.getnframes() != self.source_frames:
                raise ValueError("The source frame count does not match its manifest.")
            channels = reader.getnchannels()
            reader.setpos(self.view_start)
            raw = reader.readframes(self.view_end - self.view_start)
        samples = np.frombuffer(raw, dtype="<i2")
        if channels > 1 and samples.size:
            samples = samples.reshape(-1, channels).mean(axis=1)
        else:
            samples = samples.astype(np.float32, copy=False)
        if not samples.size:
            return np.zeros(1), np.zeros(1)
        bin_count = min(self.WAVEFORM_BINS, samples.size)
        edges = np.linspace(
            0,
            samples.size,
            bin_count + 1,
            dtype=np.int64,
        )
        starts = edges[:-1]
        minimum = np.minimum.reduceat(samples, starts) / 32768.0
        maximum = np.maximum.reduceat(samples, starts) / 32768.0
        return minimum, maximum

    def _canvas_resized(self, _event: tk.Event[Any]) -> None:
        self.window.after_idle(self._redraw)

    def _sample_to_x(self, sample: int) -> float:
        width = max(1, self.canvas.winfo_width())
        duration = max(1, self.view_end - self.view_start)
        return (sample - self.view_start) * width / duration

    def _x_to_sample(self, x: float) -> int:
        width = max(1, self.canvas.winfo_width())
        fraction = max(0.0, min(1.0, x / width))
        return int(
            round(
                self.view_start
                + fraction * (self.view_end - self.view_start)
            )
        )

    def _mouse_wheel(self, event: tk.Event[Any]) -> str:
        delta = int(getattr(event, "delta", 0))
        if not delta:
            return "break"
        steps = min(4, max(1, abs(delta) // 120))
        factor = (0.80 if delta > 0 else 1.25) ** steps
        self._change_time_scale(event.x, factor)
        return "break"

    def _mouse_wheel_up(self, event: tk.Event[Any]) -> str:
        self._change_time_scale(event.x, 0.80)
        return "break"

    def _mouse_wheel_down(self, event: tk.Event[Any]) -> str:
        self._change_time_scale(event.x, 1.25)
        return "break"

    def _change_time_scale(self, x: float, zoom_factor: float) -> None:
        self._cancel_pan_refresh()
        markers_visible = (
            self.view_start <= self.start_sample
            and self.end_sample <= self.view_end
        )
        new_start, new_end = _zoomed_sample_window(
            view_start=self.view_start,
            view_end=self.view_end,
            context_start=self.context_start,
            context_end=self.context_end,
            selection_start=self.start_sample,
            selection_end=self.end_sample,
            anchor_sample=self._x_to_sample(x),
            zoom_factor=zoom_factor,
            sample_rate=self.sample_rate,
            keep_selection_visible=markers_visible,
        )
        if (new_start, new_end) == (self.view_start, self.view_end):
            return
        self.view_start = new_start
        self.view_end = new_end
        self._envelope_min, self._envelope_max = self._read_waveform()
        self._update_bounds_text()
        self._redraw()

    def _pan_started(self, event: tk.Event[Any]) -> str:
        self._pan_start_x = event.x
        self._pan_view_start = self.view_start
        self._pan_view_end = self.view_end
        self.canvas.configure(cursor="fleur")
        return "break"

    def _pan_moved(self, event: tk.Event[Any]) -> str:
        if self._pan_start_x is None:
            return "break"
        width = max(1, self.canvas.winfo_width())
        new_start, new_end = _panned_sample_window(
            view_start=self._pan_view_start,
            view_end=self._pan_view_end,
            context_start=self.context_start,
            context_end=self.context_end,
            drag_pixels=event.x - self._pan_start_x,
            canvas_width=width,
        )
        if (new_start, new_end) != (self.view_start, self.view_end):
            self.view_start = new_start
            self.view_end = new_end
            self._schedule_pan_refresh()
        return "break"

    def _pan_finished(self, event: tk.Event[Any]) -> str:
        if self._pan_start_x is None:
            return "break"
        self._pan_moved(event)
        self._pan_start_x = None
        self._cancel_pan_refresh()
        self._refresh_waveform()
        self._canvas_motion(event)
        return "break"

    def _schedule_pan_refresh(self) -> None:
        if self._pan_render_after_id is None:
            self._pan_render_after_id = self.window.after(
                50,
                self._refresh_waveform,
            )

    def _cancel_pan_refresh(self) -> None:
        if self._pan_render_after_id is not None:
            self.window.after_cancel(self._pan_render_after_id)
            self._pan_render_after_id = None

    def _refresh_waveform(self) -> None:
        self._pan_render_after_id = None
        if not self.window.winfo_exists():
            return
        self._envelope_min, self._envelope_max = self._read_waveform()
        self._update_bounds_text()
        self._redraw()

    def _redraw(self) -> None:
        if not self.window.winfo_exists():
            return
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        middle = height / 2.0
        start_x = self._sample_to_x(self.start_sample)
        end_x = self._sample_to_x(self.end_sample)
        self.canvas.create_rectangle(
            start_x,
            0,
            end_x,
            height,
            fill="#dbeafe",
            outline="",
        )
        self.canvas.create_line(0, middle, width, middle, fill="#cbd5e1")

        count = len(self._envelope_min)
        if count:
            x_values = np.linspace(0.0, width, count, endpoint=False)
            amplitude = max(1.0, middle - 24.0)
            for x, low, high in zip(
                x_values,
                self._envelope_min,
                self._envelope_max,
            ):
                self.canvas.create_line(
                    float(x),
                    middle - float(high) * amplitude,
                    float(x),
                    middle - float(low) * amplitude,
                    fill="#334155",
                )

        for fraction in np.linspace(0.0, 1.0, 7):
            x = float(fraction * width)
            sample = int(
                round(
                    self.view_start
                    + fraction * (self.view_end - self.view_start)
                )
            )
            self.canvas.create_line(
                x,
                height - 18,
                x,
                height,
                fill="#94a3b8",
            )
            self.canvas.create_text(
                x,
                height - 20,
                text=_clock_time(sample, self.sample_rate),
                anchor="s",
                fill="#475569",
                font=("Segoe UI", 8),
            )

        self._draw_boundary(
            start_x,
            "#16a34a",
            f"Start {_clock_time(self.start_sample, self.sample_rate)}",
            "nw",
        )
        self._draw_boundary(
            end_x,
            "#dc2626",
            f"End {_clock_time(self.end_sample, self.sample_rate)}",
            "ne",
        )
        self._draw_playhead()

    def _draw_playhead(self) -> None:
        self.canvas.delete("playhead")
        if (
            self._playhead_sample is None
            or not (self._playback_active or self._drag_playhead)
            or not self.view_start <= self._playhead_sample <= self.view_end
        ):
            return
        x = self._sample_to_x(self._playhead_sample)
        height = max(1, self.canvas.winfo_height())
        self.canvas.create_line(
            x,
            0,
            x,
            height,
            fill="#2563eb",
            width=3,
            tags=("playhead",),
        )
        self.canvas.create_polygon(
            x - 6,
            0,
            x + 6,
            0,
            x,
            8,
            fill="#2563eb",
            outline="",
            tags=("playhead",),
        )

    def _draw_boundary(
        self,
        x: float,
        color: str,
        label: str,
        anchor: str,
    ) -> None:
        height = max(1, self.canvas.winfo_height())
        self.canvas.create_line(x, 0, x, height, fill=color, width=3)
        offset = 5 if anchor == "nw" else -5
        self.canvas.create_text(
            x + offset,
            5,
            text=label,
            anchor=anchor,
            fill=color,
            font=("Segoe UI", 9, "bold"),
        )

    def _nearest_boundary(self, x: float) -> str | None:
        distances = {
            "start": abs(x - self._sample_to_x(self.start_sample)),
            "end": abs(x - self._sample_to_x(self.end_sample)),
        }
        name = min(distances, key=distances.get)
        return name if distances[name] <= 12.0 else None

    def _playhead_is_near(self, x: float) -> bool:
        return bool(
            self._playback_active
            and self._playhead_sample is not None
            and abs(x - self._sample_to_x(self._playhead_sample)) <= 12.0
        )

    def _drag_started(self, event: tk.Event[Any]) -> None:
        if self._playhead_is_near(event.x):
            self._drag_playhead = True
            self._resume_after_playhead_drag = bool(
                self._playback_id is not None
                and self.player.pause(self._playback_id)
            )
            if self._resume_after_playhead_drag:
                self._cancel_playback_timer()
                self.play_button.configure(text="▶ Resume")
            self.canvas.configure(cursor="sb_h_double_arrow")
            return
        self._drag_boundary = self._nearest_boundary(event.x)
        self._drag_changed = False

    def _drag_moved(self, event: tk.Event[Any]) -> None:
        if self._drag_playhead:
            self._playhead_sample = max(
                self.start_sample,
                min(self._x_to_sample(event.x), self.end_sample),
            )
            if self._playback_id is not None:
                self.player.seek(
                    self._playback_id,
                    (self._playhead_sample - self._playback_start_sample)
                    / self.sample_rate,
                )
            self._draw_playhead()
            return
        if self._drag_boundary is None:
            return
        sample = self._x_to_sample(event.x)
        minimum_frames = max(1, int(round(self.sample_rate * 0.01)))
        if self._drag_boundary == "start":
            sample = min(sample, self.end_sample - minimum_frames)
            sample = max(self.view_start, sample)
            changed = sample != self.start_sample
            self.start_sample = sample
        else:
            sample = max(sample, self.start_sample + minimum_frames)
            sample = min(self.view_end, sample)
            changed = sample != self.end_sample
            self.end_sample = sample
        self._drag_changed = self._drag_changed or changed
        self._update_bounds_text()
        self._redraw()

    def _drag_finished(self, _event: tk.Event[Any]) -> None:
        if self._drag_playhead:
            self._drag_playhead = False
            if (
                self._resume_after_playhead_drag
                and self._playback_id is not None
                and self.player.resume(self._playback_id)
            ):
                self.play_button.configure(text="⏸ Pause")
                self._schedule_playback_update()
            else:
                position, duration, playing, paused = (
                    self.player.status(self._playback_id)
                    if self._playback_id is not None
                    else (0.0, 0.0, False, False)
                )
                if paused and position < duration:
                    self.play_button.configure(text="▶ Resume")
                elif not playing:
                    self._playback_finished()
            self._resume_after_playhead_drag = False
            self._canvas_motion(_event)
            return
        changed = self._drag_changed
        self._drag_boundary = None
        self._drag_changed = False
        if changed and self._playback_active:
            self._play_current()

    def _canvas_motion(self, event: tk.Event[Any]) -> None:
        if self._pan_start_x is not None:
            self.canvas.configure(cursor="fleur")
            return
        self.canvas.configure(
            cursor="sb_h_double_arrow"
            if self._playhead_is_near(event.x) or self._nearest_boundary(event.x)
            else ""
        )

    def _update_bounds_text(self) -> None:
        duration = (self.end_sample - self.start_sample) / self.sample_rate
        visible_seconds = (self.view_end - self.view_start) / self.sample_rate
        self.bounds_text.set(
            f"Start {_clock_time(self.start_sample, self.sample_rate)}   "
            f"End {_clock_time(self.end_sample, self.sample_rate)}   "
            f"Duration {duration:.3f} s   View {visible_seconds:.2f} s"
        )

    def play(self) -> None:
        if self._playback_active and self._playback_id is not None:
            _position, _duration, playing, paused = self.player.status(
                self._playback_id
            )
            if playing and not paused:
                if self.player.pause(self._playback_id):
                    self._cancel_playback_timer()
                    self.play_button.configure(text="▶ Resume")
                return
            if paused:
                if self.player.resume(self._playback_id):
                    self.play_button.configure(text="⏸ Pause")
                    self._schedule_playback_update()
                else:
                    self._playback_finished()
                return
        self._play_current()

    def restart_playback(self) -> None:
        self._play_current()

    def _play_current(self, start_sample: int | None = None) -> None:
        self._cancel_playback_timer()
        self.player.stop()
        self._playback_active = False
        self._playback_id = None
        self._playhead_sample = None
        self._draw_playhead()
        start_sample = self.start_sample if start_sample is None else max(
            self.start_sample,
            min(int(start_sample), self.end_sample),
        )
        preview_path = Path(self._temporary_dir.name) / "preview.wav"
        try:
            cut_pcm_wav(
                self.audio_path,
                preview_path,
                start_sample=self.start_sample,
                end_sample=self.end_sample,
                fade_ms=0.0,
            )
            self._playback_id = self.player.play(
                preview_path,
                start_seconds=(start_sample - self.start_sample) / self.sample_rate,
            )
        except Exception as error:
            self._playback_active = False
            messagebox.showerror(
                self.playback_error_title,
                str(error),
                parent=self.window,
            )
            return
        self._playback_active = True
        self._playback_start_sample = self.start_sample
        self._playhead_sample = start_sample
        self.play_button.configure(text="⏸ Pause")
        self._draw_playhead()
        self._schedule_playback_update()

    def _schedule_playback_update(self) -> None:
        self._cancel_playback_timer()
        self._playback_after_id = self.window.after(
            33,
            self._update_playback_position,
        )

    def _update_playback_position(self) -> None:
        self._playback_after_id = None
        if not self._playback_active or self._playback_id is None:
            return
        position, _duration, playing, paused = self.player.status(self._playback_id)
        if not playing and not paused:
            self._playback_finished()
            return
        if not self._drag_playhead:
            self._playhead_sample = min(
                self.end_sample,
                self._playback_start_sample
                + int(round(position * self.sample_rate)),
            )
            self._draw_playhead()
        if not paused:
            self._schedule_playback_update()

    def _playback_finished(self) -> None:
        self._cancel_playback_timer()
        self._playback_active = False
        self._playback_id = None
        self._playhead_sample = None
        self._drag_playhead = False
        self._resume_after_playhead_drag = False
        self.play_button.configure(text="▶ Play")
        self._draw_playhead()

    def _cancel_playback_timer(self) -> None:
        if self._playback_after_id is not None:
            self.window.after_cancel(self._playback_after_id)
            self._playback_after_id = None

    def save(self) -> None:
        try:
            self.save_callback(self.start_sample, self.end_sample)
        except Exception as error:
            messagebox.showerror(
                "Cannot save edited segment",
                str(error),
                parent=self.window,
            )
            return
        self.saved = True
        self._close()

    def cancel(self) -> None:
        self.saved = False
        self._close()

    def _close(self) -> None:
        self._cancel_pan_refresh()
        self._cancel_playback_timer()
        self._playback_active = False
        self._playback_id = None
        self._playhead_sample = None
        self._drag_playhead = False
        self._resume_after_playhead_drag = False
        self.player.stop()
        with contextlib.suppress(tk.TclError):
            self.window.grab_release()
        self.window.destroy()
        with contextlib.suppress(OSError):
            self._temporary_dir.cleanup()


class CandidateWaveformView(SegmentEditorDialog):
    """Embedded, read-only version of the copy/edit waveform."""

    def __init__(
        self,
        *,
        parent: ttk.Frame,
        player: AudioPlayer,
    ) -> None:
        self.player = player
        self.audio_path = Path()
        self.sample_rate = 1
        self.source_frames = 1
        self.start_sample = 0
        self.end_sample = 1
        self.context_start = 0
        self.context_end = 1
        self.view_start = 0
        self.view_end = 1
        self._envelope_min = np.zeros(1)
        self._envelope_max = np.zeros(1)
        self._pan_start_x: int | None = None
        self._pan_view_start = 0
        self._pan_view_end = 1
        self._pan_render_after_id: str | None = None
        self._playback_active = False
        self._playback_after_id: str | None = None
        self._playback_id: int | None = None
        self._playback_start_sample = 0
        self._playhead_sample: int | None = None
        self._drag_playhead = False
        self._resume_after_playhead_drag = False
        self._drag_boundary: str | None = None
        self._drag_changed = False
        self._temporary_dir = tempfile.TemporaryDirectory(
            prefix="dialogue-va-candidate-preview-"
        )
        self.playback_error_title = "Cannot play candidate"
        self._loaded = False
        self._disposed = False

        self.frame = ttk.Labelframe(
            parent,
            text="Selected candidate waveform",
            padding=8,
        )
        # Shared waveform methods schedule redraws against ``window``.
        self.window = self.frame
        ttk.Label(
            self.frame,
            text=(
                "Drag the blue playback line to seek. Mouse wheel: zoom. "
                "Right-drag: move along the timeline."
            ),
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        self.canvas = tk.Canvas(
            self.frame,
            height=190,
            background="#f8fafc",
            highlightthickness=1,
            highlightbackground="#94a3b8",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._canvas_resized)
        self.canvas.bind("<Button-1>", self._drag_started)
        self.canvas.bind("<B1-Motion>", self._drag_moved)
        self.canvas.bind("<ButtonRelease-1>", self._drag_finished)
        self.canvas.bind("<Button-3>", self._pan_started)
        self.canvas.bind("<B3-Motion>", self._pan_moved)
        self.canvas.bind("<ButtonRelease-3>", self._pan_finished)
        self.canvas.bind("<Motion>", self._canvas_motion)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", self._mouse_wheel_up)
        self.canvas.bind("<Button-5>", self._mouse_wheel_down)

        controls = ttk.Frame(self.frame, padding=(0, 8, 0, 0))
        controls.pack(fill="x")
        self.play_button = ttk.Button(
            controls,
            text="\u25b6 Play",
            command=self.play,
            state="disabled",
        )
        self.play_button.pack(side="left")
        self.bounds_text = tk.StringVar(value="Select a candidate to view it.")
        ttk.Label(
            controls,
            textvariable=self.bounds_text,
            style="Muted.TLabel",
        ).pack(side="left", padx=16)
        self.frame.after_idle(self._redraw)

    def show_segment(
        self,
        *,
        audio_path: Path,
        sample_rate: int,
        source_frames: int,
        start_sample: int,
        end_sample: int,
    ) -> None:
        self._cancel_pan_refresh()
        self._cancel_playback_timer()
        self._playback_active = False
        self._playback_id = None
        self._playhead_sample = None
        self._drag_playhead = False
        self._resume_after_playhead_drag = False
        self.player.stop()
        if not 0 <= start_sample < end_sample <= source_frames:
            raise ValueError("The candidate has invalid source sample boundaries.")
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.source_frames = source_frames
        self.start_sample = start_sample
        self.end_sample = end_sample
        context_frames = int(round(self.CONTEXT_SECONDS * sample_rate))
        self.context_start = max(0, start_sample - context_frames)
        self.context_end = min(source_frames, end_sample + context_frames)
        self.view_start, self.view_end = _initial_segment_window(
            context_start=self.context_start,
            context_end=self.context_end,
            selection_start=start_sample,
            selection_end=end_sample,
            sample_rate=sample_rate,
        )
        self._envelope_min, self._envelope_max = self._read_waveform()
        self._loaded = True
        self.play_button.configure(state="normal")
        self.play_button.configure(text="▶ Play")
        self._update_bounds_text()
        self._redraw()

    def clear(self, message: str = "Select a candidate to view it.") -> None:
        if self._disposed:
            return
        self._cancel_pan_refresh()
        self._cancel_playback_timer()
        self._playback_active = False
        self._playback_id = None
        self._playhead_sample = None
        self._drag_playhead = False
        self._resume_after_playhead_drag = False
        self.player.stop()
        self._loaded = False
        self._pan_start_x = None
        self.play_button.configure(state="disabled")
        self.play_button.configure(text="▶ Play")
        self.bounds_text.set(message)
        self._redraw()

    def _redraw(self) -> None:
        if self._disposed or not self.frame.winfo_exists():
            return
        if self._loaded:
            super()._redraw()
            return
        self.canvas.delete("all")
        self.canvas.create_text(
            max(1, self.canvas.winfo_width()) / 2.0,
            max(1, self.canvas.winfo_height()) / 2.0,
            text=self.bounds_text.get(),
            fill="#64748b",
            font=("Segoe UI", 10),
        )

    def _change_time_scale(self, x: float, zoom_factor: float) -> None:
        if self._loaded:
            super()._change_time_scale(x, zoom_factor)

    def _pan_started(self, event: tk.Event[Any]) -> str:
        if not self._loaded:
            return "break"
        return super()._pan_started(event)

    def _nearest_boundary(self, _x: float) -> str | None:
        return None

    def _canvas_motion(self, event: tk.Event[Any]) -> None:
        self.canvas.configure(
            cursor=(
                "fleur"
                if self._pan_start_x is not None
                else (
                    "sb_h_double_arrow"
                    if self._playhead_is_near(event.x)
                    else ""
                )
            )
        )

    def play(self) -> None:
        if self._loaded:
            super().play()

    def dispose(self) -> None:
        if self._disposed:
            return
        self._cancel_pan_refresh()
        self._cancel_playback_timer()
        self._playback_active = False
        self._playback_id = None
        self._playhead_sample = None
        self._drag_playhead = False
        self._resume_after_playhead_drag = False
        self.player.stop()
        self._disposed = True
        with contextlib.suppress(OSError):
            self._temporary_dir.cleanup()


class QueueWriter:
    def __init__(self, messages: queue.Queue[tuple[str, Any]]) -> None:
        self.messages = messages

    def write(self, value: str) -> int:
        if value:
            self.messages.put(("log", value))
        return len(value)

    def flush(self) -> None:
        return None


def _configured_batch_size(value: Any) -> int | str:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "auto"
    try:
        batch_size = int(normalized)
    except ValueError as error:
        raise ValueError(
            "Batch size must be 'auto' or a positive integer."
        ) from error
    if batch_size < 1:
        raise ValueError("Batch size must be 'auto' or a positive integer.")
    return batch_size


def _setting_label(key: str) -> str:
    replacements = {
        "asr": "ASR",
        "rms": "RMS",
        "vad": "VAD",
    }
    return " ".join(
        replacements.get(part, part.capitalize())
        for part in key.split("_")
    )


def _set_nested_value(
    target: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    current = target
    for component in path[:-1]:
        current = current.setdefault(component, {})
    current[path[-1]] = value


def _parse_setting_value(
    path: tuple[str, ...],
    value: Any,
    original: Any,
) -> Any:
    label = ".".join(path)
    if isinstance(original, bool):
        return bool(value)

    text = str(value).strip()
    if path[-1] == "batch_size":
        return _configured_batch_size(text)
    if original is None:
        return None if value is None or not text else text
    if isinstance(original, int):
        try:
            return int(text)
        except ValueError as error:
            raise ValueError(f"{label} must be an integer.") from error
    if isinstance(original, float):
        try:
            return float(text)
        except ValueError as error:
            raise ValueError(f"{label} must be a number.") from error
    return text


def _project_settings_from_values(
    values: dict[str, Any],
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    template = template or editable_project_settings()

    def parse_tree(
        raw: dict[str, Any],
        expected: dict[str, Any],
        path: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        parsed = {}
        for key, original in expected.items():
            raw_value = raw.get(key, original)
            if isinstance(original, dict):
                if not isinstance(raw_value, dict):
                    raise ValueError(f"{'.'.join((*path, key))} must be a group.")
                parsed[key] = parse_tree(
                    raw_value,
                    original,
                    (*path, key),
                )
            else:
                parsed[key] = _parse_setting_value(
                    (*path, key),
                    raw_value,
                    original,
                )
        return parsed

    settings = parse_tree(values, template)
    language = str(settings.get("language") or "").strip()
    if not language:
        raise ValueError("Language cannot be empty.")
    settings["language"] = language

    for section_name in ("transcription", "segment_transcription"):
        section = settings.get(section_name) or {}
        model = section.get("model")
        if section_name == "transcription" and not str(model or "").strip():
            raise ValueError("transcription.model cannot be empty.")
        device = section.get("device")
        if device is not None and str(device).lower() not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"{section_name}.device must be auto, cuda, or cpu.")
        compute_type = section.get("compute_type")
        if (
            compute_type is not None
            and str(compute_type).lower() not in COMPUTE_TYPES
        ):
            raise ValueError(
                f"{section_name}.compute_type must be one of: "
                f"{', '.join(COMPUTE_TYPES)}."
            )
        if int(section.get("beam_size", 1)) < 1:
            raise ValueError(f"{section_name}.beam_size must be positive.")
        if int(section.get("batch_size_max", 1)) < 1:
            raise ValueError(
                f"{section_name}.batch_size_max must be positive."
            )

    if "alignment" in settings:
        AlignmentSettings.from_value(settings["alignment"])
    export = settings.get("export") or {}
    if export:
        if int(export.get("sample_rate", 0)) < 1:
            raise ValueError("export.sample_rate must be positive.")
        if int(export.get("channels", 0)) < 1:
            raise ValueError("export.channels must be positive.")
        if int(export.get("bits_per_sample", 0)) != 16:
            raise ValueError("export.bits_per_sample must be 16.")
        extension = str(export.get("extension") or "").strip()
        if not extension.startswith(".") or len(extension) < 2:
            raise ValueError("export.extension must start with a dot.")
        export["extension"] = extension
    return settings


class CollapsibleSettingsGroup:
    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        *,
        expanded: bool,
    ) -> None:
        self.title = title
        self.expanded = expanded
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", pady=(0, 7))
        self.button = ttk.Button(
            self.frame,
            command=self.toggle,
            style="Heading.TButton",
        )
        self.button.pack(fill="x")
        self.body = ttk.Frame(self.frame, padding=(18, 10, 12, 12))
        self.body.columnconfigure(1, weight=1)
        self._render()

    def _render(self) -> None:
        self.button.configure(
            text=f"{'▼' if self.expanded else '▶'}  {self.title}"
        )
        if self.expanded:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()

    def toggle(self) -> None:
        self.expanded = not self.expanded
        self._render()

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self._render()


class ProjectSettingsDialog:
    def __init__(
        self,
        parent: tk.Misc,
        settings: dict[str, Any],
        *,
        title: str,
        submit_label: str,
    ) -> None:
        self.result: dict[str, Any] | None = None
        self.settings = settings
        self.fields: list[tuple[tuple[str, ...], tk.Variable, Any]] = []
        self.groups: list[CollapsibleSettingsGroup] = []

        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("860x760")
        self.window.minsize(680, 520)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)

        heading = ttk.Frame(self.window, padding=(24, 20, 24, 10))
        heading.grid(row=0, column=0, sticky="ew")
        heading.columnconfigure(0, weight=1)
        ttk.Label(
            heading,
            text=title,
            style="Heading.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            heading,
            text=(
                "Every editable project.json setting is available below. "
                "Expand a group to inspect or change it."
            ),
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        controls = ttk.Frame(heading)
        controls.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Button(
            controls,
            text="Expand all",
            command=lambda: self._set_all_groups(True),
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            controls,
            text="Collapse all",
            command=lambda: self._set_all_groups(False),
        ).pack(side="left")

        content = ttk.Frame(self.window)
        content.grid(row=1, column=0, sticky="nsew", padx=(24, 8))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(content, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            content,
            orient="vertical",
            command=self.canvas.yview,
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.form = ttk.Frame(self.canvas)
        self.form_window = self.canvas.create_window(
            (0, 0),
            window=self.form,
            anchor="nw",
        )
        self.form.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(
                self.form_window,
                width=event.width,
            ),
        )
        self.canvas.bind(
            "<Enter>",
            lambda _event: self.canvas.bind_all(
                "<MouseWheel>",
                self._on_mousewheel,
            ),
        )
        self.canvas.bind(
            "<Leave>",
            lambda _event: self.canvas.unbind_all("<MouseWheel>"),
        )

        general = {
            key: value
            for key, value in settings.items()
            if not isinstance(value, dict)
        }
        grouped = [
            (key, value)
            for key, value in settings.items()
            if isinstance(value, dict)
        ]
        if general:
            self._add_group(
                "General",
                general,
                (),
                expanded=True,
            )
        for index, (key, value) in enumerate(grouped):
            self._add_group(
                _setting_label(key),
                value,
                (key,),
                expanded=index == 0,
            )

        buttons = ttk.Frame(self.window, padding=(24, 12, 24, 20))
        buttons.grid(row=2, column=0, sticky="ew")
        ttk.Button(buttons, text="Cancel", command=self.cancel).pack(
            side="right",
        )
        ttk.Button(
            buttons,
            text=submit_label,
            style="Primary.TButton",
            command=self.accept,
        ).pack(side="right", padx=(0, 8))

        self.window.bind("<Escape>", lambda _event: self.cancel())
        self.window.update_idletasks()
        x = parent.winfo_rootx() + max(
            0,
            (parent.winfo_width() - self.window.winfo_width()) // 2,
        )
        y = parent.winfo_rooty() + max(
            0,
            (parent.winfo_height() - self.window.winfo_height()) // 2,
        )
        self.window.geometry(f"+{x}+{y}")
        self.window.wait_visibility()
        self.window.grab_set()
        self.window.wait_window()

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _set_all_groups(self, expanded: bool) -> None:
        for group in self.groups:
            group.set_expanded(expanded)
        self.form.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _add_group(
        self,
        title: str,
        values: dict[str, Any],
        path: tuple[str, ...],
        *,
        expanded: bool,
    ) -> None:
        group = CollapsibleSettingsGroup(
            self.form,
            title,
            expanded=expanded,
        )
        self.groups.append(group)
        self._add_fields(group.body, values, path)

    def _add_fields(
        self,
        parent: ttk.Frame,
        values: dict[str, Any],
        path: tuple[str, ...],
        *,
        depth: int = 0,
        row: int = 0,
    ) -> int:
        for key, value in values.items():
            field_path = (*path, key)
            if isinstance(value, dict):
                ttk.Label(
                    parent,
                    text=_setting_label(key),
                    style="FieldName.TLabel",
                ).grid(
                    row=row,
                    column=0,
                    columnspan=2,
                    sticky="w",
                    padx=(depth * 14, 0),
                    pady=(10, 3),
                )
                row += 1
                row = self._add_fields(
                    parent,
                    value,
                    field_path,
                    depth=depth + 1,
                    row=row,
                )
                continue

            ttk.Label(
                parent,
                text=_setting_label(key),
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(depth * 14, 18),
                pady=4,
            )
            variable, widget = self._setting_widget(
                parent,
                field_path,
                value,
            )
            widget.grid(row=row, column=1, sticky="ew", pady=4)
            self.fields.append((field_path, variable, value))
            row += 1
        return row

    def _setting_widget(
        self,
        parent: ttk.Frame,
        path: tuple[str, ...],
        value: Any,
    ) -> tuple[tk.Variable, tk.Widget]:
        if isinstance(value, bool):
            variable = tk.BooleanVar(value=value)
            return variable, ttk.Checkbutton(parent, variable=variable)

        display_value = "" if value is None else str(value)
        variable = tk.StringVar(value=display_value)
        key = path[-1]
        choices: tuple[str, ...] | None = None
        readonly = False
        if key == "device":
            choices = ("", "auto", "cuda", "cpu")
            readonly = True
        elif key == "compute_type":
            choices = ("", *COMPUTE_TYPES)
            readonly = True
        elif key == "batch_size":
            choices = ("auto", "1", "2", "4", "8", "12", "16", "24", "32")
        elif key in {"policy", "duplicate_line_policy"}:
            choices = ("review", "weak_order", "reuse")
            readonly = True
        elif key == "bits_per_sample":
            choices = ("16",)
            readonly = True

        if choices is not None:
            return variable, ttk.Combobox(
                parent,
                textvariable=variable,
                values=choices,
                state="readonly" if readonly else "normal",
            )
        return variable, ttk.Entry(parent, textvariable=variable)

    def accept(self) -> None:
        raw_values: dict[str, Any] = {}
        for path, variable, _original in self.fields:
            _set_nested_value(raw_values, path, variable.get())
        try:
            self.result = _project_settings_from_values(
                raw_values,
                self.settings,
            )
        except ValueError as error:
            messagebox.showerror(
                "Invalid project settings",
                str(error),
                parent=self.window,
            )
            return
        self.window.destroy()

    def cancel(self) -> None:
        self.result = None
        self.window.destroy()


class DialogueReviewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1440x850")
        self.root.minsize(1050, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.player = AudioPlayer()
        self.project_dir: Path | None = None
        self.project: dict[str, Any] | None = None
        self.review_path: Path | None = None
        self.review_data: dict[str, Any] | None = None
        self.base_segments_by_session: dict[str, list[dict[str, Any]]] = {}
        self.selected_line_id: str | None = None
        self.status_filter = tk.StringVar(value="All")
        self.line_sort_column = "status"
        self.line_sort_descending = False
        self.status_text = tk.StringVar(value="")
        self._worker_messages: queue.Queue[tuple[str, Any]] | None = None
        self._log_text: tk.Text | None = None
        self._progress: ttk.Progressbar | None = None
        self._cancel_event: threading.Event | None = None
        self._cancel_button: ttk.Button | None = None
        self._cancel_status: tk.StringVar | None = None
        self._worker_thread: threading.Thread | None = None
        self._candidate_transcription_runtime: dict[str, Any] = {}
        self.selected_candidate_id: str | None = None
        self.selected_base_segment_id: str | None = None
        self._candidate_line_id: str | None = None
        self.candidate_waveform: CandidateWaveformView | None = None
        self._open_candidate_roots: set[str] = set()
        self._mapping_project: dict[str, Any] | None = None
        self._mapping_source_data: dict[str, Any] | None = None
        self._mapping_inventory: dict[Path, dict[str, Any]] = {}
        self._mapping_reprocessing = False
        self._mapping_session_id: str | None = None
        self._mapping_row_values: dict[str, str] = {}
        self._mapping_line_id_values: dict[str, list[str]] = {}

        self._configure_styles()
        self.show_start()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"))
        style.configure("Heading.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Heading.TButton", font=("Segoe UI", 11, "bold"))
        style.configure("FieldName.TLabel", font=("Segoe UI", 9, "bold"))
        style.configure("Muted.TLabel", foreground="#475569")
        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"))

    def close(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self.candidate_waveform is not None:
            self.candidate_waveform.dispose()
            self.candidate_waveform = None
        self.player.close()
        self.root.destroy()

    def _clear(self) -> None:
        self.player.stop()
        if self.candidate_waveform is not None:
            self.candidate_waveform.dispose()
            self.candidate_waveform = None
        for child in self.root.winfo_children():
            child.destroy()
        self._log_text = None
        self._progress = None
        self._cancel_button = None
        self._cancel_status = None

    def show_start(self) -> None:
        self._clear()
        self.project_dir = None
        self.project = None
        self.review_path = None
        self.review_data = None
        self.base_segments_by_session = {}
        self.selected_line_id = None
        self.selected_candidate_id = None
        self.selected_base_segment_id = None
        self._candidate_line_id = None
        self._mapping_project = None
        self._mapping_source_data = None
        self._mapping_inventory = {}
        self._mapping_reprocessing = False
        self._mapping_session_id = None
        self._mapping_row_values = {}
        self._mapping_line_id_values = {}

        outer = ttk.Frame(self.root, padding=40)
        outer.pack(fill="both", expand=True)
        card = ttk.Frame(outer, padding=48)
        card.place(relx=0.5, rely=0.45, anchor="center")
        ttk.Label(card, text=APP_TITLE, style="Title.TLabel").pack(pady=(0, 10))
        ttk.Label(
            card,
            text="Split recordings, align takes, review candidates, and export.",
            style="Muted.TLabel",
        ).pack(pady=(0, 32))
        ttk.Button(
            card,
            text="Open Project",
            style="Primary.TButton",
            command=self.choose_existing_project,
            width=40,
        ).pack(ipady=8, pady=7)
        ttk.Button(
            card,
            text="Create or Reprocess Project",
            style="Primary.TButton",
            command=self.choose_new_project,
            width=40,
        ).pack(ipady=8, pady=7)
        ttk.Button(
            card,
            text="Refresh Segments from Updated Audio",
            style="Primary.TButton",
            command=self.choose_audio_refresh,
            width=40,
        ).pack(ipady=8, pady=7)

    def choose_existing_project(self) -> None:
        selected = filedialog.askdirectory(title="Select project directory")
        if not selected:
            return
        try:
            self.open_project(Path(selected))
        except Exception as error:
            messagebox.showerror(
                "Cannot open project",
                f"{error}\n\nSelect a directory containing {REVIEW_FILE_NAME}.",
                parent=self.root,
            )
            self.show_start()

    def choose_new_project(self) -> None:
        project_dir_value = filedialog.askdirectory(
            title="Select a new or existing project directory",
            mustexist=False,
        )
        if not project_dir_value:
            return
        project_dir = Path(project_dir_value).resolve()
        if (project_dir / "project.json").is_file():
            try:
                _loaded_dir, project = load_project(
                    project_file_from_arg(project_dir)
                )
            except Exception as error:
                messagebox.showerror(
                    "Cannot read project settings",
                    str(error),
                    parent=self.root,
                )
                return
            project_settings = self.ask_project_settings(
                editable_project_settings(project),
                reprocessing=True,
            )
            if project_settings is None:
                return
            self.run_existing_project(project_dir, project_settings)
            return

        workbook = filedialog.askopenfilename(
            title="Select lines spreadsheet",
            filetypes=[
                ("Dialogue spreadsheets", "*.xlsm *.xlsx *.ods"),
                ("All files", "*.*"),
            ],
            initialdir=str(project_dir.parent),
        )
        if not workbook:
            return
        audio_dir = filedialog.askdirectory(
            title="Select directory containing recorded WAV files",
            initialdir=str(Path(workbook).parent),
        )
        if not audio_dir:
            return
        project_settings = self.ask_project_settings(
            editable_project_settings(),
            reprocessing=False,
        )
        if project_settings is None:
            return
        self.run_new_project(
            workbook_path=Path(workbook),
            audio_dir=Path(audio_dir),
            project_dir=project_dir,
            project_settings=project_settings,
        )

    def choose_audio_refresh(self) -> None:
        selected = filedialog.askdirectory(
            title="Select project with updated source audio"
        )
        if not selected:
            return
        project_dir = Path(selected).resolve()
        try:
            project_file_from_arg(project_dir)
            if not (project_dir / "segments_manifest.json").is_file():
                raise FileNotFoundError(
                    f"Missing segment manifest: "
                    f"{project_dir / 'segments_manifest.json'}"
                )
            if not (project_dir / REVIEW_FILE_NAME).is_file():
                raise FileNotFoundError(
                    f"Missing review data: {project_dir / REVIEW_FILE_NAME}"
                )
        except Exception as error:
            messagebox.showerror(
                "Cannot refresh project audio",
                str(error),
                parent=self.root,
            )
            return

        confirmed = messagebox.askokcancel(
            "Refresh segments from updated audio?",
            (
                "Continue only if the recordings contain the same spoken "
                "performances at the same timing (for example, only volume "
                "was adjusted).\n\n"
                "Existing segment WAV files will be replaced from their "
                "stored sample ranges. Transcripts, alignment results, and "
                "review selections will be kept."
            ),
            parent=self.root,
        )
        if not confirmed:
            return
        self.run_audio_refresh(project_dir)

    def ask_project_settings(
        self,
        settings: dict[str, Any],
        *,
        reprocessing: bool,
    ) -> dict[str, Any] | None:
        return ProjectSettingsDialog(
            self.root,
            settings,
            title=(
                "Reprocess Project Settings"
                if reprocessing
                else "New Project Settings"
            ),
            submit_label="Reprocess Project" if reprocessing else "Create Project",
        ).result

    def show_session_mapping_review(
        self,
        project_dir: Path,
        *,
        reprocessing: bool,
    ) -> None:
        """Pause processing so every recording-to-sheet mapping can be confirmed."""

        loaded_dir, project = load_project(project_file_from_arg(project_dir))
        source_data = load_source_data(loaded_dir, project)
        inventory = inventory_by_path(loaded_dir, project)
        sessions = list(project.get("sessions") or [])
        if not sessions:
            raise ValueError("The audio inventory contains no recording sessions.")

        self._clear()
        self.project_dir = loaded_dir
        self._mapping_project = copy.deepcopy(project)
        session_ids: set[str] = set()
        for session in self._mapping_project.get("sessions") or []:
            session_id = str(session.get("id") or "")
            if not session_id:
                raise ValueError("An inventoried audio session has no ID.")
            if session_id in session_ids:
                raise ValueError(f"Duplicate audio session ID: {session_id}")
            session_ids.add(session_id)
            session["sheets"] = list(
                dict.fromkeys(str(sheet) for sheet in session.get("sheets") or [])
            )
        self._mapping_source_data = source_data
        self._mapping_inventory = inventory
        self._mapping_reprocessing = reprocessing
        self._mapping_session_id = None
        self._mapping_row_values = {
            str(session["id"]): _format_excel_rows(session.get("excel_rows"))
            for session in sessions
        }
        self._mapping_line_id_values = {
            str(session["id"]): [str(value) for value in session.get("line_ids") or []]
            for session in sessions
        }

        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)
        heading = ttk.Frame(outer)
        heading.pack(fill="x", pady=(0, 6))
        ttk.Label(
            heading,
            text="Review Audio / Script Matching",
            style="Title.TLabel",
        ).pack(side="left")
        ttk.Label(
            outer,
            text=(
                "Confirm which workbook sheets are spoken in each recording. "
                "Processing will begin only after these mappings are saved."
            ),
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 14))

        panes = ttk.Panedwindow(outer, orient="horizontal")
        panes.pack(fill="both", expand=True)
        recordings = ttk.Labelframe(panes, text="Audio recordings", padding=8)
        mapping = ttk.Labelframe(panes, text="Selected recording", padding=12)
        panes.add(recordings, weight=2)
        panes.add(mapping, weight=3)

        recording_columns = ("enabled", "audio", "duration", "sheets", "state")
        self.mapping_recording_tree = ttk.Treeview(
            recordings,
            columns=recording_columns,
            show="headings",
            selectmode="browse",
        )
        for column, label, width, anchor in (
            ("enabled", "Use", 55, "center"),
            ("audio", "Audio file", 210, "w"),
            ("duration", "Length", 70, "e"),
            ("sheets", "Script sheets", 280, "w"),
            ("state", "Mapping", 85, "center"),
        ):
            self.mapping_recording_tree.heading(column, text=label)
            self.mapping_recording_tree.column(
                column,
                width=width,
                minwidth=45,
                anchor=anchor,
                stretch=column in {"audio", "sheets"},
            )
        recording_scroll = ttk.Scrollbar(
            recordings,
            orient="vertical",
            command=self.mapping_recording_tree.yview,
        )
        self.mapping_recording_tree.configure(yscrollcommand=recording_scroll.set)
        self.mapping_recording_tree.pack(side="left", fill="both", expand=True)
        recording_scroll.pack(side="right", fill="y")
        self.mapping_recording_tree.tag_configure("review", background="#fef3c7")
        self.mapping_recording_tree.tag_configure("disabled", foreground="#64748b")
        self.mapping_recording_tree.bind(
            "<<TreeviewSelect>>",
            self._mapping_recording_selected,
        )

        self.mapping_audio_name = tk.StringVar(value="")
        self.mapping_audio_details = tk.StringVar(value="")
        ttk.Label(
            mapping,
            textvariable=self.mapping_audio_name,
            style="Heading.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            mapping,
            textvariable=self.mapping_audio_details,
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 10))

        self.mapping_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            mapping,
            text="Use this recording during processing",
            variable=self.mapping_enabled,
            command=self._mapping_enabled_changed,
        ).pack(anchor="w", pady=(0, 10))

        row_filter = ttk.Frame(mapping)
        row_filter.pack(fill="x", pady=(0, 10))
        ttk.Label(row_filter, text="Excel rows (optional):").pack(side="left")
        self.mapping_rows = tk.StringVar(value="")
        row_entry = ttk.Entry(row_filter, textvariable=self.mapping_rows, width=30)
        row_entry.pack(side="left", padx=(8, 8))
        row_entry.bind("<FocusOut>", self._mapping_rows_changed)
        row_entry.bind("<Return>", self._mapping_rows_changed)
        ttk.Label(
            row_filter,
            text="Example: 12, 18-24. Blank includes every dialogue row.",
            style="Muted.TLabel",
        ).pack(side="left")

        self.mapping_keep_line_ids = tk.BooleanVar(value=False)
        self.mapping_line_id_label = tk.StringVar(value="")
        self.mapping_line_id_check = ttk.Checkbutton(
            mapping,
            textvariable=self.mapping_line_id_label,
            variable=self.mapping_keep_line_ids,
            command=self._mapping_line_ids_changed,
        )
        self.mapping_line_id_check.pack(anchor="w", pady=(0, 10))

        ttk.Label(
            mapping,
            text="Workbook sheets (double-click or press Space to add/remove):",
            style="FieldName.TLabel",
        ).pack(anchor="w", pady=(0, 5))
        sheet_frame = ttk.Frame(mapping)
        sheet_frame.pack(fill="both", expand=True)
        sheet_columns = ("mapped", "order", "sheet", "lines", "voice")
        self.mapping_sheet_tree = ttk.Treeview(
            sheet_frame,
            columns=sheet_columns,
            show="headings",
            selectmode="browse",
        )
        for column, label, width, anchor in (
            ("mapped", "Mapped", 70, "center"),
            ("order", "Order", 55, "center"),
            ("sheet", "Sheet", 170, "w"),
            ("lines", "Lines", 60, "e"),
            ("voice", "Voice / section", 300, "w"),
        ):
            self.mapping_sheet_tree.heading(column, text=label)
            self.mapping_sheet_tree.column(
                column,
                width=width,
                minwidth=45,
                anchor=anchor,
                stretch=column in {"sheet", "voice"},
            )
        sheet_scroll = ttk.Scrollbar(
            sheet_frame,
            orient="vertical",
            command=self.mapping_sheet_tree.yview,
        )
        self.mapping_sheet_tree.configure(yscrollcommand=sheet_scroll.set)
        self.mapping_sheet_tree.pack(side="left", fill="both", expand=True)
        sheet_scroll.pack(side="right", fill="y")
        self.mapping_sheet_tree.tag_configure("mapped", background="#dcfce7")
        self.mapping_sheet_tree.tag_configure("missing", background="#fee2e2")
        self.mapping_sheet_tree.bind("<Double-1>", self._mapping_sheet_double_click)
        self.mapping_sheet_tree.bind("<space>", self._mapping_sheet_space)
        self.mapping_sheet_tree.bind(
            "<<TreeviewSelect>>",
            self._update_mapping_sheet_actions,
        )

        sheet_controls = ttk.Frame(mapping)
        sheet_controls.pack(fill="x", pady=(8, 0))
        self.mapping_sheet_action_label = tk.StringVar(value="Add")
        self.mapping_sheet_action_button = ttk.Button(
            sheet_controls,
            textvariable=self.mapping_sheet_action_label,
            command=self._toggle_selected_mapping_sheet,
        )
        self.mapping_sheet_action_button.pack(side="left")
        self.mapping_all_sheets_action_label = tk.StringVar(value="Add All")
        self.mapping_all_sheets_action_button = ttk.Button(
            sheet_controls,
            textvariable=self.mapping_all_sheets_action_label,
            command=self._toggle_all_mapping_sheets,
        )
        self.mapping_all_sheets_action_button.pack(side="left", padx=(8, 0))
        ttk.Button(
            sheet_controls,
            text="Move Up",
            command=lambda: self._move_mapping_sheet(-1),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            sheet_controls,
            text="Move Down",
            command=lambda: self._move_mapping_sheet(1),
        ).pack(side="left", padx=(8, 0))
        self.mapping_line_summary = tk.StringVar(value="")
        ttk.Label(
            sheet_controls,
            textvariable=self.mapping_line_summary,
            style="Muted.TLabel",
        ).pack(side="right")

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(14, 0))
        ttk.Label(
            footer,
            text=(
                "Yellow recordings were inferred ambiguously and especially need "
                "review. Disabled recordings are kept in the project but skipped."
            ),
            style="Muted.TLabel",
        ).pack(side="left")
        ttk.Button(
            footer,
            text="Back to Start",
            command=self.show_start,
        ).pack(side="right")
        ttk.Button(
            footer,
            text="Confirm Mappings and Process",
            style="Primary.TButton",
            command=self._confirm_session_mappings,
        ).pack(side="right", padx=(0, 8), ipady=3)

        self._refresh_mapping_recordings()
        first_session_id = str(sessions[0]["id"])
        self.mapping_recording_tree.selection_set(first_session_id)
        self.mapping_recording_tree.focus(first_session_id)
        self.mapping_recording_tree.see(first_session_id)
        self._show_mapping_session(first_session_id)

    def _mapping_sessions(self) -> list[dict[str, Any]]:
        if self._mapping_project is None:
            return []
        return list(self._mapping_project.get("sessions") or [])

    def _mapping_session(self, session_id: str | None = None) -> dict[str, Any] | None:
        wanted = session_id or self._mapping_session_id
        return next(
            (
                session
                for session in self._mapping_sessions()
                if str(session.get("id")) == wanted
            ),
            None,
        )

    def _remember_mapping_rows(self) -> None:
        if self._mapping_session_id and hasattr(self, "mapping_rows"):
            value = self.mapping_rows.get()
            self._mapping_row_values[self._mapping_session_id] = value
            session = self._mapping_session()
            if session is not None:
                try:
                    session["excel_rows"] = _parse_excel_rows(value)
                except ValueError:
                    pass

    def _refresh_mapping_recordings(self) -> None:
        if self._mapping_project is None or self._mapping_source_data is None:
            return
        selected = self._mapping_session_id
        self.mapping_recording_tree.delete(*self.mapping_recording_tree.get_children())
        for session in self._mapping_sessions():
            session_id = str(session["id"])
            audio_path = resolve_project_path(
                self.project_dir or Path.cwd(),
                str(session["audio"]),
            )
            metadata = self._mapping_inventory.get(audio_path.resolve(), {})
            duration = metadata.get("duration_seconds")
            duration_text = f"{float(duration) / 60:.1f} min" if duration is not None else "—"
            sheets = list(session.get("sheets") or [])
            enabled = bool(session.get("enabled", True))
            state = "Review" if session.get("needs_mapping_review") else "Inferred"
            if not enabled:
                state = "Skipped"
            tags = (
                "disabled" if not enabled else "review" if session.get("needs_mapping_review") else "",
            )
            self.mapping_recording_tree.insert(
                "",
                "end",
                iid=session_id,
                values=(
                    "Yes" if enabled else "No",
                    audio_path.name,
                    duration_text,
                    " → ".join(str(sheet) for sheet in sheets) or "(none)",
                    state,
                ),
                tags=tuple(tag for tag in tags if tag),
            )
        if selected and self.mapping_recording_tree.exists(selected):
            self.mapping_recording_tree.selection_set(selected)

    def _mapping_recording_selected(self, _event: tk.Event[Any]) -> None:
        selection = self.mapping_recording_tree.selection()
        if not selection:
            return
        session_id = str(selection[0])
        if session_id == self._mapping_session_id:
            return
        self._remember_mapping_rows()
        self._show_mapping_session(session_id)

    def _show_mapping_session(self, session_id: str) -> None:
        session = self._mapping_session(session_id)
        if session is None or self._mapping_source_data is None:
            return
        self._mapping_session_id = session_id
        audio_path = resolve_project_path(
            self.project_dir or Path.cwd(),
            str(session["audio"]),
        )
        metadata = self._mapping_inventory.get(audio_path.resolve(), {})
        detail_parts = [str(audio_path)]
        if metadata:
            detail_parts.append(
                f"{float(metadata.get('duration_seconds') or 0):.1f} s, "
                f"{int(metadata.get('sample_rate') or 0)} Hz, "
                f"{int(metadata.get('channels') or 0)} channel(s)"
            )
        detail_parts.append(f"pass: {session.get('pass') or 'main'}")
        self.mapping_audio_name.set(audio_path.name)
        self.mapping_audio_details.set("  •  ".join(detail_parts))
        self.mapping_enabled.set(bool(session.get("enabled", True)))
        self.mapping_rows.set(self._mapping_row_values.get(session_id, ""))
        exact_id_count = len(self._mapping_line_id_values.get(session_id, []))
        self.mapping_keep_line_ids.set(bool(session.get("line_ids")))
        self.mapping_line_id_label.set(
            f"Keep {exact_id_count} existing exact line ID filter(s)"
            if exact_id_count
            else "No exact line ID filters"
        )
        self.mapping_line_id_check.configure(
            state="normal" if exact_id_count else "disabled"
        )
        self._refresh_mapping_sheet_tree()

    def _mapping_enabled_changed(self) -> None:
        session = self._mapping_session()
        if session is None:
            return
        session["enabled"] = bool(self.mapping_enabled.get())
        self._refresh_mapping_recordings()

    def _mapping_rows_changed(self, _event: tk.Event[Any]) -> None:
        session = self._mapping_session()
        if session is None or self._mapping_session_id is None:
            return
        value = self.mapping_rows.get()
        self._mapping_row_values[self._mapping_session_id] = value
        try:
            session["excel_rows"] = _parse_excel_rows(value)
        except ValueError:
            return
        self._refresh_mapping_sheet_tree()

    def _mapping_line_ids_changed(self) -> None:
        session = self._mapping_session()
        if session is None or self._mapping_session_id is None:
            return
        session["line_ids"] = (
            list(self._mapping_line_id_values.get(self._mapping_session_id, []))
            if self.mapping_keep_line_ids.get()
            else []
        )
        self._refresh_mapping_sheet_tree()

    def _refresh_mapping_sheet_tree(self) -> None:
        session = self._mapping_session()
        if session is None or self._mapping_source_data is None:
            return
        selected_sheet = next(iter(self.mapping_sheet_tree.selection()), None)
        mapped = [str(sheet) for sheet in session.get("sheets") or []]
        sheet_by_name = {
            str(sheet["name"]): sheet
            for sheet in self._mapping_source_data.get("sheets", [])
        }
        display_order = [
            *mapped,
            *[sheet for sheet in sheet_by_name if sheet not in mapped],
        ]
        self.mapping_sheet_tree.delete(*self.mapping_sheet_tree.get_children())
        for sheet_name in display_order:
            sheet = sheet_by_name.get(sheet_name)
            is_mapped = sheet_name in mapped
            is_missing = sheet is None
            self.mapping_sheet_tree.insert(
                "",
                "end",
                iid=sheet_name,
                values=(
                    "Yes" if is_mapped else "",
                    mapped.index(sheet_name) + 1 if is_mapped else "",
                    sheet_name,
                    int(sheet.get("line_count") or 0) if sheet else 0,
                    (
                        str(sheet.get("voice_header") or "")
                        if sheet
                        else "Missing from current workbook — remove this mapping"
                    ),
                ),
                tags=("missing",) if is_missing else ("mapped",) if is_mapped else (),
            )
        if selected_sheet and self.mapping_sheet_tree.exists(selected_sheet):
            self.mapping_sheet_tree.selection_set(selected_sheet)
        line_count = len(lines_for_session(self._mapping_source_data, session))
        exact_ids = len(session.get("line_ids") or [])
        summary = f"{line_count} mapped line(s)"
        if exact_ids:
            summary += f"; {exact_ids} exact line ID filter(s) retained"
        self.mapping_line_summary.set(summary)
        self._update_mapping_sheet_actions()
        self._refresh_mapping_recordings()

    def _update_mapping_sheet_actions(
        self,
        _event: tk.Event[Any] | None = None,
    ) -> None:
        session = self._mapping_session()
        if session is None or self._mapping_source_data is None:
            return
        selection = self.mapping_sheet_tree.selection()
        selected_sheet = str(selection[0]) if selection else None
        mapped = [str(sheet) for sheet in session.get("sheets") or []]
        available = [
            str(sheet["name"])
            for sheet in self._mapping_source_data.get("sheets", [])
        ]
        selected_action, all_action = _mapping_sheet_action_names(
            mapped_sheets=mapped,
            available_sheets=available,
            selected_sheet=selected_sheet,
        )
        self.mapping_sheet_action_label.set(selected_action)
        self.mapping_sheet_action_button.configure(
            state="normal" if selected_sheet else "disabled"
        )
        self.mapping_all_sheets_action_label.set(all_action)
        self.mapping_all_sheets_action_button.configure(
            state="normal" if available or mapped else "disabled"
        )

    def _mapping_sheet_double_click(self, event: tk.Event[Any]) -> None:
        item = self.mapping_sheet_tree.identify_row(event.y)
        if item:
            self.mapping_sheet_tree.selection_set(item)
            self._toggle_selected_mapping_sheet()

    def _mapping_sheet_space(self, _event: tk.Event[Any]) -> str:
        self._toggle_selected_mapping_sheet()
        return "break"

    def _toggle_selected_mapping_sheet(self) -> None:
        session = self._mapping_session()
        selection = self.mapping_sheet_tree.selection()
        if session is None or not selection:
            return
        sheet_name = str(selection[0])
        mapped = [str(sheet) for sheet in session.get("sheets") or []]
        if sheet_name in mapped:
            mapped.remove(sheet_name)
        else:
            mapped.append(sheet_name)
        session["sheets"] = mapped
        self._refresh_mapping_sheet_tree()

    def _toggle_all_mapping_sheets(self) -> None:
        session = self._mapping_session()
        if session is None or self._mapping_source_data is None:
            return
        mapped = [str(sheet) for sheet in session.get("sheets") or []]
        available = [
            str(sheet["name"])
            for sheet in self._mapping_source_data.get("sheets", [])
        ]
        _selected_action, all_action = _mapping_sheet_action_names(
            mapped_sheets=mapped,
            available_sheets=available,
            selected_sheet=None,
        )
        if all_action == "Remove All":
            session["sheets"] = []
        else:
            session["sheets"] = [
                *mapped,
                *[sheet for sheet in available if sheet not in mapped],
            ]
        self._refresh_mapping_sheet_tree()

    def _move_mapping_sheet(self, direction: int) -> None:
        session = self._mapping_session()
        selection = self.mapping_sheet_tree.selection()
        if session is None or not selection:
            return
        sheet_name = str(selection[0])
        mapped = [str(sheet) for sheet in session.get("sheets") or []]
        if sheet_name not in mapped:
            return
        current = mapped.index(sheet_name)
        destination = current + direction
        if destination < 0 or destination >= len(mapped):
            return
        mapped[current], mapped[destination] = mapped[destination], mapped[current]
        session["sheets"] = mapped
        self._refresh_mapping_sheet_tree()
        self.mapping_sheet_tree.selection_set(sheet_name)
        self.mapping_sheet_tree.see(sheet_name)

    def _confirm_session_mappings(self) -> None:
        if self._mapping_project is None or self._mapping_source_data is None:
            return
        self._remember_mapping_rows()
        try:
            for session in self._mapping_sessions():
                session_id = str(session["id"])
                session["excel_rows"] = _parse_excel_rows(
                    self._mapping_row_values.get(session_id, "")
                )
            _validate_session_mappings(
                self._mapping_sessions(),
                self._mapping_source_data,
            )
        except ValueError as error:
            messagebox.showerror(
                "Cannot save mappings",
                str(error),
                parent=self.root,
            )
            return

        for session in self._mapping_sessions():
            session["needs_mapping_review"] = False
        project_dir = self.project_dir
        if project_dir is None:
            return
        try:
            write_json(project_dir / "project.json", self._mapping_project)
        except Exception as error:
            messagebox.showerror(
                "Cannot save mappings",
                str(error),
                parent=self.root,
            )
            return
        reprocessing = self._mapping_reprocessing
        self._mapping_project = None
        self._mapping_source_data = None
        self._mapping_inventory = {}
        self._mapping_session_id = None
        self._mapping_row_values = {}
        self._mapping_line_id_values = {}
        self._continue_project_processing(
            project_dir,
            reprocessing=reprocessing,
        )

    def open_project(self, project_dir: Path) -> None:
        project_dir = project_dir.resolve()
        review_path = project_dir / REVIEW_FILE_NAME
        review_data = load_line_review(review_path)
        project_file = project_file_from_arg(project_dir)
        loaded_dir, project = load_project(project_file)
        manifest_path = loaded_dir / "segments_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        self.project_dir = loaded_dir
        self.project = project
        self.review_path = review_path
        self.review_data = review_data
        self.base_segments_by_session = {
            str(session.get("session_id") or ""): list(
                session.get("segments") or []
            )
            for session in manifest.get("sessions") or []
            if str(session.get("session_id") or "")
        }
        self.selected_line_id = (
            review_data["lines"][0]["line_id"] if review_data["lines"] else None
        )
        self.show_review()

    def run_new_project(
        self,
        *,
        workbook_path: Path,
        audio_dir: Path,
        project_dir: Path,
        project_settings: dict[str, Any],
    ) -> None:
        self.show_progress(project_dir, title="Building project inventory")

        def work() -> Path:
            check_processing_cancelled()
            print("Creating project and inventorying source audio.", flush=True)
            create_project(
                workbook_path=workbook_path,
                audio_dir=audio_dir,
                project_dir=project_dir,
                project_settings=project_settings,
            )
            check_processing_cancelled()
            print("Audio inventory complete. Review session mappings next.", flush=True)
            return project_dir.resolve()

        self._start_worker(
            work,
            lambda path: self._inventory_finished(path, reprocessing=False),
        )

    def run_existing_project(
        self,
        project_dir: Path,
        project_settings: dict[str, Any],
    ) -> None:
        """Rerun an existing project without recreating or clearing it."""
        self.show_progress(project_dir, title="Preparing project for reprocessing")

        def work() -> Path:
            check_processing_cancelled()
            loaded_dir, project = load_project(
                project_file_from_arg(project_dir)
            )
            apply_project_settings(project, project_settings)
            write_json(loaded_dir / "project.json", project)
            print(
                "[existing project] Updated settings while preserving mappings "
                "and cached artifacts.",
                flush=True,
            )
            check_processing_cancelled()
            print("Audio inventory is ready. Review session mappings next.", flush=True)
            return loaded_dir

        self._start_worker(
            work,
            lambda path: self._inventory_finished(path, reprocessing=True),
        )

    def _run_pipeline_stages(
        self,
        project_dir: Path,
        project: dict[str, Any],
        *,
        reprocessing: bool,
    ) -> None:
        check_processing_cancelled()
        print("[phase 1/4] Transcribing source recordings", flush=True)
        transcribe_project(project_dir=project_dir, project=project)
        check_processing_cancelled()
        print("[phase 2/4] Segmenting recordings", flush=True)
        segment_project(project_dir=project_dir, project=project)
        check_processing_cancelled()
        print("[phase 3/4] Transcribing temporary segments", flush=True)
        transcribe_segments_project(project_dir=project_dir, project=project)
        check_processing_cancelled()
        print("[phase 4/4] Aligning segments to script lines", flush=True)
        align_project(project_dir=project_dir, project=project)
        check_processing_cancelled()
        print(
            "Project reprocessing complete." if reprocessing else "Pipeline complete.",
            flush=True,
        )

    def _inventory_finished(self, project_dir: Path, *, reprocessing: bool) -> None:
        try:
            self.show_session_mapping_review(
                project_dir,
                reprocessing=reprocessing,
            )
        except Exception as error:
            self._pipeline_failed(error)

    def _continue_project_processing(
        self,
        project_dir: Path,
        *,
        reprocessing: bool,
    ) -> None:
        self.show_progress(
            project_dir,
            title="Reprocessing project" if reprocessing else "Processing project",
        )

        def work() -> Path:
            loaded_dir, project = load_project(project_file_from_arg(project_dir))
            self._run_pipeline_stages(
                loaded_dir,
                project,
                reprocessing=reprocessing,
            )
            return loaded_dir

        self._start_worker(work, self._pipeline_finished)

    def run_audio_refresh(self, project_dir: Path) -> None:
        self.show_progress(
            project_dir,
            title="Refreshing segments from updated audio",
        )

        def work() -> tuple[Path, dict[str, Any]]:
            check_processing_cancelled()
            loaded_dir, project = load_project(
                project_file_from_arg(project_dir)
            )
            print(
                "Refreshing source inventory and re-cutting stored base and "
                "derived segments.",
                flush=True,
            )
            result = refresh_project_audio(
                project_dir=loaded_dir,
                project=project,
            )
            check_processing_cancelled()
            print(
                "Audio refresh complete: "
                f"{result['changed_source_count']} changed source(s), "
                f"{result['total_segment_count']} segment file(s).",
                flush=True,
            )
            return loaded_dir, result

        self._start_worker(work, self._audio_refresh_finished)

    def show_progress(self, project_dir: Path, *, title: str) -> None:
        self._clear()
        outer = ttk.Frame(self.root, padding=30)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text=title, style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            outer,
            text=str(project_dir.resolve()),
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 18))
        self._progress = ttk.Progressbar(outer, mode="indeterminate")
        self._progress.pack(fill="x", pady=(0, 18))
        self._progress.start(12)
        self._log_text = tk.Text(
            outer,
            wrap="word",
            state="disabled",
            font=("Cascadia Mono", 10),
            background="#0f172a",
            foreground="#e2e8f0",
            insertbackground="#e2e8f0",
        )
        self._log_text.pack(fill="both", expand=True)
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(12, 0))
        self._cancel_status = tk.StringVar(value="")
        ttk.Label(
            controls,
            textvariable=self._cancel_status,
            style="Muted.TLabel",
        ).pack(side="left")
        self._cancel_button = ttk.Button(
            controls,
            text="Cancel",
            command=self.cancel_processing,
        )
        self._cancel_button.pack(side="right")
        self._cancel_event = threading.Event()

    def cancel_processing(self) -> None:
        event = self._cancel_event
        if event is None or event.is_set():
            return
        event.set()
        if self._cancel_button is not None:
            self._cancel_button.configure(state="disabled")
        if self._cancel_status is not None:
            self._cancel_status.set("Cancelling at the next safe point…")
        self._append_log(
            "\nCancellation requested; finishing the active safe unit.\n"
        )

    def _start_worker(
        self,
        function: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._worker_messages = messages
        cancel_event = self._cancel_event or threading.Event()
        self._cancel_event = cancel_event

        def worker() -> None:
            writer = QueueWriter(messages)
            try:
                with (
                    cancellation_scope(cancel_event),
                    contextlib.redirect_stdout(writer),
                    contextlib.redirect_stderr(writer),
                ):
                    result = function()
                    check_processing_cancelled()
                messages.put(("done", (on_success, result)))
            except ProcessingCancelled:
                messages.put(("cancelled", None))
            except Exception as error:
                if cancel_event.is_set():
                    messages.put(("cancelled", None))
                else:
                    messages.put(("log", traceback.format_exc()))
                    messages.put(("error", error))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()
        self.root.after(100, self._poll_worker)

    def _poll_worker(self) -> None:
        messages = self._worker_messages
        if messages is None:
            return
        finished = False
        while True:
            try:
                kind, payload = messages.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(str(payload))
            elif kind == "done":
                callback, result = payload
                self._stop_progress()
                self._worker_messages = None
                self._cancel_event = None
                self._worker_thread = None
                callback(result)
                finished = True
            elif kind == "error":
                self._stop_progress()
                self._worker_messages = None
                self._cancel_event = None
                self._worker_thread = None
                self._pipeline_failed(payload)
                finished = True
            elif kind == "cancelled":
                self._stop_progress()
                self._worker_messages = None
                self._cancel_event = None
                self._worker_thread = None
                self.show_start()
                finished = True
        if not finished and self._worker_messages is not None:
            self.root.after(100, self._poll_worker)

    def _append_log(self, value: str) -> None:
        if self._log_text is None:
            return
        self._log_text.configure(state="normal")
        self._log_text.insert("end", value)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()

    def _pipeline_finished(self, project_dir: Path) -> None:
        try:
            self.open_project(project_dir)
        except Exception as error:
            self._pipeline_failed(error)

    def _audio_refresh_finished(
        self,
        payload: tuple[Path, dict[str, Any]],
    ) -> None:
        project_dir, result = payload
        try:
            self.open_project(project_dir)
        except Exception as error:
            self._pipeline_failed(error)
            return
        messagebox.showinfo(
            "Audio refresh complete",
            (
                f"Updated {result['total_segment_count']} segment file(s) "
                f"from {result['changed_source_count']} changed source "
                "recording(s).\n\n"
                "Existing transcripts, alignment, and review selections "
                "were preserved."
            ),
            parent=self.root,
        )

    def _pipeline_failed(self, error: Exception) -> None:
        messagebox.showerror("Pipeline failed", str(error), parent=self.root)
        if self._log_text is not None:
            parent = self._log_text.master
            ttk.Button(
                parent,
                text="Back to Start",
                command=self.show_start,
            ).pack(anchor="e", pady=(12, 0))

    def show_review(self) -> None:
        assert self.review_data is not None
        assert self.project_dir is not None
        self._clear()
        self.selected_candidate_id = None
        self.selected_base_segment_id = None
        self._candidate_line_id = None

        toolbar = ttk.Frame(self.root, padding=(16, 12))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Line Review", style="Heading.TLabel").pack(
            side="left"
        )
        ttk.Label(
            toolbar,
            text=str(self.project_dir),
            style="Muted.TLabel",
        ).pack(side="left", padx=16)
        ttk.Button(toolbar, text="Close Project", command=self.show_start).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(
            toolbar,
            text="Finalize Selected Lines",
            style="Primary.TButton",
            command=self.finalize,
        ).pack(side="right")
        ttk.Button(
            toolbar,
            text="Export retakes script",
            command=self.export_retakes,
        ).pack(side="right", padx=(0, 8))

        controls = ttk.Frame(self.root, padding=(16, 0, 16, 10))
        controls.pack(fill="x")
        ttk.Label(controls, text="Status:").pack(side="left")
        status_values = ["All", *sorted(LINE_STATUSES)]
        status_box = ttk.Combobox(
            controls,
            textvariable=self.status_filter,
            values=status_values,
            state="readonly",
            width=22,
        )
        status_box.pack(side="left", padx=(6, 18))
        status_box.bind("<<ComboboxSelected>>", lambda _event: self.render_lines())
        ttk.Label(
            controls,
            textvariable=self.status_text,
            style="Muted.TLabel",
        ).pack(side="right")

        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        left = ttk.Labelframe(panes, text="Lines", padding=8)
        right = ttk.Labelframe(panes, text="Candidate segments", padding=8)
        panes.add(left, weight=1)
        panes.add(right, weight=1)

        line_columns = (
            "index",
            "sheet",
            "line",
            "target",
            "type",
            "status",
            "score",
            "audio",
        )
        self.line_tree = ttk.Treeview(
            left,
            columns=line_columns,
            show="headings",
            selectmode="browse",
        )
        line_headings = {
            "index": ("Index", 65),
            "sheet": ("Sheet", 115),
            "line": ("Line text", 350),
            "target": ("Target file", 170),
            "type": ("Type", 90),
            "status": ("Status", 130),
            "score": ("Selected score", 105),
            "audio": ("", 48),
        }
        for column, (heading, width) in line_headings.items():
            if column == "audio":
                self.line_tree.heading(column, text=heading)
            else:
                self.line_tree.heading(
                    column,
                    text=heading,
                    command=lambda value=column: self.sort_lines_by(value),
                )
            self.line_tree.column(
                column,
                width=width,
                minwidth=40 if column == "audio" else 70,
                stretch=False,
                anchor=(
                    "center"
                    if column in {"index", "type", "status", "score", "audio"}
                    else "w"
                ),
            )
        for status, color in STATUS_COLORS.items():
            self.line_tree.tag_configure(status, background=color)
        self.line_vertical_scrollbar = ttk.Scrollbar(
            left,
            orient="vertical",
            command=self.line_tree.yview,
        )
        self.line_horizontal_scrollbar = ttk.Scrollbar(
            left,
            orient="horizontal",
            command=self.line_tree.xview,
        )
        self.line_tree.configure(
            yscrollcommand=self.line_vertical_scrollbar.set,
            xscrollcommand=self.line_horizontal_scrollbar.set,
        )
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self.line_tree.grid(row=0, column=0, sticky="nsew")
        self.line_vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        self.line_horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        self.line_tree.bind("<<TreeviewSelect>>", self._tree_line_selected)
        self.line_tree.bind("<ButtonRelease-1>", self._line_table_clicked)
        self.line_tree.bind("<Motion>", self._line_table_motion)

        selected_details = ttk.Frame(right, padding=(4, 2, 4, 6))
        selected_details.grid(row=0, column=0, columnspan=2, sticky="ew")
        selected_details.grid_columnconfigure(1, weight=1)
        detail_fields = [
            ("Context", "selected_context_text"),
            ("Line", "selected_line_label"),
            ("Acting note", "selected_acting_note_label"),
        ]
        for row, (title, attribute) in enumerate(detail_fields):
            ttk.Label(
                selected_details,
                text=f"{title}:",
                style="FieldName.TLabel",
                anchor="nw",
            ).grid(row=row, column=0, sticky="nw", padx=(0, 10), pady=2)
            if title == "Context":
                context_field = ttk.Frame(selected_details)
                context_field.grid(row=row, column=1, sticky="ew", pady=2)
                context_field.grid_columnconfigure(0, weight=1)
                value_text = tk.Text(
                    context_field,
                    height=3,
                    wrap="word",
                    relief="solid",
                    borderwidth=1,
                    padx=4,
                    pady=3,
                    font=("Segoe UI", 9),
                    background="#ffffff",
                    foreground="#1f2937",
                    state="disabled",
                )
                context_scrollbar = ttk.Scrollbar(
                    context_field,
                    orient="vertical",
                    command=value_text.yview,
                )
                value_text.configure(yscrollcommand=context_scrollbar.set)
                value_text.grid(row=0, column=0, sticky="ew")
                context_scrollbar.grid(row=0, column=1, sticky="ns")
                setattr(self, attribute, value_text)
                continue
            value_label = ttk.Label(
                selected_details,
                text="—",
                anchor="nw",
                justify="left",
                wraplength=600,
            )
            value_label.grid(row=row, column=1, sticky="ew", pady=2)
            setattr(self, attribute, value_label)

        self.candidate_description = ttk.Label(
            right,
            text="",
            wraplength=650,
            style="Muted.TLabel",
            padding=(4, 2, 4, 8),
        )
        self.candidate_description.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
        )
        candidate_actions = ttk.Frame(right, padding=(4, 0, 4, 8))
        candidate_actions.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
        )
        self.mark_retake_button = ttk.Button(
            candidate_actions,
            text="Mark for retake",
            command=self.mark_for_retake,
        )
        self.mark_retake_button.pack(side="left")
        self.review_tabs = ttk.Notebook(right)
        self.review_tabs.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="nsew",
        )
        self.candidates_tab = ttk.Frame(self.review_tabs)
        self.base_segments_tab = ttk.Frame(self.review_tabs)
        self.review_tabs.add(self.candidates_tab, text="Candidates")
        self.review_tabs.add(self.base_segments_tab, text="Base segments")
        self.review_tabs.bind("<<NotebookTabChanged>>", self._review_tab_changed)

        candidate_columns = (
            "segment",
            "transcript",
            "score",
            "audio",
            "selection",
            "edit",
            "transcribe",
            "delete",
        )
        self.candidate_tree = ttk.Treeview(
            self.candidates_tab,
            columns=candidate_columns,
            show="tree headings",
            selectmode="browse",
        )
        self.candidate_tree.heading("#0", text="Take")
        self.candidate_tree.column(
            "#0",
            width=145,
            minwidth=90,
            stretch=False,
            anchor="w",
        )
        candidate_headings = {
            "segment": ("Segment", 210, "w"),
            "transcript": ("Transcript", 430, "w"),
            "score": ("Score", 70, "center"),
            "audio": ("", 48, "center"),
            "selection": ("Selection", 90, "center"),
            "edit": ("Copy/Edit", 80, "center"),
            "transcribe": ("Transcribe", 90, "center"),
            "delete": ("Delete", 65, "center"),
        }
        for column, (heading, width, anchor) in candidate_headings.items():
            self.candidate_tree.heading(column, text=heading)
            self.candidate_tree.column(
                column,
                width=width,
                minwidth=40,
                stretch=False,
                anchor=anchor,
            )
        self.candidate_tree.tag_configure("selected", background="#bbf7d0")
        self.candidate_tree.tag_configure("selected_elsewhere", background="#fde68a")
        self.candidate_vertical_scrollbar = ttk.Scrollbar(
            self.candidates_tab,
            orient="vertical",
            command=self.candidate_tree.yview,
        )
        self.candidate_horizontal_scrollbar = ttk.Scrollbar(
            self.candidates_tab,
            orient="horizontal",
            command=self.candidate_tree.xview,
        )
        self.candidate_tree.configure(
            yscrollcommand=self.candidate_vertical_scrollbar.set,
            xscrollcommand=self.candidate_horizontal_scrollbar.set,
        )
        self.candidates_tab.grid_rowconfigure(0, weight=1)
        self.candidates_tab.grid_columnconfigure(0, weight=1)
        self.candidate_tree.grid(row=0, column=0, sticky="nsew")
        self.candidate_vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        self.candidate_horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        ttk.Label(
            self.base_segments_tab,
            text=(
                "Green segments already contribute to this line; yellow segments "
                "contribute to another line."
            ),
            style="Muted.TLabel",
            padding=(4, 5, 4, 7),
        ).grid(row=0, column=0, columnspan=2, sticky="ew")
        base_columns = (
            "session",
            "segment",
            "transcript",
            "duration",
            "audio",
            "add",
        )
        self.base_segment_tree = ttk.Treeview(
            self.base_segments_tab,
            columns=base_columns,
            show="headings",
            selectmode="browse",
        )
        base_headings = {
            "session": ("Session", 125, "w"),
            "segment": ("Segment", 210, "w"),
            "transcript": ("Transcript", 470, "w"),
            "duration": ("Duration", 75, "center"),
            "audio": ("", 48, "center"),
            "add": ("Add", 65, "center"),
        }
        for column, (heading, width, anchor) in base_headings.items():
            self.base_segment_tree.heading(column, text=heading)
            self.base_segment_tree.column(
                column,
                width=width,
                minwidth=40,
                stretch=False,
                anchor=anchor,
            )
        self.base_segment_tree.tag_configure(
            "candidate_for_line",
            background="#bbf7d0",
        )
        self.base_segment_tree.tag_configure(
            "candidate_for_other_line",
            background="#fde68a",
        )
        self.base_segment_vertical_scrollbar = ttk.Scrollbar(
            self.base_segments_tab,
            orient="vertical",
            command=self.base_segment_tree.yview,
        )
        self.base_segment_horizontal_scrollbar = ttk.Scrollbar(
            self.base_segments_tab,
            orient="horizontal",
            command=self.base_segment_tree.xview,
        )
        self.base_segment_tree.configure(
            yscrollcommand=self.base_segment_vertical_scrollbar.set,
            xscrollcommand=self.base_segment_horizontal_scrollbar.set,
        )
        self.base_segments_tab.grid_rowconfigure(1, weight=1)
        self.base_segments_tab.grid_columnconfigure(0, weight=1)
        self.base_segment_tree.grid(row=1, column=0, sticky="nsew")
        self.base_segment_vertical_scrollbar.grid(row=1, column=1, sticky="ns")
        self.base_segment_horizontal_scrollbar.grid(row=2, column=0, sticky="ew")
        self.base_segment_tree.bind(
            "<ButtonRelease-1>",
            self._base_segment_table_clicked,
        )
        self.base_segment_tree.bind("<Motion>", self._base_segment_table_motion)
        self.base_segment_tree.bind(
            "<<TreeviewSelect>>",
            self._base_segment_tree_selected,
        )

        right.grid_rowconfigure(3, weight=3)
        right.grid_rowconfigure(5, weight=2)
        right.grid_columnconfigure(0, weight=1)
        self.candidate_tree.bind("<ButtonRelease-1>", self._candidate_table_clicked)
        self.candidate_tree.bind("<Motion>", self._candidate_table_motion)
        self.candidate_tree.bind(
            "<<TreeviewSelect>>",
            self._candidate_tree_selected,
        )
        self.candidate_waveform = CandidateWaveformView(
            parent=right,
            player=self.player,
        )
        self.candidate_waveform.frame.grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="nsew",
            pady=(8, 0),
        )
        self.render_lines()
        self.render_candidates()

    def _filtered_lines(self) -> list[dict[str, Any]]:
        assert self.review_data is not None
        lines = list(self.review_data["lines"])
        line_indexes = {
            line["line_id"]: index
            for index, line in enumerate(self.review_data["lines"], start=1)
        }
        selected_status = self.status_filter.get()
        if selected_status != "All":
            lines = [
                line for line in lines if line["status"] == selected_status
            ]

        def selected_score(line: dict[str, Any]) -> float:
            score = _selected_segment_score(self.review_data, line)
            return score if score is not None else -1.0

        keys: dict[str, Callable[[dict[str, Any]], Any]] = {
            "index": lambda line: line_indexes[line["line_id"]],
            "status": lambda line: (
                line["status"],
                line["sheet"],
                line["excel_row"],
            ),
            "sheet": lambda line: (line["sheet"], line["excel_row"]),
            "line": lambda line: line["line_text"].casefold(),
            "target": lambda line: line["target_filename"].casefold(),
            "type": lambda line: (
                line["type"],
                line["sheet"],
                line["excel_row"],
            ),
            "score": selected_score,
        }
        return sorted(
            lines,
            key=keys[self.line_sort_column],
            reverse=self.line_sort_descending,
        )

    def sort_lines_by(self, column: str) -> None:
        if column == self.line_sort_column:
            self.line_sort_descending = not self.line_sort_descending
        else:
            self.line_sort_column = column
            self.line_sort_descending = False
        self.render_lines()

    def render_lines(self) -> None:
        assert self.review_data is not None
        children = self.line_tree.get_children()
        if children:
            self.line_tree.delete(*children)
        lines = self._filtered_lines()
        self.status_text.set(
            f"{len(lines)} of {len(self.review_data['lines'])} lines"
        )
        line_indexes = {
            line["line_id"]: index
            for index, line in enumerate(self.review_data["lines"], start=1)
        }
        for line in lines:
            selected_score = _selected_segment_score(self.review_data, line)
            self.line_tree.insert(
                "",
                "end",
                iid=line["line_id"],
                values=(
                    line_indexes[line["line_id"]],
                    line["sheet"],
                    line["line_text"],
                    line["target_filename"],
                    line["type"],
                    line["status"],
                    (
                        f"{selected_score:.1f}"
                        if selected_score is not None
                        else ""
                    ),
                    "\u25b6" if line.get("selected_segment_id") else "",
                ),
                tags=(line["status"],),
            )
        visible_ids = {line["line_id"] for line in lines}
        selection_changed = False
        if self.selected_line_id not in visible_ids:
            self.selected_line_id = lines[0]["line_id"] if lines else None
            selection_changed = True
        if self.selected_line_id in visible_ids:
            self.line_tree.selection_set(self.selected_line_id)
            self.line_tree.focus(self.selected_line_id)
            self.line_tree.see(self.selected_line_id)
        self._update_line_headings()
        if selection_changed:
            self.render_candidates()

    def _update_line_headings(self) -> None:
        names = {
            "index": "Index",
            "sheet": "Sheet",
            "line": "Line text",
            "target": "Target file",
            "type": "Type",
            "status": "Status",
            "score": "Selected score",
        }
        arrow = "▼" if self.line_sort_descending else "▲"
        for column, name in names.items():
            suffix = f" {arrow}" if column == self.line_sort_column else ""
            self.line_tree.heading(column, text=name + suffix)

    def _selected_line(self) -> dict[str, Any] | None:
        if self.review_data is None or self.selected_line_id is None:
            return None
        return next(
            (
                line
                for line in self.review_data["lines"]
                if line["line_id"] == self.selected_line_id
            ),
            None,
        )

    def _tree_line_selected(self, _event: tk.Event[Any]) -> None:
        selected = self.line_tree.selection()
        if not selected:
            return
        self.selected_line_id = selected[0]
        self.render_candidates()

    def _line_table_clicked(self, event: tk.Event[Any]) -> None:
        row_id = self.line_tree.identify_row(event.y)
        if not row_id:
            return
        if self.line_tree.identify_column(event.x) == "#8":
            line = next(
                (
                    item
                    for item in self.review_data["lines"]
                    if item["line_id"] == row_id
                ),
                None,
            )
            if line and line.get("selected_segment_id"):
                self.play_segment(str(line["selected_segment_id"]))

    def _line_table_motion(self, event: tk.Event[Any]) -> None:
        row_id = self.line_tree.identify_row(event.y)
        is_audio = self.line_tree.identify_column(event.x) == "#8"
        has_audio = bool(
            row_id and self.line_tree.set(row_id, "audio")
        )
        self.line_tree.configure(
            cursor="hand2" if is_audio and has_audio else ""
        )

    def _set_context_display(self, value: Any) -> None:
        text = _context_display_text(value) or "\u2014"
        self.selected_context_text.configure(state="normal")
        self.selected_context_text.delete("1.0", "end")
        self.selected_context_text.insert("1.0", text)
        self.selected_context_text.configure(state="disabled")
        self.selected_context_text.yview_moveto(0.0)

    def render_candidates(self) -> None:
        assert self.review_data is not None
        children = self.candidate_tree.get_children()
        self._open_candidate_roots.update(
            str(item_id)
            for item_id in children
            if bool(self.candidate_tree.item(item_id, "open"))
        )
        self._open_candidate_roots.difference_update(
            str(item_id)
            for item_id in children
            if not bool(self.candidate_tree.item(item_id, "open"))
        )
        if children:
            self.candidate_tree.delete(*children)
        line = self._selected_line()
        if line is None:
            self.selected_candidate_id = None
            self.selected_base_segment_id = None
            self._candidate_line_id = None
            base_children = self.base_segment_tree.get_children()
            if base_children:
                self.base_segment_tree.delete(*base_children)
            self.review_tabs.tab(self.base_segments_tab, state="disabled")
            self.review_tabs.select(self.candidates_tab)
            self.mark_retake_button.configure(state="disabled")
            self._set_context_display("")
            self.selected_line_label.configure(text="—")
            self.selected_acting_note_label.configure(text="—")
            self.candidate_description.configure(
                text="Select a line to review its candidates."
            )
            if self.candidate_waveform is not None:
                self.candidate_waveform.clear()
            return

        line_id = str(line["line_id"])
        line_changed = self._candidate_line_id != line_id
        if line_changed:
            self.selected_candidate_id = None
            self.selected_base_segment_id = None
        self._candidate_line_id = line_id
        is_verbal = line["type"] == "normal"
        self.review_tabs.tab(
            self.base_segments_tab,
            state="normal" if is_verbal else "disabled",
        )
        if not is_verbal and self._base_segments_tab_active():
            self.review_tabs.select(self.candidates_tab)
        if is_verbal:
            self._render_base_segments(line, reset_focus=line_changed)
        else:
            base_children = self.base_segment_tree.get_children()
            if base_children:
                self.base_segment_tree.delete(*base_children)
            self.selected_base_segment_id = None
        self.mark_retake_button.configure(
            state=("disabled" if line["status"] == "RETAKE" else "normal")
        )
        self._set_context_display(line.get("context"))
        self.selected_line_label.configure(text=line["line_text"] or "—")
        self.selected_acting_note_label.configure(
            text=line.get("acting_note") or "—"
        )
        uses_unmatched = _uses_unmatched_candidates(line)
        description = (
            "This line is marked for retake. Selecting a candidate will remove "
            "the retake status."
            if line["status"] == "RETAKE"
            else ""
        )
        if uses_unmatched:
            unmatched_description = (
                "Unmatched audible segments are shown for nonverbal lines. "
                "Amber candidates are already selected for another line."
                if line["type"] == "nonverbal"
                else (
                    "No alignment candidate was found; all unmatched audible "
                    "segments are shown."
                )
            )
            description = "\n".join(
                value for value in (description, unmatched_description) if value
            )
        self.candidate_description.configure(text=description)

        candidates = self._displayed_candidates(line)
        if uses_unmatched:
            manual_ids = {
                str(candidate["segment_id"])
                for candidate in line["candidates"]
                if candidate.get("manual_edit")
            }
            candidates = sorted(
                candidates,
                key=lambda candidate: (
                    str(candidate["segment_id"]) not in manual_ids,
                    str(candidate["segment_id"]).casefold(),
                ),
            )
        else:
            candidates = sorted(
                candidates,
                key=lambda candidate: float(candidate.get("score", 0.0)),
                reverse=True,
            )
        if not candidates:
            self.selected_candidate_id = None
            self.candidate_description.configure(
                text=(
                    f"{description}\n\n" if description else ""
                )
                + "No candidate segments are available."
            )
            if self._base_segments_tab_active():
                self._show_selected_base_segment_waveform()
            elif self.candidate_waveform is not None:
                self.candidate_waveform.clear("No candidate waveform is available.")
            return

        selected_line_ids = _selected_line_ids_by_segment(self.review_data)

        def insert_candidate(
            candidate: dict[str, Any],
            *,
            parent: str,
            take_label: str,
            is_open: bool = False,
        ) -> None:
            is_manual_edit = bool(candidate.get("manual_edit"))
            has_custom_asr = (
                str(candidate.get("transcript_source") or "")
                == "candidate_asr_manual_copy_edit"
            )
            transcript = str(candidate.get("transcript") or "").strip()
            if not transcript:
                transcript = (
                    (
                        "[Transcribed - no speech recognized]"
                        if has_custom_asr
                        else "[Custom segment - not transcribed]"
                    )
                    if is_manual_edit
                    else "[No transcript]"
                )
            segment_id = str(candidate["segment_id"])
            score_text = (
                ""
                if candidate.get("manually_added_base_segment")
                else f"{float(candidate.get('score', 0.0)):.1f}"
            )
            selection_text, tags = _candidate_selection_display(
                line=line,
                segment_id=segment_id,
                selected_line_ids=selected_line_ids,
            )
            self.candidate_tree.insert(
                parent,
                "end",
                iid=segment_id,
                text=take_label,
                open=is_open,
                values=(
                    segment_id,
                    transcript,
                    score_text,
                    "▶",
                    selection_text,
                    "\u2398 \u270e",
                    (
                        "Retranscribe"
                        if is_manual_edit and has_custom_asr
                        else ("Transcribe" if is_manual_edit else "")
                    ),
                    "Delete" if is_manual_edit else "",
                ),
                tags=tags,
            )

        take_groups = _candidate_take_groups(
            candidates,
            self.base_segments_by_session,
        )
        for take_number, group in enumerate(take_groups, start=1):
            root_candidate = group[0]
            root_id = str(root_candidate["segment_id"])
            root_label = f"Take {take_number}"
            if len(group) > 1:
                root_label += f" ({len(group)} versions)"
            elif root_candidate.get("manual_edit"):
                root_label += " · Custom"
            elif root_candidate.get("manually_added_base_segment"):
                root_label += " · Added"
            insert_candidate(
                root_candidate,
                parent="",
                take_label=root_label,
                is_open=root_id in self._open_candidate_roots,
            )
            for alternative_number, candidate in enumerate(group[1:], start=1):
                insert_candidate(
                    candidate,
                    parent=root_id,
                    take_label=(
                        "Custom edit"
                        if candidate.get("manual_edit")
                        else f"Alternative {alternative_number}"
                    ),
                )

        candidate_ids = [str(candidate["segment_id"]) for candidate in candidates]
        preferred_id = self.selected_candidate_id
        if preferred_id not in candidate_ids:
            preferred_id = next(
                (
                    str(candidate_id)
                    for candidate_id in (
                        line.get("selected_segment_id"),
                        line.get("suggested_segment_id"),
                    )
                    if candidate_id and str(candidate_id) in candidate_ids
                ),
                candidate_ids[0],
            )
        self.selected_candidate_id = preferred_id
        self.candidate_tree.selection_set(preferred_id)
        self.candidate_tree.focus(preferred_id)
        self.candidate_tree.see(preferred_id)
        if self._base_segments_tab_active():
            self._show_selected_base_segment_waveform()
        else:
            self._show_candidate_waveform(preferred_id)

    def _base_segments_tab_active(self) -> bool:
        return str(self.review_tabs.select()) == str(self.base_segments_tab)

    def _render_base_segments(
        self,
        line: dict[str, Any],
        *,
        reset_focus: bool,
    ) -> None:
        assert self.review_data is not None
        children = self.base_segment_tree.get_children()
        if children:
            self.base_segment_tree.delete(*children)

        current_keys, other_keys = _base_segment_candidate_usage(
            self.review_data,
            str(line["line_id"]),
        )
        segment_id_by_key: dict[tuple[str, int], str] = {}
        segment_ids: list[str] = []
        for session_id, segments in self.base_segments_by_session.items():
            for position, segment in enumerate(segments):
                base_indices = segment.get("base_indices") or [position]
                base_index = int(base_indices[0])
                key = (session_id, base_index)
                segment_id = str(segment["segment_id"])
                segment_ids.append(segment_id)
                segment_id_by_key[key] = segment_id
                if key in current_keys:
                    tags = ("candidate_for_line",)
                    add_text = ""
                elif key in other_keys:
                    tags = ("candidate_for_other_line",)
                    add_text = "Add"
                else:
                    tags = ()
                    add_text = "Add"
                transcript = str(segment.get("transcript") or "").strip()
                duration = float(
                    (segment.get("metrics") or {}).get(
                        "duration_seconds",
                        float(segment.get("end_seconds", 0.0))
                        - float(segment.get("start_seconds", 0.0)),
                    )
                )
                self.base_segment_tree.insert(
                    "",
                    "end",
                    iid=segment_id,
                    values=(
                        session_id,
                        segment_id,
                        transcript or "[No transcript]",
                        f"{duration:.2f}s",
                        "▶",
                        add_text,
                    ),
                    tags=tags,
                )

        preferred_id = None if reset_focus else self.selected_base_segment_id
        if preferred_id not in segment_ids:
            focus_key = _first_line_base_segment_key(line)
            preferred_id = segment_id_by_key.get(focus_key) if focus_key else None
        if preferred_id not in segment_ids:
            preferred_id = segment_ids[0] if segment_ids else None
        self.selected_base_segment_id = preferred_id
        if preferred_id is not None:
            self.base_segment_tree.selection_set(preferred_id)
            self.base_segment_tree.focus(preferred_id)
            self.base_segment_tree.see(preferred_id)

    def _review_tab_changed(self, _event: tk.Event[Any]) -> None:
        if self._base_segments_tab_active():
            self._show_selected_base_segment_waveform()
        elif self.selected_candidate_id:
            self._show_candidate_waveform(self.selected_candidate_id)
        elif self.candidate_waveform is not None:
            self.candidate_waveform.clear("No candidate waveform is available.")

    def _base_segment_tree_selected(self, _event: tk.Event[Any]) -> None:
        selected = self.base_segment_tree.selection()
        if not selected:
            return
        segment_id = str(selected[0])
        changed = segment_id != self.selected_base_segment_id
        self.selected_base_segment_id = segment_id
        if changed and self._base_segments_tab_active():
            self._show_candidate_waveform(segment_id)

    def _show_selected_base_segment_waveform(self) -> None:
        if self.selected_base_segment_id:
            self._show_candidate_waveform(self.selected_base_segment_id)
        elif self.candidate_waveform is not None:
            self.candidate_waveform.clear("No base-segment waveform is available.")

    def _base_segment_table_clicked(self, event: tk.Event[Any]) -> None:
        segment_id = self.base_segment_tree.identify_row(event.y)
        if not segment_id:
            return
        column = self.base_segment_tree.identify_column(event.x)
        if column == "#5":
            self.selected_base_segment_id = segment_id
            self.base_segment_tree.selection_set(segment_id)
            self.base_segment_tree.focus(segment_id)
            self._show_candidate_waveform(segment_id)
            if self.candidate_waveform is not None and self.candidate_waveform._loaded:
                self.candidate_waveform.restart_playback()
        elif column == "#6" and self.base_segment_tree.set(segment_id, "add"):
            self.add_base_segment_as_candidate(segment_id)

    def _base_segment_table_motion(self, event: tk.Event[Any]) -> None:
        segment_id = self.base_segment_tree.identify_row(event.y)
        column = self.base_segment_tree.identify_column(event.x)
        is_action = bool(
            segment_id
            and (
                column == "#5"
                or (
                    column == "#6"
                    and self.base_segment_tree.set(segment_id, "add")
                )
            )
        )
        self.base_segment_tree.configure(cursor="hand2" if is_action else "")

    def add_base_segment_as_candidate(self, segment_id: str) -> None:
        assert self.project_dir is not None
        assert self.review_path is not None
        assert self.review_data is not None
        line = self._selected_line()
        if line is None:
            return
        try:
            add_base_segment_candidate(
                project_dir=self.project_dir,
                review_path=self.review_path,
                review_data=self.review_data,
                line_id=str(line["line_id"]),
                segment_id=segment_id,
            )
        except Exception as error:
            messagebox.showerror(
                "Cannot add base segment",
                str(error),
                parent=self.root,
            )
            return
        self.render_lines()
        self.render_candidates()

    def _candidate_tree_selected(self, _event: tk.Event[Any]) -> None:
        selected = self.candidate_tree.selection()
        if not selected:
            return
        segment_id = str(selected[0])
        if segment_id == self.selected_candidate_id:
            return
        self.selected_candidate_id = segment_id
        if not self._base_segments_tab_active():
            self._show_candidate_waveform(segment_id)

    def _show_candidate_waveform(self, segment_id: str) -> None:
        if self.candidate_waveform is None or self.project_dir is None:
            return
        try:
            source = segment_edit_source(
                project_dir=self.project_dir,
                segment_id=segment_id,
            )
            segment = source["segment"]
            self.candidate_waveform.show_segment(
                audio_path=source["audio_path"],
                sample_rate=int(source["sample_rate"]),
                source_frames=int(source["source_frames"]),
                start_sample=int(segment["start_sample"]),
                end_sample=int(segment["end_sample"]),
            )
        except Exception as error:
            self.candidate_waveform.clear(f"Waveform unavailable: {error}")

    def _candidate_table_clicked(self, event: tk.Event[Any]) -> None:
        segment_id = self.candidate_tree.identify_row(event.y)
        if not segment_id:
            return
        column = self.candidate_tree.identify_column(event.x)
        if column == "#4":
            self.play_segment(segment_id)
        elif column == "#5":
            self.toggle_candidate(segment_id)
        elif column == "#6":
            self.copy_and_edit_candidate(segment_id)
        elif column == "#7" and self.candidate_tree.set(segment_id, "transcribe"):
            self.transcribe_custom_candidate(segment_id)
        elif column == "#8" and self.candidate_tree.set(segment_id, "delete"):
            self.delete_custom_candidate(segment_id)

    def _candidate_table_motion(self, event: tk.Event[Any]) -> None:
        segment_id = self.candidate_tree.identify_row(event.y)
        column = self.candidate_tree.identify_column(event.x)
        manual_action = bool(
            segment_id
            and column in {"#7", "#8"}
            and self.candidate_tree.set(
                segment_id,
                "transcribe" if column == "#7" else "delete",
            )
        )
        self.candidate_tree.configure(
            cursor=(
                "hand2"
                if segment_id and (column in {"#4", "#5", "#6"} or manual_action)
                else ""
            )
        )

    def _displayed_candidates(
        self,
        line: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assert self.review_data is not None
        if line["type"] == "nonverbal" or _uses_unmatched_candidates(line):
            combined = [
                *line["candidates"],
                *self.review_data["unmatched_segments"],
            ]
            unique: dict[str, dict[str, Any]] = {}
            for candidate in combined:
                unique.setdefault(str(candidate["segment_id"]), candidate)
            return list(unique.values())
        return list(line["candidates"])

    def copy_and_edit_candidate(self, segment_id: str) -> None:
        assert self.project_dir is not None
        assert self.project is not None
        assert self.review_path is not None
        assert self.review_data is not None
        line = self._selected_line()
        if line is None:
            return
        candidate = next(
            (
                item
                for item in self._displayed_candidates(line)
                if str(item["segment_id"]) == segment_id
            ),
            None,
        )
        if candidate is None:
            return
        try:
            source = segment_edit_source(
                project_dir=self.project_dir,
                segment_id=segment_id,
            )
            segment = source["segment"]
            self.player.stop()
            dialog = SegmentEditorDialog(
                parent=self.root,
                player=self.player,
                segment_id=segment_id,
                audio_path=source["audio_path"],
                sample_rate=int(source["sample_rate"]),
                source_frames=int(source["source_frames"]),
                start_sample=int(segment["start_sample"]),
                end_sample=int(segment["end_sample"]),
                save_callback=lambda start, end: save_edited_candidate(
                    project_dir=self.project_dir,
                    project=self.project,
                    review_path=self.review_path,
                    review_data=self.review_data,
                    line_id=str(line["line_id"]),
                    source_candidate=candidate,
                    start_sample=start,
                    end_sample=end,
                ),
            )
            self.root.wait_window(dialog.window)
        except Exception as error:
            messagebox.showerror(
                "Cannot edit segment",
                str(error),
                parent=self.root,
            )
            return
        if dialog.saved:
            self.render_lines()
            self.render_candidates()

    def transcribe_custom_candidate(self, segment_id: str) -> None:
        assert self.project_dir is not None
        assert self.project is not None
        assert self.review_path is not None
        assert self.review_data is not None
        self.candidate_description.configure(
            text="Transcribing the custom segment..."
        )
        self.root.configure(cursor="wait")
        self.root.update_idletasks()
        try:
            transcribe_edited_candidate(
                project_dir=self.project_dir,
                project=self.project,
                review_path=self.review_path,
                review_data=self.review_data,
                segment_id=segment_id,
                transcription_runtime=self._candidate_transcription_runtime,
            )
        except Exception as error:
            messagebox.showerror(
                "Cannot transcribe custom segment",
                str(error),
                parent=self.root,
            )
        finally:
            self.root.configure(cursor="")
            self.render_lines()
            self.render_candidates()

    def delete_custom_candidate(self, segment_id: str) -> None:
        assert self.project_dir is not None
        assert self.review_path is not None
        assert self.review_data is not None
        if not messagebox.askyesno(
            "Delete custom segment",
            "Permanently delete this copy-and-edit segment?",
            parent=self.root,
        ):
            return
        try:
            self.player.stop()
            result = delete_edited_candidate(
                project_dir=self.project_dir,
                review_path=self.review_path,
                review_data=self.review_data,
                segment_id=segment_id,
            )
        except Exception as error:
            messagebox.showerror(
                "Cannot delete custom segment",
                str(error),
                parent=self.root,
            )
            return
        self.render_lines()
        self.render_candidates()
        if result["warnings"]:
            messagebox.showwarning(
                "Custom segment removed with warnings",
                "\n".join(result["warnings"]),
                parent=self.root,
            )

    def play_segment(self, segment_id: str) -> None:
        assert self.project_dir is not None
        assert self.review_data is not None
        if (
            self.candidate_waveform is not None
            and self.candidate_tree.exists(segment_id)
        ):
            if self.selected_candidate_id != segment_id:
                self.selected_candidate_id = segment_id
                self.candidate_tree.selection_set(segment_id)
                self.candidate_tree.focus(segment_id)
                self.candidate_tree.see(segment_id)
                self._show_candidate_waveform(segment_id)
            if self.candidate_waveform._loaded:
                self.candidate_waveform.restart_playback()
                return
        try:
            path = segment_file_for_id(
                project_dir=self.project_dir,
                review_data=self.review_data,
                segment_id=segment_id,
            )
            self.player.play(path)
        except Exception as error:
            messagebox.showerror("Cannot play segment", str(error), parent=self.root)

    def toggle_candidate(self, segment_id: str) -> None:
        assert self.review_path is not None
        assert self.review_data is not None
        line = self._selected_line()
        if line is None:
            return
        if line.get("selected_segment_id") == segment_id:
            line["selected_segment_id"] = None
            line["status"] = (
                "MISSING"
                if line["type"] == "normal" and not line["candidates"]
                else "REVIEW"
            )
        else:
            line["selected_segment_id"] = segment_id
            line["status"] = "MANUALLY_REVIEWED"
        save_line_review(self.review_path, self.review_data)
        self.render_lines()
        self.render_candidates()

    def mark_for_retake(self) -> None:
        assert self.review_path is not None
        assert self.review_data is not None
        line = self._selected_line()
        if line is None:
            return
        line["selected_segment_id"] = None
        line["status"] = "RETAKE"
        save_line_review(self.review_path, self.review_data)
        self.render_lines()
        self.render_candidates()

    def export_retakes(self) -> None:
        assert self.project_dir is not None
        assert self.project is not None
        assert self.review_path is not None
        assert self.review_data is not None
        retake_count = sum(
            line["status"] == "RETAKE" for line in self.review_data["lines"]
        )
        if not retake_count:
            messagebox.showwarning(
                "No retakes",
                "No lines are marked for retake.",
                parent=self.root,
            )
            return

        source_name = Path(str(self.project["workbook"])).name
        source_suffix = Path(source_name).suffix.lower()
        output = filedialog.asksaveasfilename(
            title="Export retakes script",
            initialdir=str(self.project_dir),
            initialfile=(
                f"{Path(source_name).stem}_retakes{source_suffix}"
            ),
            defaultextension=source_suffix,
            filetypes=[
                ("Excel workbook", f"*{source_suffix}"),
                ("All files", "*.*"),
            ],
        )
        if not output:
            return
        try:
            result = export_retake_script(
                project_dir=self.project_dir,
                project=self.project,
                review_path=self.review_path,
                output_path=Path(output),
                overwrite=True,
            )
        except Exception as error:
            messagebox.showerror(
                "Retake export failed",
                str(error),
                parent=self.root,
            )
            return
        messagebox.showinfo(
            "Retake export complete",
            (
                f"Exported {result['export_count']} retake line(s) to:\n"
                f"{result['output_path']}"
            ),
            parent=self.root,
        )

    def finalize(self) -> None:
        assert self.project_dir is not None
        assert self.project is not None
        assert self.review_path is not None
        output = filedialog.askdirectory(
            title="Select final output directory",
            mustexist=False,
            initialdir=str(self.project_dir),
        )
        if not output:
            return
        output_dir = Path(output).resolve()
        overwrite = False
        if output_dir.exists() and any(output_dir.iterdir()):
            overwrite = messagebox.askyesno(
                "Existing output",
                "Replace final files that already exist in this directory?",
                parent=self.root,
            )
        try:
            result = finalize_review(
                project_dir=self.project_dir,
                project=self.project,
                review_path=self.review_path,
                output_dir=output_dir,
                overwrite=overwrite,
            )
        except Exception as error:
            messagebox.showerror("Finalization failed", str(error), parent=self.root)
            return
        messagebox.showinfo(
            "Finalization complete",
            (
                f"Exported {result['export_count']} selected line(s) to:\n"
                f"{output_dir}"
            ),
            parent=self.root,
        )


def main() -> int:
    root = tk.Tk()
    DialogueReviewApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
