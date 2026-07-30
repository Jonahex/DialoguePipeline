from __future__ import annotations

import contextlib
import queue
import subprocess
import sys
import tempfile
import threading
import traceback
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .alignment import align_project
from .alignment_settings import AlignmentSettings
from .audio import cut_pcm_wav
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
    load_project,
)
from .retakes import export_retake_script
from .review import (
    LINE_STATUSES,
    REVIEW_FILE_NAME,
    load_line_review,
    save_line_review,
    save_edited_candidate,
    segment_edit_source,
    segment_file_for_id,
)
from .segmentation import refresh_project_audio, segment_project
from .transcription import transcribe_project, transcribe_segments_project
from .util import project_file_from_arg, write_json


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


def _uses_unmatched_candidates(line: dict[str, Any]) -> bool:
    return bool(
        line["type"] == "nonverbal"
        or (
            not line.get("candidates")
            and line["status"] in {"MISSING", "MANUALLY_REVIEWED", "RETAKE"}
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


class AudioPlayer:
    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None

    def stop(self) -> None:
        if sys.platform == "win32":
            try:
                import winsound

                winsound.PlaySound(None, 0)
            except (ImportError, RuntimeError):
                pass
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
            self._process = None

    def play(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        self.stop()
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(
                str(path),
                winsound.SND_FILENAME | winsound.SND_ASYNC,
            )
            return
        self._process = subprocess.Popen(
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                str(path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
        self._temporary_dir = tempfile.TemporaryDirectory(
            prefix="dialogue-va-segment-editor-"
        )
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
                "Mouse wheel: zoom. Right-drag: move along the timeline."
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
        with wave.open(str(self.audio_path), "rb") as reader:
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

    def _drag_started(self, event: tk.Event[Any]) -> None:
        self._drag_boundary = self._nearest_boundary(event.x)
        self._drag_changed = False

    def _drag_moved(self, event: tk.Event[Any]) -> None:
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
            if self._nearest_boundary(event.x)
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
        self._play_current()

    def _play_current(self) -> None:
        self._cancel_playback_timer()
        self.player.stop()
        preview_path = Path(self._temporary_dir.name) / "preview.wav"
        try:
            cut_pcm_wav(
                self.audio_path,
                preview_path,
                start_sample=self.start_sample,
                end_sample=self.end_sample,
                fade_ms=0.0,
            )
            self.player.play(preview_path)
        except Exception as error:
            self._playback_active = False
            messagebox.showerror(
                "Cannot play edited segment",
                str(error),
                parent=self.window,
            )
            return
        self._playback_active = True
        duration_ms = int(
            round(
                1000.0
                * (self.end_sample - self.start_sample)
                / self.sample_rate
            )
        )
        self._playback_after_id = self.window.after(
            max(100, duration_ms + 100),
            self._playback_finished,
        )

    def _playback_finished(self) -> None:
        self._playback_after_id = None
        self._playback_active = False

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
        self.player.stop()
        with contextlib.suppress(tk.TclError):
            self.window.grab_release()
        self.window.destroy()
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
        self.player.stop()
        self.root.destroy()

    def _clear(self) -> None:
        self.player.stop()
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
        self.selected_line_id = None

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
                ("Excel workbooks", "*.xlsm *.xlsx"),
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

    def open_project(self, project_dir: Path) -> None:
        project_dir = project_dir.resolve()
        review_path = project_dir / REVIEW_FILE_NAME
        review_data = load_line_review(review_path)
        project_file = project_file_from_arg(project_dir)
        loaded_dir, project = load_project(project_file)
        self.project_dir = loaded_dir
        self.project = project
        self.review_path = review_path
        self.review_data = review_data
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
        self.show_progress(project_dir, title="Creating project")

        def work() -> Path:
            check_processing_cancelled()
            print("[phase 1/5] Creating project and inventory", flush=True)
            project = create_project(
                workbook_path=workbook_path,
                audio_dir=audio_dir,
                project_dir=project_dir,
                project_settings=project_settings,
            )
            check_processing_cancelled()
            print("[phase 2/5] Transcribing source recordings", flush=True)
            transcribe_project(project_dir=project_dir, project=project)
            check_processing_cancelled()
            print("[phase 3/5] Segmenting recordings", flush=True)
            segment_project(project_dir=project_dir, project=project)
            check_processing_cancelled()
            print("[phase 4/5] Transcribing temporary segments", flush=True)
            transcribe_segments_project(project_dir=project_dir, project=project)
            check_processing_cancelled()
            print("[phase 5/5] Aligning segments to script lines", flush=True)
            align_project(project_dir=project_dir, project=project)
            check_processing_cancelled()
            print("Pipeline complete.", flush=True)
            return project_dir

        self._start_worker(work, self._pipeline_finished)

    def run_existing_project(
        self,
        project_dir: Path,
        project_settings: dict[str, Any],
    ) -> None:
        """Rerun an existing project without recreating or clearing it."""
        self.show_progress(project_dir, title="Reprocessing project")

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
            print("[phase 1/4] Transcribing source recordings", flush=True)
            transcribe_project(project_dir=loaded_dir, project=project)
            check_processing_cancelled()
            print("[phase 2/4] Segmenting recordings", flush=True)
            segment_project(project_dir=loaded_dir, project=project)
            check_processing_cancelled()
            print("[phase 3/4] Transcribing temporary segments", flush=True)
            transcribe_segments_project(project_dir=loaded_dir, project=project)
            check_processing_cancelled()
            print("[phase 4/4] Aligning segments to script lines", flush=True)
            align_project(project_dir=loaded_dir, project=project)
            check_processing_cancelled()
            print("Project reprocessing complete.", flush=True)
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
                stretch=column in {"line", "target"},
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
            ("Context", "selected_context_label"),
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
        candidate_columns = (
            "segment",
            "transcript",
            "score",
            "audio",
            "selection",
            "edit",
        )
        self.candidate_tree = ttk.Treeview(
            right,
            columns=candidate_columns,
            show="headings",
            selectmode="none",
        )
        candidate_headings = {
            "segment": ("Segment", 210, "w"),
            "transcript": ("Transcript", 430, "w"),
            "score": ("Score", 70, "center"),
            "audio": ("", 48, "center"),
            "selection": ("Selection", 90, "center"),
            "edit": ("Copy/Edit", 80, "center"),
        }
        for column, (heading, width, anchor) in candidate_headings.items():
            self.candidate_tree.heading(column, text=heading)
            self.candidate_tree.column(
                column,
                width=width,
                minwidth=40,
                stretch=column == "transcript",
                anchor=anchor,
            )
        self.candidate_tree.tag_configure("selected", background="#bbf7d0")
        self.candidate_vertical_scrollbar = ttk.Scrollbar(
            right,
            orient="vertical",
            command=self.candidate_tree.yview,
        )
        self.candidate_horizontal_scrollbar = ttk.Scrollbar(
            right,
            orient="horizontal",
            command=self.candidate_tree.xview,
        )
        self.candidate_tree.configure(
            yscrollcommand=self.candidate_vertical_scrollbar.set,
            xscrollcommand=self.candidate_horizontal_scrollbar.set,
        )
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.candidate_tree.grid(row=3, column=0, sticky="nsew")
        self.candidate_vertical_scrollbar.grid(row=3, column=1, sticky="ns")
        self.candidate_horizontal_scrollbar.grid(
            row=4,
            column=0,
            sticky="ew",
        )
        self.candidate_tree.bind("<ButtonRelease-1>", self._candidate_table_clicked)
        self.candidate_tree.bind("<Motion>", self._candidate_table_motion)
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

    def render_candidates(self) -> None:
        assert self.review_data is not None
        children = self.candidate_tree.get_children()
        if children:
            self.candidate_tree.delete(*children)
        line = self._selected_line()
        if line is None:
            self.mark_retake_button.configure(state="disabled")
            self.selected_context_label.configure(text="—")
            self.selected_line_label.configure(text="—")
            self.selected_acting_note_label.configure(text="—")
            self.candidate_description.configure(
                text="Select a line to review its candidates."
            )
            return

        self.mark_retake_button.configure(
            state=("disabled" if line["status"] == "RETAKE" else "normal")
        )
        self.selected_context_label.configure(text=line.get("context") or "—")
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
                "Unmatched audible segments are shown for nonverbal lines."
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
            self.candidate_description.configure(
                text=(
                    f"{description}\n\n" if description else ""
                )
                + "No candidate segments are available."
            )
            return

        selected_id = line.get("selected_segment_id")
        for candidate in candidates:
            is_selected = candidate["segment_id"] == selected_id
            transcript = str(candidate.get("transcript") or "[No transcript]")
            segment_id = str(candidate["segment_id"])
            self.candidate_tree.insert(
                "",
                "end",
                iid=segment_id,
                values=(
                    segment_id,
                    transcript,
                    f"{float(candidate.get('score', 0.0)):.1f}",
                    "▶",
                    "Unselect" if is_selected else "Select",
                    "\u2398 \u270e",
                ),
                tags=("selected",) if is_selected else (),
            )

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

    def _candidate_table_motion(self, event: tk.Event[Any]) -> None:
        segment_id = self.candidate_tree.identify_row(event.y)
        column = self.candidate_tree.identify_column(event.x)
        self.candidate_tree.configure(
            cursor=(
                "hand2"
                if segment_id and column in {"#4", "#5", "#6"}
                else ""
            )
        )

    def _displayed_candidates(
        self,
        line: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assert self.review_data is not None
        if line["type"] == "nonverbal":
            combined = [
                *line["candidates"],
                *self.review_data["unmatched_segments"],
            ]
            unique: dict[str, dict[str, Any]] = {}
            for candidate in combined:
                unique.setdefault(str(candidate["segment_id"]), candidate)
            return list(unique.values())
        if _uses_unmatched_candidates(line):
            return list(self.review_data["unmatched_segments"])
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

    def play_segment(self, segment_id: str) -> None:
        assert self.project_dir is not None
        assert self.review_data is not None
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
