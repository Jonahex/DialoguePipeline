from __future__ import annotations

import contextlib
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .alignment import align_project
from .finalize import finalize_review
from .project import create_project, load_project
from .review import (
    LINE_STATUSES,
    REVIEW_FILE_NAME,
    load_line_review,
    save_line_review,
    segment_file_for_id,
)
from .segmentation import segment_project
from .transcription import transcribe_project, transcribe_segments_project
from .util import project_file_from_arg


APP_TITLE = "Dialogue VA Pipeline"
STATUS_COLORS = {
    "AUTO_OK": "#dcfce7",
    "REVIEW": "#fef3c7",
    "MISSING": "#fee2e2",
    "MANUALLY_REVIEWED": "#dbeafe",
}


def _uses_unmatched_candidates(line: dict[str, Any]) -> bool:
    return bool(
        line["type"] == "nonverbal"
        or (
            not line.get("candidates")
            and line["status"] in {"MISSING", "MANUALLY_REVIEWED"}
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


class QueueWriter:
    def __init__(self, messages: queue.Queue[tuple[str, Any]]) -> None:
        self.messages = messages

    def write(self, value: str) -> int:
        if value:
            self.messages.put(("log", value))
        return len(value)

    def flush(self) -> None:
        return None


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

        self._configure_styles()
        self.show_start()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"))
        style.configure("Heading.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("FieldName.TLabel", font=("Segoe UI", 9, "bold"))
        style.configure("Muted.TLabel", foreground="#475569")
        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"))

    def close(self) -> None:
        self.player.stop()
        self.root.destroy()

    def _clear(self) -> None:
        self.player.stop()
        for child in self.root.winfo_children():
            child.destroy()

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
            width=28,
        ).pack(ipady=8, pady=7)
        ttk.Button(
            card,
            text="Create New Project",
            style="Primary.TButton",
            command=self.choose_new_project,
            width=28,
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
        workbook = filedialog.askopenfilename(
            title="Select lines spreadsheet",
            filetypes=[
                ("Excel workbooks", "*.xlsm *.xlsx"),
                ("All files", "*.*"),
            ],
        )
        if not workbook:
            return
        audio_dir = filedialog.askdirectory(
            title="Select directory containing recorded WAV files"
        )
        if not audio_dir:
            return
        project_dir = filedialog.askdirectory(
            title="Select an empty directory for the new project",
            mustexist=False,
            initialdir=str(Path(workbook).parent),
        )
        if not project_dir:
            return
        self.run_new_project(
            workbook_path=Path(workbook),
            audio_dir=Path(audio_dir),
            project_dir=Path(project_dir),
        )

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
    ) -> None:
        self.show_progress(project_dir)

        def work() -> Path:
            print("[phase 1/5] Creating project and inventory", flush=True)
            project = create_project(
                workbook_path=workbook_path,
                audio_dir=audio_dir,
                project_dir=project_dir,
            )
            print("[phase 2/5] Transcribing source recordings", flush=True)
            transcribe_project(project_dir=project_dir, project=project)
            print("[phase 3/5] Segmenting recordings", flush=True)
            segment_project(project_dir=project_dir, project=project)
            print("[phase 4/5] Transcribing temporary segments", flush=True)
            transcribe_segments_project(project_dir=project_dir, project=project)
            print("[phase 5/5] Aligning segments to script lines", flush=True)
            align_project(project_dir=project_dir, project=project)
            print("Pipeline complete.", flush=True)
            return project_dir

        self._start_worker(work, self._pipeline_finished)

    def show_progress(self, project_dir: Path) -> None:
        self._clear()
        outer = ttk.Frame(self.root, padding=30)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Creating project", style="Title.TLabel").pack(
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

    def _start_worker(
        self,
        function: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._worker_messages = messages

        def worker() -> None:
            writer = QueueWriter(messages)
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(
                    writer
                ):
                    result = function()
                messages.put(("done", (on_success, result)))
            except Exception as error:
                messages.put(("log", traceback.format_exc()))
                messages.put(("error", error))

        threading.Thread(target=worker, daemon=True).start()
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
                callback(result)
                finished = True
            elif kind == "error":
                self._stop_progress()
                self._worker_messages = None
                self._pipeline_failed(payload)
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
        candidate_columns = ("segment", "transcript", "score", "audio", "selection")
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
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.candidate_tree.grid(row=2, column=0, sticky="nsew")
        self.candidate_vertical_scrollbar.grid(row=2, column=1, sticky="ns")
        self.candidate_horizontal_scrollbar.grid(
            row=3,
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
            self.selected_context_label.configure(text="—")
            self.selected_line_label.configure(text="—")
            self.selected_acting_note_label.configure(text="—")
            self.candidate_description.configure(
                text="Select a line to review its candidates."
            )
            return

        self.selected_context_label.configure(text=line.get("context") or "—")
        self.selected_line_label.configure(text=line["line_text"] or "—")
        self.selected_acting_note_label.configure(
            text=line.get("acting_note") or "—"
        )
        uses_unmatched = _uses_unmatched_candidates(line)
        description = ""
        if uses_unmatched:
            description = (
                "Unmatched audible segments are shown for nonverbal lines."
                if line["type"] == "nonverbal"
                else (
                    "No alignment candidate was found; all unmatched audible "
                    "segments are shown."
                )
            )
        self.candidate_description.configure(text=description)

        candidates = (
            self.review_data["unmatched_segments"]
            if uses_unmatched
            else line["candidates"]
        )
        if uses_unmatched:
            candidates = sorted(
                candidates,
                key=lambda candidate: str(candidate["segment_id"]).casefold(),
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

    def _candidate_table_motion(self, event: tk.Event[Any]) -> None:
        segment_id = self.candidate_tree.identify_row(event.y)
        column = self.candidate_tree.identify_column(event.x)
        self.candidate_tree.configure(
            cursor=(
                "hand2"
                if segment_id and column in {"#4", "#5"}
                else ""
            )
        )

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
