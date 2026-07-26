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


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(
            self,
            borderwidth=0,
            highlightthickness=0,
            background="#f8fafc",
        )
        scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
        )
        self.content = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window(
            (0, 0),
            window=self.content,
            anchor="nw",
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.content.bind("<Configure>", self._content_changed)
        self.canvas.bind("<Configure>", self._canvas_changed)
        self.canvas.bind_all("<MouseWheel>", self._mousewheel)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def clear(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()
        self.canvas.yview_moveto(0)

    def destroy(self) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        super().destroy()

    def _content_changed(self, _event: tk.Event[Any]) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _canvas_changed(self, event: tk.Event[Any]) -> None:
        self.canvas.itemconfigure(self.window, width=event.width)

    def _mousewheel(self, event: tk.Event[Any]) -> None:
        widget = self.winfo_containing(
            self.winfo_pointerx(),
            self.winfo_pointery(),
        )
        while widget is not None:
            if widget is self:
                self.canvas.yview_scroll(int(-event.delta / 120), "units")
                return
            widget = widget.master


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
        self.sort_field = tk.StringVar(value="Status")
        self.sort_descending = tk.BooleanVar(value=False)
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
        ttk.Label(controls, text="Sort:").pack(side="left")
        sort_box = ttk.Combobox(
            controls,
            textvariable=self.sort_field,
            values=["Status", "Sheet", "Line", "Target file", "Type"],
            state="readonly",
            width=18,
        )
        sort_box.pack(side="left", padx=(6, 10))
        sort_box.bind("<<ComboboxSelected>>", lambda _event: self.render_lines())
        ttk.Checkbutton(
            controls,
            text="Descending",
            variable=self.sort_descending,
            command=self.render_lines,
        ).pack(side="left")
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

        line_columns = ("sheet", "line", "target", "type", "status")
        self.line_tree = ttk.Treeview(
            left,
            columns=line_columns,
            show="headings",
            selectmode="browse",
        )
        line_headings = {
            "sheet": ("Sheet", 115),
            "line": ("Line text", 410),
            "target": ("Target file", 190),
            "type": ("Type", 90),
            "status": ("Status", 145),
        }
        for column, (heading, width) in line_headings.items():
            self.line_tree.heading(column, text=heading)
            self.line_tree.column(
                column,
                width=width,
                minwidth=70,
                stretch=column in {"line", "target"},
            )
        for status, color in STATUS_COLORS.items():
            self.line_tree.tag_configure(status, background=color)
        line_scrollbar = ttk.Scrollbar(
            left,
            orient="vertical",
            command=self.line_tree.yview,
        )
        self.line_tree.configure(yscrollcommand=line_scrollbar.set)
        self.line_action_bar = ttk.Frame(left)
        self.line_action_bar.pack(side="bottom", fill="x", pady=(8, 0))
        self.play_selected_button = ttk.Button(
            self.line_action_bar,
            text="Play selected segment",
            command=self.play_selected_line,
        )
        self.line_tree.pack(side="left", fill="both", expand=True)
        line_scrollbar.pack(side="right", fill="y")
        self.line_tree.bind("<<TreeviewSelect>>", self._tree_line_selected)
        self.candidate_list = ScrollableFrame(right)
        self.candidate_list.pack(fill="both", expand=True)
        self.render_lines()
        self.render_candidates()

    def _filtered_lines(self) -> list[dict[str, Any]]:
        assert self.review_data is not None
        lines = list(self.review_data["lines"])
        selected_status = self.status_filter.get()
        if selected_status != "All":
            lines = [
                line for line in lines if line["status"] == selected_status
            ]
        keys: dict[str, Callable[[dict[str, Any]], Any]] = {
            "Status": lambda line: (line["status"], line["sheet"], line["excel_row"]),
            "Sheet": lambda line: (line["sheet"], line["excel_row"]),
            "Line": lambda line: line["line_text"].casefold(),
            "Target file": lambda line: line["target_filename"].casefold(),
            "Type": lambda line: (line["type"], line["sheet"], line["excel_row"]),
        }
        return sorted(
            lines,
            key=keys[self.sort_field.get()],
            reverse=self.sort_descending.get(),
        )

    def render_lines(self) -> None:
        assert self.review_data is not None
        children = self.line_tree.get_children()
        if children:
            self.line_tree.delete(*children)
        lines = self._filtered_lines()
        self.status_text.set(
            f"{len(lines)} of {len(self.review_data['lines'])} lines"
        )
        for line in lines:
            self.line_tree.insert(
                "",
                "end",
                iid=line["line_id"],
                values=(
                    line["sheet"],
                    line["line_text"],
                    line["target_filename"],
                    line["type"],
                    line["status"],
                ),
                tags=(line["status"],),
            )
        visible_ids = {line["line_id"] for line in lines}
        if self.selected_line_id in visible_ids:
            self.line_tree.selection_set(self.selected_line_id)
            self.line_tree.focus(self.selected_line_id)
            self.line_tree.see(self.selected_line_id)
        self._update_selected_play_button()

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
        self._update_selected_play_button()
        self.render_candidates()

    def _update_selected_play_button(self) -> None:
        line = self._selected_line()
        if line and line.get("selected_segment_id"):
            if not self.play_selected_button.winfo_manager():
                self.play_selected_button.pack(side="right")
        else:
            self.play_selected_button.pack_forget()

    def play_selected_line(self) -> None:
        line = self._selected_line()
        if line and line.get("selected_segment_id"):
            self.play_segment(str(line["selected_segment_id"]))

    def render_candidates(self) -> None:
        assert self.review_data is not None
        self.candidate_list.clear()
        line = self._selected_line()
        if line is None:
            ttk.Label(
                self.candidate_list.content,
                text="Select a line to review its candidates.",
                style="Muted.TLabel",
                padding=16,
            ).pack(anchor="w")
            return

        description = (
            "Unmatched audible segments are shown for nonverbal lines."
            if line["type"] == "nonverbal"
            else line["line_text"]
        )
        ttk.Label(
            self.candidate_list.content,
            text=description,
            wraplength=600,
            style="Muted.TLabel",
            padding=(4, 4, 4, 10),
        ).pack(anchor="w", fill="x")

        candidates = (
            self.review_data["unmatched_segments"]
            if line["type"] == "nonverbal"
            else line["candidates"]
        )
        candidates = sorted(
            candidates,
            key=lambda candidate: float(candidate.get("score", 0.0)),
            reverse=True,
        )
        if not candidates:
            ttk.Label(
                self.candidate_list.content,
                text="No candidate segments are available.",
                padding=16,
            ).pack(anchor="w")
            return

        header = tk.Frame(self.candidate_list.content, background="#334155")
        header.pack(fill="x", pady=(0, 2))
        for column, (text, weight) in enumerate(
            [("Transcript", 5), ("Score", 1), ("Audio", 1), ("Selection", 1)]
        ):
            header.grid_columnconfigure(column, weight=weight)
            tk.Label(
                header,
                text=text,
                background="#334155",
                foreground="white",
                font=("Segoe UI", 9, "bold"),
                padx=6,
                pady=6,
            ).grid(row=0, column=column, sticky="nsew")

        selected_id = line.get("selected_segment_id")
        for candidate in candidates:
            is_selected = candidate["segment_id"] == selected_id
            background = "#bbf7d0" if is_selected else "#ffffff"
            row = tk.Frame(
                self.candidate_list.content,
                background=background,
                highlightthickness=2 if is_selected else 1,
                highlightbackground="#16a34a" if is_selected else "#cbd5e1",
            )
            row.pack(fill="x", pady=2)
            for column, weight in enumerate([5, 1, 1, 1]):
                row.grid_columnconfigure(column, weight=weight)
            transcript = str(candidate.get("transcript") or "[No transcript]")
            segment_id = str(candidate["segment_id"])
            tk.Label(
                row,
                text=f"{transcript}\n{segment_id}",
                background=background,
                anchor="w",
                justify="left",
                wraplength=520,
                padx=7,
                pady=7,
            ).grid(row=0, column=0, sticky="nsew")
            tk.Label(
                row,
                text=f"{float(candidate.get('score', 0.0)):.1f}",
                background=background,
                padx=7,
                pady=7,
            ).grid(row=0, column=1, sticky="nsew")
            ttk.Button(
                row,
                text="Play",
                command=lambda value=segment_id: self.play_segment(value),
                width=7,
            ).grid(row=0, column=2, padx=5, pady=5)
            ttk.Button(
                row,
                text="Unselect" if is_selected else "Select",
                command=lambda value=segment_id: self.toggle_candidate(value),
                width=9,
            ).grid(row=0, column=3, padx=5, pady=5)

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
            line["status"] = "REVIEW"
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
