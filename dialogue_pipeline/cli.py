from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .alignment import align_project
from .doctor import run_doctor
from .finalize import finalize_review
from .project import create_project, load_project
from .review import REVIEW_FILE_NAME
from .segmentation import refresh_project_audio, segment_project
from .transcription import transcribe_project, transcribe_segments_project
from .util import project_file_from_arg


def _project_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        required=True,
        help="Project directory or path to project.json.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dialogue-pipeline",
        description=(
            "Split long voice recordings, align takes to an Excel script, "
            "generate review data, and finalize selected assets."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a traceback when a command fails.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check Python packages and media tools.")

    init_parser = subparsers.add_parser(
        "init", help="Create a project configuration and source inventory."
    )
    init_parser.add_argument("--workbook", required=True, type=Path)
    init_parser.add_argument("--audio-dir", required=True, type=Path)
    init_parser.add_argument("--project-dir", required=True, type=Path)
    init_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Refresh source metadata while backing up and preserving existing "
            "settings and reviewed session mappings."
        ),
    )

    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe configured source recordings."
    )
    _project_argument(transcribe_parser)
    transcribe_parser.add_argument(
        "--session",
        action="append",
        default=[],
        help="Process only this session ID; may be specified multiple times.",
    )
    transcribe_parser.add_argument("--force", action="store_true")
    transcribe_parser.add_argument("--model")
    transcribe_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"]
    )

    segment_parser = subparsers.add_parser(
        "segment", help="Create deterministic temporary WAV segments."
    )
    _project_argument(segment_parser)
    segment_parser.add_argument(
        "--session",
        action="append",
        default=[],
        help="Process only this session ID; may be specified multiple times.",
    )
    segment_parser.add_argument("--force", action="store_true")

    refresh_audio_parser = subparsers.add_parser(
        "refresh-audio",
        help=(
            "Re-probe changed source recordings and re-cut every stored "
            "segment without transcription or alignment."
        ),
    )
    _project_argument(refresh_audio_parser)

    segment_transcribe_parser = subparsers.add_parser(
        "transcribe-segments",
        help="Independently transcribe every temporary WAV segment.",
    )
    _project_argument(segment_transcribe_parser)
    segment_transcribe_parser.add_argument(
        "--session",
        action="append",
        default=[],
        help="Process only this session ID; may be specified multiple times.",
    )
    segment_transcribe_parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="Process only this exact base Segment ID; may be repeated.",
    )
    segment_transcribe_parser.add_argument("--force", action="store_true")
    segment_transcribe_parser.add_argument("--model")
    segment_transcribe_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"]
    )

    align_parser = subparsers.add_parser(
        "align",
        help=f"Align segments and generate {REVIEW_FILE_NAME}.",
    )
    _project_argument(align_parser)
    align_parser.add_argument(
        "--session",
        action="append",
        default=[],
        help="Align only this session ID; may be specified multiple times.",
    )

    process_parser = subparsers.add_parser(
        "process",
        help=(
            "Run recording ASR, segmentation, independent segment ASR, "
            "and alignment."
        ),
    )
    _project_argument(process_parser)
    process_parser.add_argument(
        "--session",
        action="append",
        default=[],
        help="Process only this session ID; may be specified multiple times.",
    )
    process_parser.add_argument("--force-transcription", action="store_true")
    process_parser.add_argument("--force-segmentation", action="store_true")
    process_parser.add_argument(
        "--force-segment-transcription",
        action="store_true",
    )
    process_parser.add_argument("--model")
    process_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"]
    )

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Copy user-selected segments to final script filenames.",
    )
    _project_argument(finalize_parser)
    finalize_parser.add_argument(
        "--review",
        type=Path,
        help=f"Review data; defaults to <project>/{REVIEW_FILE_NAME}.",
    )
    finalize_parser.add_argument("--output", required=True, type=Path)
    finalize_parser.add_argument("--overwrite", action="store_true")
    finalize_parser.add_argument("--allow-incomplete", action="store_true")
    finalize_parser.add_argument(
        "--allow-segment-reuse",
        action="store_true",
        default=None,
        help=(
            "Allow one selected segment to be copied to multiple target "
            "filenames; defaults to export.allow_segment_reuse."
        ),
    )
    finalize_parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_project_argument(value: str) -> tuple[Path, dict[str, Any]]:
    return load_project(project_file_from_arg(value))


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        checks, ok = run_doctor()
        for check in checks:
            print(
                f"{check['status'].upper():7} {check['check']}: {check['detail']}"
            )
        return 0 if ok else 1

    if args.command == "init":
        project = create_project(
            workbook_path=args.workbook,
            audio_dir=args.audio_dir,
            project_dir=args.project_dir,
            force=args.force,
        )
        needs_review = [
            session["id"]
            for session in project["sessions"]
            if session.get("needs_mapping_review")
        ]
        _print_json(
            {
                "project": str((args.project_dir / "project.json").resolve()),
                "sessions": len(project["sessions"]),
                "mapping_review_required": needs_review,
            }
        )
        return 0

    project_dir, project = _load_project_argument(args.project)
    if args.command == "transcribe":
        outputs = transcribe_project(
            project_dir=project_dir,
            project=project,
            session_filter=set(args.session) or None,
            force=args.force,
            model_override=args.model,
            device_override=args.device,
        )
        _print_json({"transcripts": outputs})
        return 0
    if args.command == "segment":
        output = segment_project(
            project_dir=project_dir,
            project=project,
            session_filter=set(args.session) or None,
            force=args.force,
        )
        _print_json({"segments_manifest": output})
        return 0
    if args.command == "refresh-audio":
        _print_json(
            refresh_project_audio(
                project_dir=project_dir,
                project=project,
            )
        )
        return 0
    if args.command == "transcribe-segments":
        output = transcribe_segments_project(
            project_dir=project_dir,
            project=project,
            session_filter=set(args.session) or None,
            segment_filter=set(args.segment) or None,
            force=args.force,
            model_override=args.model,
            device_override=args.device,
        )
        _print_json({"segments_manifest": output})
        return 0
    if args.command == "align":
        _print_json(
            align_project(
                project_dir=project_dir,
                project=project,
                session_filter=set(args.session) or None,
            )
        )
        return 0
    if args.command == "process":
        session_filter = set(args.session) or None
        transcribe_project(
            project_dir=project_dir,
            project=project,
            session_filter=session_filter,
            force=args.force_transcription,
            model_override=args.model,
            device_override=args.device,
        )
        segment_project(
            project_dir=project_dir,
            project=project,
            session_filter=session_filter,
            force=args.force_segmentation,
        )
        transcribe_segments_project(
            project_dir=project_dir,
            project=project,
            session_filter=session_filter,
            force=args.force_segment_transcription,
            model_override=args.model,
            device_override=args.device,
        )
        _print_json(
            align_project(
                project_dir=project_dir,
                project=project,
                session_filter=session_filter,
                segment_model_override=args.model,
                segment_device_override=args.device,
            )
        )
        return 0
    if args.command == "finalize":
        review_path = (
            args.review.resolve()
            if args.review
            else project_dir / REVIEW_FILE_NAME
        )
        result = finalize_review(
            project_dir=project_dir,
            project=project,
            review_path=review_path,
            output_dir=args.output.resolve(),
            overwrite=args.overwrite,
            allow_incomplete=args.allow_incomplete,
            allow_segment_reuse=args.allow_segment_reuse,
            dry_run=args.dry_run,
        )
        _print_json(result)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
