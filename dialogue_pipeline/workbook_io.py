from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


HEADER_ALIASES = {
    "quest": {"quest"},
    "context": {"context"},
    "line": {"line to speak", "line", "dialogue", "text"},
    "acting_note": {"acting note", "acting notes", "note"},
    "emotion": {"facial emotion", "emotion"},
    "filename": {"filename", "file name", "output filename"},
}


def _normalized_header(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _find_header_row(worksheet) -> tuple[int, dict[str, int]]:
    for row_number in range(1, min(worksheet.max_row, 20) + 1):
        found: dict[str, int] = {}
        for column_number in range(1, worksheet.max_column + 1):
            value = _normalized_header(worksheet.cell(row_number, column_number).value)
            for canonical, aliases in HEADER_ALIASES.items():
                if value in aliases:
                    found[canonical] = column_number
        if "line" in found and "filename" in found:
            return row_number, found
    raise ValueError(f"Could not find dialogue headers in sheet {worksheet.title!r}")


def parse_workbook(path: Path) -> dict[str, Any]:
    workbook = load_workbook(
        filename=path,
        read_only=False,
        data_only=True,
        keep_links=False,
    )
    lines: list[dict[str, Any]] = []
    sheets: list[dict[str, Any]] = []
    filenames: list[str] = []

    try:
        for sheet_index, worksheet in enumerate(workbook.worksheets):
            header_row, columns = _find_header_row(worksheet)
            voice_header = worksheet.cell(1, 2).value
            sheet_lines: list[str] = []
            current_quest = ""

            for row_number in range(header_row + 1, worksheet.max_row + 1):
                def get(field: str) -> str:
                    column = columns.get(field)
                    if not column:
                        return ""
                    value = worksheet.cell(row_number, column).value
                    return str(value).strip() if value is not None else ""

                quest_raw = get("quest")
                if quest_raw:
                    current_quest = quest_raw
                spoken_line = get("line")
                filename = get("filename")
                if not spoken_line and not filename:
                    continue
                if not spoken_line or not filename:
                    raise ValueError(
                        f"Incomplete dialogue row at {worksheet.title}!{row_number}: "
                        f"line={spoken_line!r}, filename={filename!r}"
                    )

                line_id = f"{worksheet.title}::R{row_number}"
                record = {
                    "line_id": line_id,
                    "sheet": worksheet.title,
                    "sheet_index": sheet_index,
                    "excel_row": row_number,
                    "quest": current_quest,
                    "context": get("context"),
                    "line": spoken_line,
                    "acting_note": get("acting_note"),
                    "emotion": get("emotion"),
                    "target_filename": filename,
                }
                lines.append(record)
                sheet_lines.append(line_id)
                filenames.append(filename)

            sheets.append(
                {
                    "name": worksheet.title,
                    "index": sheet_index,
                    "voice_header": str(voice_header or "").strip(),
                    "line_ids": sheet_lines,
                    "line_count": len(sheet_lines),
                }
            )
    finally:
        workbook.close()

    duplicates = sorted(name for name, count in Counter(filenames).items() if count > 1)
    if duplicates:
        raise ValueError(
            "Duplicate target filenames in workbook: " + ", ".join(duplicates[:20])
        )

    return {
        "schema_version": 1,
        "source_workbook": str(path.resolve()),
        "line_count": len(lines),
        "sheet_count": len(sheets),
        "sheets": sheets,
        "lines": lines,
    }


def lines_for_session(
    source_data: dict[str, Any], session: dict[str, Any]
) -> list[dict[str, Any]]:
    sheet_order = {name: index for index, name in enumerate(session.get("sheets", []))}
    allowed_rows = set(session.get("excel_rows") or [])
    allowed_ids = set(session.get("line_ids") or [])

    selected = []
    for line in source_data["lines"]:
        if allowed_ids and line["line_id"] not in allowed_ids:
            continue
        if line["sheet"] not in sheet_order:
            continue
        if allowed_rows and line["excel_row"] not in allowed_rows:
            continue
        selected.append(line)

    return sorted(
        selected,
        key=lambda line: (
            sheet_order[line["sheet"]],
            line["excel_row"],
        ),
    )
