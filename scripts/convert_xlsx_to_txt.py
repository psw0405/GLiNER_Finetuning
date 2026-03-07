"""
Convert an Excel (.xlsx) file containing sentences to a plain text file.

Each non-empty cell value from the target column is written as one line.

Usage:
    python scripts/convert_xlsx_to_txt.py \
        --input  data/dataset_sentences.xlsx \
        --output data/raw/dataset_sentences.txt \
        --column sentence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openpyxl


def convert_xlsx_to_txt(
    input_path: Path,
    output_path: Path,
    column: str | int | None,
    sheet: str | int | None,
    skip_header: bool,
) -> int:
    """
    Read *input_path* (xlsx) and write sentences to *output_path* (txt).

    Parameters
    ----------
    column:
        Column header name (str) or 1-based column index (int).
        When None the first column is used.
    sheet:
        Sheet name (str) or 0-based sheet index (int).
        When None the active (first) sheet is used.
    skip_header:
        When True the first row is treated as a header and skipped.

    Returns
    -------
    int
        Number of sentences written.
    """
    wb = openpyxl.load_workbook(input_path, read_only=True, data_only=True)

    if sheet is None:
        ws = wb.active
    elif isinstance(sheet, int):
        ws = wb.worksheets[sheet]
    else:
        ws = wb[sheet]

    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return 0

    # Resolve column index (0-based internally)
    if column is None:
        col_idx = 0
    elif isinstance(column, int):
        col_idx = column - 1  # user provides 1-based
    else:
        # Search header row for matching column name
        header_row = rows[0]
        col_idx = None
        for i, cell in enumerate(header_row):
            if cell is not None and str(cell).strip() == column:
                col_idx = i
                break
        if col_idx is None:
            available = [str(c) for c in header_row if c is not None]
            raise ValueError(
                f"Column '{column}' not found in header row. "
                f"Available columns: {available}"
            )

    # Always skip the header row when column is identified by name (the header
    # row was consumed for the column-name lookup above).  For numeric / default
    # column selection, respect the caller's skip_header flag.
    if isinstance(column, str):
        start_row = 1
    else:
        start_row = 1 if skip_header else 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows[start_row:]:
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            if cell is None:
                continue
            text = str(cell).strip()
            if not text:
                continue
            fh.write(text + "\n")
            written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an xlsx file of sentences to a plain-text file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/dataset_sentences.xlsx"),
        help="Path to the input .xlsx file (default: data/dataset_sentences.xlsx)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to the output .txt file. "
            "Defaults to <input stem>.txt in data/raw/."
        ),
    )
    parser.add_argument(
        "--column",
        default=None,
        help=(
            "Column to extract: header name (str) or 1-based index (int). "
            "Defaults to the first column."
        ),
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Sheet name or 0-based sheet index. Defaults to the active sheet.",
    )
    parser.add_argument(
        "--no-skip-header",
        dest="skip_header",
        action="store_false",
        default=True,
        help="Do not skip the first row even when --column is a numeric index.",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path: Path = args.output or Path("data/raw") / (input_path.stem + ".txt")

    # Parse --column as int if it looks like a number
    column: str | int | None = args.column
    if column is not None:
        try:
            column = int(column)
        except ValueError:
            pass  # keep as str

    # Parse --sheet as int if it looks like a number
    sheet: str | int | None = args.sheet
    if sheet is not None:
        try:
            sheet = int(sheet)
        except ValueError:
            pass  # keep as str

    count = convert_xlsx_to_txt(
        input_path=input_path,
        output_path=output_path,
        column=column,
        sheet=sheet,
        skip_header=args.skip_header,
    )

    print(f"written={count} output={output_path}")


if __name__ == "__main__":
    main()
