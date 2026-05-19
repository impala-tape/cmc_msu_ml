from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
SOURCE_XLSX = ROOT / "Data_zad4_2026.xlsx"
OUT_DIR = ROOT / "prepared"

NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

CANONICAL_COLUMNS = {
    "MMBБ(MicexIndexCF)": "MICEX",
    "GAZP": "GAZP",
    "FEES(ФСК)": "FEES",
    "LKOH": "LKOH",
    "SBER03": "SBER",
    "ROSN": "ROSN",
    "VTBR": "VTBR",
}

ASSET_ORDER = ["GAZP", "ROSN", "LKOH", "FEES", "SBER", "VTBR"]
TRADING_DAYS_PER_YEAR = 252


def excel_serial_to_date(value: str) -> date:
    return (datetime(1899, 12, 30) + timedelta(days=float(value))).date()


def col_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - 64
    return idx - 1


def shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    out = []
    for si in root.findall("a:si", NS):
        out.append("".join((t.text or "") for t in si.iter(f"{{{NS['a']}}}t")))
    return out


def cell_value(cell: ET.Element, strings: list[str]) -> str | None:
    cell_type = cell.attrib.get("t")
    node = cell.find("a:v", NS)
    if cell_type == "s" and node is not None:
        return strings[int(node.text)]
    if cell_type == "inlineStr":
        return "".join((t.text or "") for t in cell.iter(f"{{{NS['a']}}}t"))
    return None if node is None else node.text


def row_values(row: ET.Element, strings: list[str]) -> list[str | None]:
    values: list[str | None] = []
    current = 0
    for cell in row.findall("a:c", NS):
        index = col_to_index(cell.attrib.get("r", "A1"))
        while current < index:
            values.append(None)
            current += 1
        values.append(cell_value(cell, strings))
        current = index + 1
    return values


def read_first_sheet(path: Path) -> list[list[str | None]]:
    with ZipFile(path) as zf:
        strings = shared_strings(zf)
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        relroot = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rels = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in relroot.findall(f"{{{PACKAGE_REL}}}Relationship")
        }
        sheet = workbook.find("a:sheets", NS)[0]
        rel_id = sheet.attrib[f"{{{NS['r']}}}id"]
        target = rels[rel_id]
        sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
        root = ET.fromstring(zf.read(sheet_path))
        rows = []
        for row in root.findall("a:sheetData/a:row", NS):
            values = row_values(row, strings)
            if any(value not in (None, "") for value in values):
                rows.append(values)
    return rows


def as_float(value: str | None) -> float:
    if value is None:
        raise ValueError("Empty numeric cell")
    return float(value.replace(",", "."))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not SOURCE_XLSX.exists():
        raise FileNotFoundError(SOURCE_XLSX)

    rows = read_first_sheet(SOURCE_XLSX)
    header = [str(value) if value is not None else "" for value in rows[0]]
    positions = {name: header.index(name) for name in CANONICAL_COLUMNS}

    quote_rows: list[dict[str, object]] = []
    for raw in rows[1:]:
        item: dict[str, object] = {
            "weekday": raw[0],
            "date": excel_serial_to_date(str(raw[1])).isoformat(),
        }
        for original, canonical in CANONICAL_COLUMNS.items():
            item[canonical] = as_float(raw[positions[original]])
        quote_rows.append(item)

    quote_fields = ["date", "weekday", "MICEX", *ASSET_ORDER]
    write_csv(OUT_DIR / "quotes.csv", quote_rows, quote_fields)

    return_rows: list[dict[str, object]] = []
    for prev, cur in zip(quote_rows[:-1], quote_rows[1:]):
        item = {"date": cur["date"]}
        for asset in ASSET_ORDER:
            item[asset] = float(cur[asset]) / float(prev[asset]) - 1.0
        return_rows.append(item)
    write_csv(OUT_DIR / "returns_simple.csv", return_rows, ["date", *ASSET_ORDER])

    expected_daily = {
        asset: sum(float(row[asset]) for row in return_rows) / len(return_rows)
        for asset in ASSET_ORDER
    }
    expected_annual = {
        asset: expected_daily[asset] * TRADING_DAYS_PER_YEAR
        for asset in ASSET_ORDER
    }
    estimate_rows = [
        {
            "asset": asset,
            "expected_daily": expected_daily[asset],
            "expected_annual": expected_annual[asset],
        }
        for asset in ASSET_ORDER
    ]
    write_csv(
        OUT_DIR / "expected_returns.csv",
        estimate_rows,
        ["asset", "expected_daily", "expected_annual"],
    )

    metadata = {
        "source_file": SOURCE_XLSX.name,
        "source_sheet": "Исх_данные",
        "source_rows_nonempty": len(rows),
        "quote_rows": len(quote_rows),
        "return_rows": len(return_rows),
        "date_from": quote_rows[0]["date"],
        "date_to": quote_rows[-1]["date"],
        "asset_order": ASSET_ORDER,
        "index_column": "MICEX",
        "trading_days_per_year": TRADING_DAYS_PER_YEAR,
        "note": (
            "The screenshot mentions 2016-04-01..2016-04-10, "
            "but Data_zad4_2026.xlsx contains 2010-09-01..2010-10-01."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Prepared {len(quote_rows)} quote rows and {len(return_rows)} return rows in {OUT_DIR}")


if __name__ == "__main__":
    main()
