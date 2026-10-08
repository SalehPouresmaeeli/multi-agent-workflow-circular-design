### Tests for src/tools/excel_sheet_to_json.py — converting the guideline sheet to JSON.
### Each test builds a small Excel file in a temporary folder (tmp_path), so the real files in docs/ are never used.

import json
import pandas as pd
from src.tools.excel_sheet_to_json import excel_to_json


def write_sheet(path, rows):
    pd.DataFrame(rows).to_excel(path, index=False, engine="openpyxl")


def test_rows_become_json_records(tmp_path):
    excel_file = tmp_path / "guidelines.xlsx"
    json_file = tmp_path / "guidelines.json"
    write_sheet(excel_file, [
        {"DfReX": "Remanufacturing", "Ref": "[4]", "Relevance": "High"},
        {"DfReX": "Repair", "Ref": "[8]", "Relevance": "Medium"},
    ])

    excel_to_json(excel_file, json_file)

    assert json.loads(json_file.read_text(encoding="utf-8")) == [
        {"DfReX": "Remanufacturing", "Ref": "[4]", "Relevance": "High"},
        {"DfReX": "Repair", "Ref": "[8]", "Relevance": "Medium"},
    ]


def test_non_breaking_hyphens_are_replaced(tmp_path):
    excel_file = tmp_path / "guidelines.xlsx"
    json_file = tmp_path / "guidelines.json"
    write_sheet(excel_file, [{"Design criteria": "Use non‑permanent, corrosion‑resistant joints"}])

    excel_to_json(excel_file, json_file)

    text = json.loads(json_file.read_text(encoding="utf-8"))[0]["Design criteria"]
    assert text == "Use non-permanent, corrosion-resistant joints"


def test_other_special_characters_are_kept(tmp_path):
    excel_file = tmp_path / "guidelines.xlsx"
    json_file = tmp_path / "guidelines.json"
    write_sheet(excel_file, [{"Design criteria": "Prefer aluminium — avoid PVC (≥ 5 °C)"}])

    excel_to_json(excel_file, json_file)

    raw = json_file.read_text(encoding="utf-8")
    assert "Prefer aluminium — avoid PVC (≥ 5 °C)" in raw     # written as real characters, not \u escapes


def test_missing_excel_file_reports_error_and_writes_nothing(tmp_path, capsys):
    json_file = tmp_path / "guidelines.json"

    excel_to_json(tmp_path / "does_not_exist.xlsx", json_file)

    assert "could not be found" in capsys.readouterr().out
    assert not json_file.exists()
