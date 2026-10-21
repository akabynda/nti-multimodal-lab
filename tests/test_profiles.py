import pandas as pd
import pytest

from nti_lab.research import profile_roster


def workbook_fixture(monkeypatch, tmp_path, frame):
    class Excel:
        sheet_names = ["Лист1"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(pd, "ExcelFile", lambda path: Excel())
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: frame.copy())
    path = tmp_path / "workbook-source.bin"
    path.write_bytes(b"immutable source fixture")
    return path


def test_profiles_join_by_isu_preserving_zero_missing_values_and_source_rows(monkeypatch, tmp_path):
    frame = pd.DataFrame([
        {"Таб.": 123456.0, "Естественный": "Первый", "Эмоц+": 4, "Эмоц-": 4, "Эмоц_Баланс": 0},
        {"Таб.": None},
        {"Таб.": " 654321 ", "Естественный": "Второй", "Выраженность": 2.06},
    ])
    path = workbook_fixture(monkeypatch, tmp_path, frame)
    roster, source = profile_roster(path, [
        {"participant_id": "second", "tab_number": "654321", "tab_number_evidence": "user"},
        {"participant_id": "first", "tab_number": "123456"},
        {"participant_id": "absent", "tab_number": "999999"},
        {"participant_id": "unknown"},
    ])
    indexed = roster.set_index("participant_id")
    assert len(source) == 2
    assert indexed.loc["second", "profile_natural_raw"] == "Второй"
    assert indexed.loc["second", "expression"] == 2.06
    assert indexed.loc["second", "profile_source_row"] == 4
    assert indexed.loc["first", "profile_source_row"] == 2
    assert indexed.loc["first", "emotion_balance"] == 0
    assert pd.isna(indexed.loc["second", "emotion_balance"])
    assert indexed.loc["absent", "profile_link_status"] == "requested_id_not_found"
    assert pd.isna(indexed.loc["absent", "profile_source_row"])
    assert indexed.loc["unknown", "profile_link_status"] == "unlinked"
    assert indexed.loc["second", "profile_source_sheet"] == "Лист1"
    assert indexed.loc["second", "profile_link_evidence"] == "user"
    assert path.read_bytes() == b"immutable source fixture"


def test_profiles_reject_duplicate_source_identifiers(monkeypatch, tmp_path):
    path = workbook_fixture(monkeypatch, tmp_path, pd.DataFrame({"Таб.": [123456, "123456.0"]}))
    with pytest.raises(ValueError, match="Duplicate tab"):
        profile_roster(path, [{"participant_id": "A", "tab_number": "123456"}])


def test_profiles_reject_two_participants_assigned_one_isu(monkeypatch, tmp_path):
    path = workbook_fixture(monkeypatch, tmp_path, pd.DataFrame({"Таб.": [123456]}))
    with pytest.raises(ValueError, match="Duplicate participant"):
        profile_roster(path, [{"participant_id": pid, "tab_number": "123456"} for pid in ["A", "B"]])


@pytest.mark.parametrize("value", [123456.7, "123x56"])
def test_profiles_do_not_truncate_invalid_identifiers(monkeypatch, tmp_path, value):
    path = workbook_fixture(monkeypatch, tmp_path, pd.DataFrame({"Таб.": [value]}))
    with pytest.raises(ValueError, match="Invalid ISU"):
        profile_roster(path, [])
