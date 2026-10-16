import json
import uuid
import zipfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from complete_automation.core.datasets import (
    DataFactory,
    DataLoader,
    load_csv,
    load_json,
    load_xlsx,
    load_yaml,
    read_xlsx_rows,
    validate_records,
)
from complete_automation.core.exceptions import TestDataError

REPO_DATA = Path(__file__).resolve().parents[3] / "data"
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


class User(BaseModel):
    name: str
    age: int


def write_xlsx(path: Path, rows: dict[int, dict[str, str]], *, inline: bool = False) -> Path:
    """Minimal workbook; ``rows`` maps row number to {cell column: text}. Missing cells stay absent."""
    strings: list[str] = []
    xml_rows = []
    for number, cells in sorted(rows.items()):
        xml_cells = []
        for column, text in cells.items():
            if inline:
                xml_cells.append(f'<c r="{column}{number}" t="inlineStr"><is><t>{text}</t></is></c>')
            else:
                strings.append(text)
                xml_cells.append(f'<c r="{column}{number}" t="s"><v>{len(strings) - 1}</v></c>')
        xml_rows.append(f'<row r="{number}">{"".join(xml_cells)}</row>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{NS}"><sheetData>{"".join(xml_rows)}</sheetData></worksheet>',
        )
        if not inline:
            archive.writestr(
                "xl/sharedStrings.xml",
                f'<sst xmlns="{NS}">' + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>",
            )
    return path


def test_json_yaml_and_csv_load(tmp_path: Path) -> None:
    (tmp_path / "d.json").write_text('[{"name": "a", "age": 1}]', encoding="utf-8")
    (tmp_path / "d.yaml").write_text("- name: a\n  age: 1\n", encoding="utf-8")
    (tmp_path / "d.csv").write_text("name,age\na,1\nb,2\n", encoding="utf-8")

    assert load_json(tmp_path / "d.json") == [{"name": "a", "age": 1}]
    assert load_yaml(tmp_path / "d.yaml") == [{"name": "a", "age": 1}]
    assert load_csv(tmp_path / "d.csv") == [{"name": "a", "age": "1"}, {"name": "b", "age": "2"}]


def test_missing_files_raise(tmp_path: Path) -> None:
    for loader in (load_json, load_yaml, load_csv, load_xlsx):
        with pytest.raises(TestDataError, match="not found"):
            loader(tmp_path / "missing")


def test_malformed_json_reports_position(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text('{"a": 1,\n "b": }', encoding="utf-8")

    with pytest.raises(TestDataError, match="line 2"):
        load_json(tmp_path / "bad.json")


def test_malformed_yaml_and_unsafe_tags_raise(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("a: [unclosed\n", encoding="utf-8")
    (tmp_path / "unsafe.yaml").write_text("!!python/object/apply:os.system ['echo hi']\n", encoding="utf-8")

    for name in ("bad.yaml", "unsafe.yaml"):
        with pytest.raises(TestDataError, match="malformed YAML"):
            load_yaml(tmp_path / name)


@pytest.mark.parametrize(
    "text",
    ["name,age\na,1,extra\n", "name,age\na\n", "name,name\na,b\n", "name,\na,b\n", ""],
)
def test_malformed_csv_is_rejected(tmp_path: Path, text: str) -> None:
    (tmp_path / "bad.csv").write_text(text, encoding="utf-8")

    with pytest.raises(TestDataError):
        load_csv(tmp_path / "bad.csv")


def test_xlsx_keeps_blank_cells_in_place(tmp_path: Path) -> None:
    path = write_xlsx(tmp_path / "gap.xlsx", {1: {"A": "a", "B": "b", "C": "c"}, 2: {"A": "1", "C": "3"}})

    assert read_xlsx_rows(path) == [["a", "b", "c"], ["1", "", "3"]]
    assert load_xlsx(path) == [{"a": "1", "b": "", "c": "3"}]


def test_xlsx_inline_strings(tmp_path: Path) -> None:
    path = write_xlsx(tmp_path / "inline.xlsx", {1: {"A": "k"}, 2: {"A": "v"}}, inline=True)

    assert load_xlsx(path) == [{"k": "v"}]


def test_repository_dataset_loads_by_header() -> None:
    records = load_xlsx(REPO_DATA / "llm" / "eval_cases.xlsx")
    twin = json.loads((REPO_DATA / "llm" / "eval_cases.json").read_text(encoding="utf-8"))

    assert records == twin


def test_non_workbooks_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "fake.xlsx").write_bytes(b"not a zip")

    with pytest.raises(TestDataError, match="not a readable xlsx"):
        read_xlsx_rows(tmp_path / "fake.xlsx")


def test_loader_dispatches_by_extension_and_validates(tmp_path: Path) -> None:
    (tmp_path / "users.json").write_text('[{"name": "a", "age": 30}]', encoding="utf-8")
    loader = DataLoader(tmp_path)

    assert loader.load_records("users.json", User) == [User(name="a", age=30)]


@pytest.mark.parametrize("relative", ["../outside.json", "/etc/passwd", "a/../../x.json"])
def test_loader_refuses_paths_outside_its_root(tmp_path: Path, relative: str) -> None:
    with pytest.raises(TestDataError, match="outside"):
        DataLoader(tmp_path / "data").load(relative)


def test_loader_rejects_unknown_formats(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")

    with pytest.raises(TestDataError, match="unsupported test data format"):
        DataLoader(tmp_path).load("x.txt")


def test_schema_errors_name_record_and_field_but_not_value() -> None:
    with pytest.raises(TestDataError) as raised:
        validate_records([{"name": "a", "age": 1}, {"name": "b", "age": "secret-not-int"}], User)

    message = str(raised.value)
    assert "record 1, age" in message
    assert "secret-not-int" not in message


def test_validation_requires_a_list() -> None:
    with pytest.raises(TestDataError, match="expected a list"):
        validate_records({"name": "a"}, User)


def test_factories_are_reproducible() -> None:
    def draw(factory: DataFactory) -> list:
        return [
            factory.integer(1, 100),
            factory.string(8),
            factory.email(),
            factory.choice(["x", "y", "z"]),
            factory.uuid(),
            factory.sequence("user"),
            factory.sequence("user"),
        ]

    first, second, other = draw(DataFactory(42)), draw(DataFactory(42)), draw(DataFactory(7))

    assert first == second
    assert first != other
    assert first[5:] == ["user-0001", "user-0002"]
    assert isinstance(first[4], uuid.UUID) and first[4].version == 4
    assert first[2].endswith("@example.test")


def test_factory_rejects_impossible_requests() -> None:
    with pytest.raises(ValueError):
        DataFactory(1).choice([])
    with pytest.raises(ValueError):
        DataFactory(1).string(-1)
