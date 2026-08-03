from pathlib import Path

from xdl.analysis import (
    flatten_mapping,
    records_to_rows,
    write_analysis_bundle,
)


class DummyRecord:
    def __init__(self, name: str, score: float) -> None:
        self.name = name
        self.score = score

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "metrics": {"score": self.score},
        }


def test_flatten_mapping_flattens_nested_dict() -> None:
    flattened = flatten_mapping({"a": {"b": 1}, "c": 2})

    assert flattened == {"a.b": 1, "c": 2}


def test_records_to_rows_accepts_to_dict_objects() -> None:
    rows = records_to_rows([DummyRecord("probe", 0.75)])

    assert rows == [{"name": "probe", "metrics.score": 0.75}]


def test_write_analysis_bundle_writes_expected_files(tmp_path: Path) -> None:
    outputs = write_analysis_bundle(
        tmp_path,
        report_name="analysis_summary",
        summary={"top1": 0.9},
        records=[DummyRecord("concept", 0.8)],
        markdown_sections={"summary": {"top1": 0.9}},
    )

    assert outputs["json"].exists()
    assert outputs["csv"].exists()
    assert outputs["md"].exists()
