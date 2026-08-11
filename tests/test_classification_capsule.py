import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "recompute_classification_metrics.py"
SPEC = importlib.util.spec_from_file_location("classification_metrics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def test_binary_metrics_include_tie_handling() -> None:
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.8, 0.8, 0.4]
    assert METRICS.average_precision(labels, scores) == pytest.approx(7 / 12)
    assert METRICS.roc_auc(labels, scores) == pytest.approx(0.625)


def test_capsule_sources_use_existing_zenodo_concept() -> None:
    manifest = json.loads(
        (ROOT / "manifests" / "classification_capsule_sources.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["zenodo"] == {
        "policy": "new_version_of_existing_record",
        "existing_record_id": 15612048,
        "existing_version_doi": "10.5281/zenodo.15612048",
        "concept_doi": "10.5281/zenodo.15612047",
        "new_version_doi": "pending_until_published",
    }
    assert [group["eligible_rows"] for group in manifest["groups"]] == [
        2335,
        7684,
        39311,
    ]
    assert all(
        not source["source_relative_path"].startswith("/")
        for group in manifest["groups"]
        for source in group["sources"].values()
    )
