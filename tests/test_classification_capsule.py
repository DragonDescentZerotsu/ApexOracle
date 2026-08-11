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
        "new_version_doi": "10.5281/zenodo.21882300",
        "new_version_record_id": 21882300,
        "status": "released",
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


def test_zenodo_v2_release_manifest_fixes_public_names_and_boundaries() -> None:
    manifest = json.loads(
        (ROOT / "manifests" / "zenodo_release_21882300.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "released"
    assert manifest["record"]["concept_doi"] == "10.5281/zenodo.15612047"
    assert manifest["record"]["version_doi"] == "10.5281/zenodo.21882300"

    files = {entry["filename"]: entry for entry in manifest["files"]}
    old_name = "noise_guidance_best_R2_all_peptide_epoch_13.pth"
    canonical_name = "mic_candidate_scorer_all_peptide_non_pad_t1e-3_epoch13.pth"
    assert old_name not in files
    assert canonical_name in files

    scorer = files[canonical_name]
    assert scorer["protocol"]["profile"] == "fixed_epsilon_non_pad"
    assert scorer["protocol"]["t"] == pytest.approx(1e-3)
    assert scorer["protocol"]["exact_clean_t0"] is False
    assert scorer["sha256"] == (
        "c0d7c2be49ef179a25a19dcd9c54c592c282b6961e51aff60e95fabc13786802"
    )

    classification = files["apexoracle_fig1b_classification_reproduction_v1.tar.gz"]
    assert classification["scope"]["exact_folds"] == 15
    assert classification["scope"]["prediction_tables"] == 9
    assert manifest["verification"]["classification_fresh_download_sha256_matches"]
