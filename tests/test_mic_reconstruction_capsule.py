import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_script("mic_capsule_builder", "build_mic_reconstruction_capsule.py")
CHECKER = load_script("mic_capsule_checker", "recompute_mic_reconstruction_metrics.py")


def write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def test_metric_helpers_handle_ties() -> None:
    assert CHECKER.ranks([3.0, 1.0, 1.0]) == [3.0, 1.5, 1.5]
    metrics = CHECKER.calculate_metrics([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert metrics == pytest.approx({"r2": 1.0, "spearman": 1.0, "pearson": 1.0})


def test_public_release_manifest_freezes_mic_capsule() -> None:
    manifest = json.loads(
        (ROOT / "manifests/zenodo_release_21883545.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "released"
    assert manifest["record"]["concept_doi"] == "10.5281/zenodo.15612047"
    assert manifest["record"]["version_doi"] == "10.5281/zenodo.21883545"
    asset = next(
        item
        for item in manifest["files"]
        if item["filename"] == "apexoracle_fixed_mic_reconstruction_v1.tar.gz"
    )
    assert asset["size_bytes"] == 40177188
    assert asset["sha256"] == (
        "25e74abde1f01be57e83b22f6bd1633634284e74257d71f3c71864f7c4b9eebc"
    )
    assert asset["scope"]["member_prediction_tables"] == 21
    assert asset["scope"]["ensemble_rows"] == 86358
    assert asset["scope"]["exact_mic_values_included"] is False


def test_small_capsule_removes_exact_mic_and_absolute_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    core = tmp_path / "core"
    prediction_columns = [
        "row_key",
        "protocol",
        "group_index",
        "group_name",
        "ensemble",
        "route",
        "DBAASP_id",
        "molecule_identity",
        "strain_name",
        "MIC_um",
        "label_z",
        "train_seen_molecule",
        "prediction",
        "peptide_mean_baseline_prediction",
        "training_mean_baseline_prediction",
    ]
    base_rows = [
        [
            "g0:r:0",
            "strain",
            0,
            "fold 1",
            0,
            "text_only",
            "private-row-1",
            "hash-1",
            "strain-1",
            8.0,
            0.1,
            True,
            0.2,
            0.0,
            -0.1,
        ],
        [
            "g0:r:1",
            "strain",
            0,
            "fold 1",
            0,
            "text_only",
            "private-row-2",
            "hash-2",
            "strain-2",
            32.0,
            -0.2,
            False,
            -0.1,
            0.0,
            -0.1,
        ],
    ]
    for member in range(2):
        rows = [list(row) for row in base_rows]
        for row in rows:
            row[4] = member
        stem = f"strain_group_0_ensemble_{member}"
        write_csv(source / "predictions" / f"{stem}.csv", prediction_columns, rows)
        (source / "predictions" / f"{stem}.json").write_text(
            json.dumps(
                {
                    "task": stem,
                    "status": "completed",
                    "checkpoint": f"/private/path/{stem}.pth",
                    "checkpoint_format": "legacy_full",
                    "contract": {},
                    "archived_r2": 0.5,
                    "evaluation_mode": "deterministic_eval",
                    "membership_status": "deterministic_candidate_not_exact_2025",
                }
            ),
            encoding="utf-8",
        )
    ensemble_columns = [
        column for column in prediction_columns if column != "ensemble"
    ] + ["ensemble_members"]
    ensemble_rows = [row[:4] + row[5:] + [2] for row in base_rows]
    write_csv(
        source / "analysis/ensemble_predictions.csv",
        ensemble_columns,
        ensemble_rows,
    )
    write_csv(
        source / "analysis/metrics.csv",
        ["protocol", "group", "legacy_subset", "model"],
        [["strain", "0", "full", "apexoracle_ensemble"]],
    )
    write_csv(
        source / "analysis/cluster_bootstrap.csv",
        ["metric", "group", "legacy_subset", "value"],
        [["r2", "all_groups_pooled", "train_unseen", 0.1]],
    )
    result_summary = {
        "full_test_measurements": 2,
        "train_seen_measurements": 1,
        "train_unseen_measurements": 1,
        "full_r2": 0.1,
        "train_seen_r2": 0.2,
        "train_unseen_r2": 0.0,
        "train_unseen_spearman": 0.1,
        "train_unseen_pearson": 0.1,
    }
    (source / "task_manifest.json").write_text(
        json.dumps({"result_summary": result_summary}), encoding="utf-8"
    )
    split = core / "experiments/hierarchical_mic/strain/legacy_protocol_manifest.json"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({"folds": []}), encoding="utf-8")

    output = tmp_path / "capsule"
    result = BUILDER.build(
        source, core, output, "10.5281/zenodo.test", groups=1, members=2
    )
    assert result["ensemble_rows"] == 2
    assert Path(result["archive"]).is_file()
    member_text = (
        output / "data/member_predictions/strain_group_0_ensemble_0.csv"
    ).read_text(encoding="utf-8")
    assert "MIC_um" not in member_text
    assert "DBAASP_id" not in member_text
    assert "private-row" not in member_text
    registry = (output / "metadata/member_registry.json").read_text(encoding="utf-8")
    assert "/private/path" not in registry
    metrics_header = (
        (output / "data/metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    )
    assert metrics_header == "protocol,group,subset,model"
