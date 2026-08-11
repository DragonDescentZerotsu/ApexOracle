import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_model_ready_capsule.py"
SPEC = importlib.util.spec_from_file_location("model_ready_capsule", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_manifest_uses_existing_zenodo_series_and_excludes_private_rows() -> None:
    manifest = json.loads(
        (ROOT / "manifests/model_ready_capsule_sources.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["zenodo"]["concept_doi"] == "10.5281/zenodo.15612047"
    assert manifest["zenodo"]["previous_version_doi"] == ("10.5281/zenodo.21883954")
    assert manifest["zenodo"]["new_version_doi"] == "10.5281/zenodo.21891064"
    mic = manifest["tables"][0]
    assert mic["source_rows"] == 120955
    assert mic["released_rows"] == 105237
    assert mic["excluded_private_rows"] == 15718
    assert mic["released_rows"] + mic["excluded_private_rows"] == mic["source_rows"]
    assert all(
        not Path(record["source_relative_path"]).is_absolute()
        for record in manifest["tables"]
    )


def test_public_release_manifests_close_model_ready_and_result_registry() -> None:
    assets = json.loads(
        (ROOT / "manifests/data_assets.yaml").read_text(encoding="utf-8")
    )
    released = {record["id"]: record for record in assets["assets"]}
    model_ready = released["apexoracle_zenodo_model_ready_public_tables"]
    assert model_ready["record_id"] == 21891064
    assert model_ready["sha256"] == (
        "ae0c76febd4e0b4d43fd68c8bf3ddfa27fc2251011f88c5f693d9aa27be95901"
    )
    assert "model-ready" not in " ".join(assets["pending"])

    registry = json.loads(
        (ROOT / "manifests/paper_result_registry.json").read_text(encoding="utf-8")
    )
    assert registry["status"] == "complete_for_released_paper_result_capsules"
    assert [record["id"] for record in registry["results"]] == [
        "fig1b_classification",
        "fixed_strainwise_mic_reconstruction",
        "synergy_replay",
    ]
    assert registry["results"][1]["checkpoint_hash_status"] == {
        "recorded": 7,
        "not_recorded": 14,
    }
    assert registry["results"][2]["checkpoint_hash_status"] == {
        "recorded": 22,
        "not_recorded": 0,
    }

    zenodo = json.loads(
        (ROOT / "manifests/zenodo_release_21891064.json").read_text(
            encoding="utf-8"
        )
    )
    assert zenodo["record"]["version_doi"] == "10.5281/zenodo.21891064"
    assert zenodo["verification"]["private_inhouse_partition_excluded"] is True


def test_builder_filters_numeric_mic_ids(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    core_root = tmp_path / "core"
    source = source_root / "mic.csv"
    source.parent.mkdir(parents=True)
    source.write_text(
        "DBAASP_id,strain_name,SMILES,MIC\n"
        "12,strain-a,C,1.0\n"
        "private_0,strain-b,N,2.0\n",
        encoding="utf-8",
    )
    compact = core_root / "mapping.json"
    compact.parent.mkdir(parents=True)
    compact.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "capsule_id": "test_capsule",
        "zenodo": {
            "concept_doi": "10.5281/zenodo.test",
            "new_version_doi": "10.5281/zenodo.test2",
        },
        "tables": [
            {
                "id": "mic_dbaasp_public_partition",
                "source_relative_path": "mic.csv",
                "source_size_bytes": source.stat().st_size,
                "source_sha256": BUILDER.sha256(source),
                "source_rows": 2,
                "output_relative_path": "tables/mic.csv",
                "partition_rule": "numeric",
                "released_rows": 1,
                "excluded_private_rows": 1,
                "columns": ["DBAASP_id", "strain_name", "SMILES", "MIC"],
            }
        ],
        "compact_release_assets": [
            {
                "source_relative_path": "mapping.json",
                "output_relative_path": "manifests/mapping.json",
                "size_bytes": compact.stat().st_size,
                "sha256": BUILDER.sha256(compact),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(BUILDER, "MANIFEST_PATH", manifest_path)
    output = tmp_path / "capsule"
    result = BUILDER.build(source_root, core_root, output)
    assert result["status"] == "built"
    rows = BUILDER.read_csv(
        output / "tables/mic.csv", ["DBAASP_id", "strain_name", "SMILES", "MIC"]
    )
    assert [row["DBAASP_id"] for row in rows] == ["12"]
    capsule = json.loads((output / "manifests/capsule.json").read_text())
    assert capsule["privacy"]["private_assay_values_included"] is False
