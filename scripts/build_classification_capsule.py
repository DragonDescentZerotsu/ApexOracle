#!/usr/bin/env python3
"""Build the external Fig. 1b classification reproduction capsule."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "manifests" / "classification_capsule_sources.json"
RECOMPUTE_SCRIPT = ROOT / "scripts" / "recompute_classification_metrics.py"
PAPER_METRICS = (
    ROOT
    / "modules"
    / "core"
    / "experiments"
    / "fig1b_antibiotic_classification"
    / "final_10member_dual_metric.csv"
)
PAPER_METRICS_SHA256 = (
    "c4c189f7b327af0cb5762499efcae660474120586f6882a71ad1604c9383a8eb"
)

sys.path.insert(0, str(ROOT / "scripts"))
from recompute_classification_metrics import (  # noqa: E402
    METHODS,
    compute_capsule_metrics,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return reader.fieldnames, list(reader)


def validate_source(
    source_root: Path, record: dict[str, object]
) -> tuple[Path, list[dict[str, str]]]:
    path = source_root / str(record["source_relative_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"{path}: size changed")
    actual_sha256 = sha256(path)
    if actual_sha256 != record["sha256"]:
        raise ValueError(f"{path}: SHA-256 changed: {actual_sha256}")
    columns, rows = read_rows(path)
    required = {"molecule_id", "label", "prediction", "group", "fold"}
    if not required.issubset(columns):
        raise ValueError(f"{path}: missing columns {sorted(required - set(columns))}")
    if len(rows) != int(record["rows"]):
        raise ValueError(f"{path}: row count changed")
    if len({row["molecule_id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate molecule IDs")
    return path, rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalized_prediction(
    row: dict[str, str], group: int, split: dict[str, dict[str, str]]
) -> dict[str, object]:
    molecule_id = row["molecule_id"]
    if molecule_id not in split:
        raise ValueError(f"group {group}: {molecule_id} is absent from fixed split")
    split_row = split[molecule_id]
    if int(float(row["label"])) != int(float(split_row["label"])):
        raise ValueError(f"group {group}: label mismatch for {molecule_id}")
    if row["fold"] and int(row["fold"]) != int(split_row["fold"]):
        raise ValueError(f"group {group}: fold mismatch for {molecule_id}")
    if int(row["group"]) != group:
        raise ValueError(f"group {group}: group column mismatch for {molecule_id}")
    return {
        "molecule_id": molecule_id,
        "label": int(float(row["label"])),
        "prediction": row["prediction"],
        "group": group,
        "fold": int(split_row["fold"]),
    }


def validate_metrics(
    computed: dict[str, object], source_manifest: dict[str, object]
) -> None:
    tolerance = 1e-12
    for group in source_manifest["groups"]:
        group_id = str(group["group"])
        actual_group = computed["groups"][group_id]
        if actual_group["common_comparison_rows"] != group["common_comparison_rows"]:
            raise ValueError(f"group {group_id}: common comparison size changed")
        for method, expected in group["expected_common_metrics"].items():
            actual = actual_group["methods"][method]
            checks = {
                "pooled_auprc": actual["common_comparison"]["auprc"],
                "pooled_auroc": actual["common_comparison"]["auroc"],
                "fold_mean_auprc": actual["fold_mean_on_common_comparison"][
                    "auprc"
                ],
                "fold_mean_auroc": actual["fold_mean_on_common_comparison"][
                    "auroc"
                ],
            }
            for metric, value in checks.items():
                if abs(value - expected[metric]) > tolerance:
                    raise ValueError(
                        f"group {group_id} {method} {metric}: "
                        f"expected {expected[metric]}, got {value}"
                    )


def render_readme(source_manifest: dict[str, object], version_doi: str) -> str:
    zenodo = source_manifest["zenodo"]
    return f"""# ApexOracle Fig. 1b classification reproduction capsule

This capsule adds frozen Fig. 1b classification folds and sample-level
predictions to a new version of the existing ApexOracle Zenodo dataset. It is
not a second Zenodo project.

- Stable concept DOI: `{zenodo['concept_doi']}`
- Previously published version DOI: `{zenodo['existing_version_doi']}`
- This version DOI: `{version_doi}`
- Data license: CC BY 4.0
- Included metric script: MIT (same license as the ApexOracle source repository)

## Contents

- `splits/group_*.csv`: exact eligible molecule IDs, labels, and deterministic
  five-fold membership (`random_state=42`).
- `predictions/apexoracle_strict_zero_shot/`: deterministic 10-member
  checkpoint inference. The fixed fold column is attached for paired reporting;
  zero-shot training itself does not use target folds.
- `predictions/apexoracle_finetuned_10member/`: 10-member out-of-fold
  predictions for every eligible molecule.
- `predictions/chemprop_10member/`: 10-member out-of-fold baseline predictions.
  Two structures that Chemprop/RDKit could not process are absent; common-cohort
  metrics intersect molecule IDs across all three methods.
- `manifests/recomputed_metrics.json`: AUPRC/AUROC independently recomputed
  from the normalized released files.
- `manifests/paper_reported_fold_metrics.csv`: the frozen plotting/reporting
  table used for the revised Fig. 1b.
- `recompute_classification_metrics.py`: standard-library-only metric checker.

The normalized prediction files deliberately omit SMILES, embeddings,
checkpoints, logs, and optimizer state. This keeps the archive compact and
separates result recomputation from redistribution of model-ready source data.

## Verify

From the unpacked capsule directory:

```bash
sha256sum -c SHA256SUMS
python recompute_classification_metrics.py \
  --capsule-root . \
  --output /tmp/apexoracle-classification-metrics.json
diff -u manifests/recomputed_metrics.json \
  /tmp/apexoracle-classification-metrics.json
```
"""


def archive_directory(directory: Path, archive: Path, archive_root: str) -> None:
    with archive.open("xb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for path in sorted(directory.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = Path(archive_root) / path.relative_to(directory)
                    info = tar.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)


def build(source_root: Path, output: Path, version_doi: str) -> dict[str, object]:
    source_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    archive = Path(f"{output}.tar.gz")
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {archive}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        source_assets = []
        for group in source_manifest["groups"]:
            group_id = int(group["group"])
            validated: dict[str, tuple[Path, list[dict[str, str]]]] = {}
            for method in METHODS:
                record = group["sources"][method]
                validated[method] = validate_source(source_root, record)
                source_assets.append(
                    {
                        "group": group_id,
                        "method": method,
                        **record,
                    }
                )

            fine_rows = validated["apexoracle_finetuned_10member"][1]
            split = {row["molecule_id"]: row for row in fine_rows}
            if len(split) != int(group["eligible_rows"]):
                raise ValueError(f"group {group_id}: eligible split size changed")
            split_rows = [
                {
                    "molecule_id": row["molecule_id"],
                    "label": int(float(row["label"])),
                    "group": group_id,
                    "fold": int(row["fold"]),
                }
                for row in fine_rows
            ]
            split_rows.sort(key=lambda row: (row["fold"], row["molecule_id"]))
            write_csv(
                temporary / "splits" / f"group_{group_id}.csv",
                ["molecule_id", "label", "group", "fold"],
                split_rows,
            )

            for method in METHODS:
                normalized = [
                    normalized_prediction(row, group_id, split)
                    for row in validated[method][1]
                ]
                normalized.sort(key=lambda row: (row["fold"], row["molecule_id"]))
                write_csv(
                    temporary / "predictions" / method / f"group_{group_id}.csv",
                    ["molecule_id", "label", "prediction", "group", "fold"],
                    normalized,
                )

        shutil.copyfile(RECOMPUTE_SCRIPT, temporary / RECOMPUTE_SCRIPT.name)
        if sha256(PAPER_METRICS) != PAPER_METRICS_SHA256:
            raise ValueError("Core paper-reported metrics CSV changed")
        metrics_destination = temporary / "manifests" / "paper_reported_fold_metrics.csv"
        metrics_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PAPER_METRICS, metrics_destination)

        computed = compute_capsule_metrics(temporary)
        validate_metrics(computed, source_manifest)
        (temporary / "manifests" / "recomputed_metrics.json").write_text(
            json.dumps(computed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        capsule_manifest = {
            "schema_version": 1,
            "capsule_id": source_manifest["capsule_id"],
            "zenodo": {
                **source_manifest["zenodo"],
                "new_version_doi": version_doi,
            },
            "source_assets": source_assets,
            "normalization": source_manifest["normalization"],
            "scientific_scope": {
                "classification_folds": "exact",
                "strict_zero_shot_predictions": "deterministic checkpoint inference",
                "finetuned_predictions": "10-member out-of-fold ensemble",
                "chemprop_predictions": "10-member out-of-fold ensemble",
                "model_ready_source_table": "not included",
                "model_checkpoints": "not included",
            },
        }
        (temporary / "manifests" / "capsule.json").write_text(
            json.dumps(capsule_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            render_readme(source_manifest, version_doi), encoding="utf-8"
        )

        checksum_paths = sorted(
            path for path in temporary.rglob("*") if path.is_file()
        )
        checksum_lines = [
            f"{sha256(path)}  {path.relative_to(temporary).as_posix()}"
            for path in checksum_paths
        ]
        (temporary / "SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
        archive_directory(output, archive, source_manifest["capsule_id"])
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "status": "built",
        "directory": str(output),
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "zenodo_concept_doi": source_manifest["zenodo"]["concept_doi"],
        "zenodo_new_version_doi": version_doi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--new-version-doi",
        default="pending_until_published",
        help="Reserved DOI for the new version, when available",
    )
    args = parser.parse_args()
    result = build(
        args.source_root.resolve(), args.output.resolve(), args.new_version_doi
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
