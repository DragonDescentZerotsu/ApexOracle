#!/usr/bin/env python3
"""Build the privacy-minimized fixed-split MIC reconstruction capsule."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/recompute_mic_reconstruction_metrics.py"
CAPSULE_ID = "apexoracle_fixed_mic_reconstruction_v1"
MEMBER_COLUMNS = [
    "row_key",
    "group_index",
    "group_name",
    "member",
    "route",
    "molecule_identity",
    "strain_id",
    "label_z",
    "low_mic_16um",
    "train_seen_exact_molecule",
    "prediction",
    "peptide_mean_baseline_prediction",
    "training_mean_baseline_prediction",
]
ENSEMBLE_COLUMNS = [column for column in MEMBER_COLUMNS if column != "member"] + [
    "ensemble_members"
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, columns: list[str], rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def normalize_prediction(
    row: dict[str, str], *, member: str | None
) -> dict[str, object]:
    output: dict[str, object] = {
        "row_key": row["row_key"],
        "group_index": int(row["group_index"]),
        "group_name": row["group_name"],
        "route": row["route"],
        "molecule_identity": row["molecule_identity"],
        "strain_id": row["strain_name"],
        "label_z": row["label_z"],
        "low_mic_16um": float(row["MIC_um"]) <= 16.0,
        "train_seen_exact_molecule": row["train_seen_molecule"],
        "prediction": row["prediction"],
        "peptide_mean_baseline_prediction": row["peptide_mean_baseline_prediction"],
        "training_mean_baseline_prediction": row["training_mean_baseline_prediction"],
    }
    if member is not None:
        output["member"] = int(member)
    else:
        output["ensemble_members"] = int(row["ensemble_members"])
    return output


def normalize_prediction_file(
    source: Path, destination: Path, *, member: str | None
) -> tuple[int, str]:
    with source.open(newline="", encoding="utf-8") as source_handle:
        reader = csv.DictReader(source_handle)
        required = {
            "row_key",
            "group_index",
            "group_name",
            "route",
            "molecule_identity",
            "strain_name",
            "MIC_um",
            "label_z",
            "train_seen_molecule",
            "prediction",
            "peptide_mean_baseline_prediction",
            "training_mean_baseline_prediction",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing columns {sorted(missing)}")
        columns = MEMBER_COLUMNS if member is not None else ENSEMBLE_COLUMNS
        rows = (normalize_prediction(row, member=member) for row in reader)
        count = write_csv(destination, columns, rows)
    return count, sha256_file(destination)


def normalize_tabular_summary(source: Path, destination: Path) -> tuple[int, str]:
    with source.open(newline="", encoding="utf-8") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None or len(reader.fieldnames) < 3:
            raise ValueError(f"{source}: invalid summary schema")
        source_subset_column = reader.fieldnames[2]
        columns = [
            "subset" if column == source_subset_column else column
            for column in reader.fieldnames
        ]

        def rows():
            for row in reader:
                normalized = dict(row)
                normalized["subset"] = normalized.pop(source_subset_column)
                yield normalized

        count = write_csv(destination, columns, rows())
    return count, sha256_file(destination)


def normalized_member_metadata(
    source: Path, prediction_name: str, prediction_rows: int, prediction_sha256: str
) -> dict[str, object]:
    data = json.loads(source.read_text(encoding="utf-8"))
    provenance = data.get("contract", {}).get("inference_checkpoint_provenance", {})
    checkpoint = Path(str(data["checkpoint"]))
    return {
        "task": data["task"],
        "status": data["status"],
        "checkpoint_filename": checkpoint.name,
        "checkpoint_format": data["checkpoint_format"],
        "checkpoint_sha256": provenance.get("source_checkpoint_sha256"),
        "checkpoint_size_bytes": provenance.get("source_checkpoint_size"),
        "checkpoint_hash_status": (
            "recorded" if provenance.get("source_checkpoint_sha256") else "not_recorded"
        ),
        "archived_r2": data["archived_r2"],
        "evaluation_mode": data["evaluation_mode"],
        "membership_status": data["membership_status"],
        "prediction_file": prediction_name,
        "prediction_rows": prediction_rows,
        "prediction_sha256": prediction_sha256,
    }


def render_readme(version_doi: str) -> str:
    return f"""# ApexOracle fixed-split MIC reconstruction

This capsule supports direct recomputation of the post-paper fixed strain-wise MIC
reconstruction. It is **not** the unrecovered membership used by the 2025 paper
checkpoints. The split is the frozen `PYTHONHASHSEED=0` deterministic legacy-codepath
candidate, and all 21 members were retrained on that fixed membership.

Version DOI: `{version_doi}`

## Contents

- `data/member_predictions/`: 21 normalized member prediction tables (3 groups × 7 members).
- `data/ensemble_predictions.csv`: 86,358 ensemble rows.
- `data/metrics.csv`: frozen full/train-seen/train-unseen metrics for four predictors.
- `data/cluster_bootstrap.csv`: frozen molecule-cluster bootstrap intervals.
- `metadata/fixed_split_manifest.json`: exact membership used by this reconstruction.
- `metadata/member_registry.json`: member prediction hashes and available checkpoint provenance.
- `metadata/release.json`: scientific scope, source hashes, and privacy boundary.
- `recompute_mic_reconstruction_metrics.py`: standard-library hash, ensemble, and metric checker.

The prediction tables contain normalized MIC labels, predictions, hashed molecule identity,
condition route, strain IDs, and a `MIC <= 16 µM` boolean. They do not contain molecule
structures, token sequences, exact MIC values, embedding tensors, checkpoints, optimizer
state, source-row identifiers, or private source tables.

Checkpoint binaries are not required to verify the released ensemble and metrics. Source
checkpoint hashes were not retained for every member; the registry reports missing hashes as
`not_recorded` rather than inventing provenance.

## Verify

From the unpacked capsule directory:

```bash
sha256sum -c SHA256SUMS
python recompute_mic_reconstruction_metrics.py .
```
"""


def archive_directory(directory: Path, archive: Path) -> None:
    with archive.open("xb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for path in sorted(directory.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = Path(CAPSULE_ID) / path.relative_to(directory)
                    info = tar.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)


def source_builder_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def build(
    source_root: Path,
    core_root: Path,
    output: Path,
    version_doi: str,
    *,
    groups: int = 3,
    members: int = 7,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    archive = Path(f"{output}.tar.gz")
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        member_registry = []
        for group in range(groups):
            for member in range(members):
                stem = f"strain_group_{group}_ensemble_{member}"
                source_csv = source_root / "predictions" / f"{stem}.csv"
                source_json = source_root / "predictions" / f"{stem}.json"
                if not source_csv.is_file() or not source_json.is_file():
                    raise FileNotFoundError(f"Missing prediction pair for {stem}")
                destination = temporary / "data/member_predictions" / f"{stem}.csv"
                rows, prediction_sha256 = normalize_prediction_file(
                    source_csv, destination, member=str(member)
                )
                member_registry.append(
                    normalized_member_metadata(
                        source_json,
                        destination.relative_to(temporary).as_posix(),
                        rows,
                        prediction_sha256,
                    )
                )

        ensemble_rows, ensemble_sha256 = normalize_prediction_file(
            source_root / "analysis/ensemble_predictions.csv",
            temporary / "data/ensemble_predictions.csv",
            member=None,
        )
        metrics_rows, metrics_sha256 = normalize_tabular_summary(
            source_root / "analysis/metrics.csv", temporary / "data/metrics.csv"
        )
        bootstrap_rows, bootstrap_sha256 = normalize_tabular_summary(
            source_root / "analysis/cluster_bootstrap.csv",
            temporary / "data/cluster_bootstrap.csv",
        )
        split_source = (
            core_root
            / "experiments/hierarchical_mic/strain/legacy_protocol_manifest.json"
        )
        (temporary / "metadata").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(split_source, temporary / "metadata/fixed_split_manifest.json")
        shutil.copyfile(CHECKER, temporary / CHECKER.name)
        (temporary / "metadata/member_registry.json").write_text(
            json.dumps(member_registry, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        task = json.loads(
            (source_root / "task_manifest.json").read_text(encoding="utf-8")
        )
        release = {
            "schema_version": 1,
            "capsule_id": CAPSULE_ID,
            "version_doi": version_doi,
            "source_builder_revision": source_builder_revision(),
            "status": "built",
            "scientific_scope": {
                "membership": "fixed PYTHONHASHSEED=0 deterministic legacy-codepath candidate",
                "historical_boundary": "post-paper reconstruction; not the unrecovered 2025 membership",
                "groups": groups,
                "members_per_group": members,
                "ensemble_rows": ensemble_rows,
                "result_summary": {
                    key: task["result_summary"][key]
                    for key in (
                        "full_test_measurements",
                        "train_seen_measurements",
                        "train_unseen_measurements",
                        "full_r2",
                        "train_seen_r2",
                        "train_unseen_r2",
                        "train_unseen_spearman",
                        "train_unseen_pearson",
                    )
                },
            },
            "source_hashes": {
                "task_manifest": sha256_file(source_root / "task_manifest.json"),
                "fixed_split_manifest": sha256_file(split_source),
                "source_ensemble_predictions": sha256_file(
                    source_root / "analysis/ensemble_predictions.csv"
                ),
                "source_metrics": sha256_file(source_root / "analysis/metrics.csv"),
                "source_cluster_bootstrap": sha256_file(
                    source_root / "analysis/cluster_bootstrap.csv"
                ),
            },
            "normalized_outputs": {
                "ensemble_predictions": {
                    "rows": ensemble_rows,
                    "sha256": ensemble_sha256,
                },
                "metrics": {"rows": metrics_rows, "sha256": metrics_sha256},
                "cluster_bootstrap": {
                    "rows": bootstrap_rows,
                    "sha256": bootstrap_sha256,
                },
            },
            "privacy_boundary": [
                "no molecule structures or token sequences",
                "no exact MIC values or source-row identifiers",
                "no embedding tensors, checkpoints, optimizer state, or private source tables",
                "normalized labels and hashed molecule identities are included for metric recomputation",
            ],
        }
        (temporary / "metadata/release.json").write_text(
            json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            render_readme(version_doi), encoding="utf-8"
        )

        manifest_files = []
        for path in sorted(path for path in temporary.rglob("*") if path.is_file()):
            manifest_files.append(
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        (temporary / "MANIFEST.json").write_text(
            json.dumps(
                {"schema_version": 1, "files": manifest_files},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        checksum_paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.relative_to(temporary).as_posix()}"
                for path in checksum_paths
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
        archive_directory(output, archive)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "status": "built",
        "directory": str(output),
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "version_doi": version_doi,
        "groups": groups,
        "members_per_group": members,
        "ensemble_rows": ensemble_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, default=ROOT / "modules/core")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version-doi", default="pending_until_reserved")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.source_root.resolve(),
                args.core_root.resolve(),
                args.output.resolve(),
                args.version_doi,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
