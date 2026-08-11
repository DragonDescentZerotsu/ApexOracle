#!/usr/bin/env python3
"""Build the public, source-partitioned model-ready data capsule."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "manifests/model_ready_capsule_sources.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file(path: Path, *, size_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != size_bytes:
        raise ValueError(f"{path}: size changed")
    observed = sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"{path}: SHA-256 changed: {observed}")


def read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"{path}: unexpected columns {reader.fieldnames}")
        return list(reader)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def archive_directory(directory: Path, archive: Path, archive_root: str) -> None:
    with archive.open("xb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for path in sorted(directory.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = Path(archive_root) / path.relative_to(directory)
                    info = tar.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)


def render_readme(manifest: dict[str, object]) -> str:
    zenodo = manifest["zenodo"]
    return f"""# ApexOracle public model-ready tables

This capsule is version `{zenodo['new_version_doi']}` of the existing
ApexOracle Zenodo series (`{zenodo['concept_doi']}`); it is not a separate
Zenodo project.

## Contents

- `tables/mic_dbaasp_model_ready.csv`: 105,237 DBAASP-derived MIC rows.
  The 15,718 private in-house rows in the authors' combined workstation table
  are explicitly excluded.
- `tables/small_molecule_classification_model_ready.csv`: the frozen 49,330-row
  paper classification table.
- `tables/synergy_model_ready.csv`: the frozen 4,285-row paper synergy source
  table. The public result capsule separately records the 2,732 mapped rows and
  exact/high-confidence split boundary used for replay.
- `manifests/paper_strain_mapping.json`: the compact mapping from source strain
  labels to runtime condition keys.
- `manifests/paper_genome_list.csv`: the 563 paper genome identities and hashes.

`manifests/capsule.json` records source hashes, partition rules, row counts, and
output hashes. `SHA256SUMS` verifies every included file.

## License and provenance boundary

The capsule compilation and documentation are CC BY 4.0. Database-derived
records retain their original source terms; CC BY 4.0 does not relicense the
underlying DBAASP records. Users should cite and check the current terms of the
source databases. No private in-house assay rows, embeddings, checkpoints,
optimizer state, or author-machine paths are included.

## Verify

```bash
sha256sum -c SHA256SUMS
```
"""


def build(source_root: Path, core_root: Path, output: Path) -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    archive = Path(f"{output}.tar.gz")
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {archive}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        released = []
        for record in manifest["tables"]:
            source = source_root / record["source_relative_path"]
            validate_file(
                source,
                size_bytes=int(record["source_size_bytes"]),
                expected_sha256=record["source_sha256"],
            )
            rows = read_csv(source, record["columns"])
            if len(rows) != int(record["source_rows"]):
                raise ValueError(f"{source}: row count changed")
            if record["id"] == "mic_dbaasp_public_partition":
                rows = [row for row in rows if row["DBAASP_id"].isdigit()]
            if len(rows) != int(record["released_rows"]):
                raise ValueError(f"{source}: released row count changed")
            destination = temporary / record["output_relative_path"]
            write_csv(destination, record["columns"], rows)
            released.append(
                {
                    **record,
                    "output_size_bytes": destination.stat().st_size,
                    "output_sha256": sha256(destination),
                }
            )

        compact_assets = []
        for record in manifest["compact_release_assets"]:
            source = core_root / record["source_relative_path"]
            validate_file(
                source,
                size_bytes=int(record["size_bytes"]),
                expected_sha256=record["sha256"],
            )
            destination = temporary / record["output_relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            compact_assets.append(record)

        capsule = {
            "schema_version": 1,
            "capsule_id": manifest["capsule_id"],
            "zenodo": manifest["zenodo"],
            "tables": released,
            "compact_release_assets": compact_assets,
            "privacy": {
                "private_inhouse_mic_rows_excluded": 15718,
                "private_assay_values_included": False,
            },
        }
        manifests = temporary / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        (manifests / "capsule.json").write_text(
            json.dumps(capsule, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(render_readme(manifest), encoding="utf-8")
        paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256(path)}  {path.relative_to(temporary).as_posix()}"
                for path in paths
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
        archive_directory(output, archive, manifest["capsule_id"])
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
        "version_doi": manifest["zenodo"]["new_version_doi"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.source_root.resolve(),
                args.core_root.resolve(),
                args.output.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
