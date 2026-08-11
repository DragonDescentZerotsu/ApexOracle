#!/usr/bin/env python3
"""Build the privacy-minimized high-confidence synergy replay capsule."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/recompute_synergy_metrics.py"
CAPSULE_ID = "apexoracle_synergy_replay_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def render_readme(version_doi: str) -> str:
    return f"""# ApexOracle synergy checkpoint replay

This capsule releases the privacy-minimized sample-level output of the complete
3-fold × 7-member synergy checkpoint replay. The fixed `PYTHONHASHSEED=0`
legacy-codepath split reproduces every archived fold AUROC/AUPRC after rounding
to four decimals. It is therefore a high-confidence historical candidate, but
is not presented as proven exact 2025 membership because the old runs did not
record the hash seed or sample-level predictions.

Version DOI: `{version_doi}`

`data/all_fold_predictions.csv` contains 2,371 token-filtered measurements,
seven member probabilities and their mean. Raw molecule identifiers,
structures and exact FICI values are excluded. Repeated pair/strain keys are
retained with a stable `measurement_index`; merging them would change the
historical metric definition.

Verify from the unpacked directory:

```bash
sha256sum -c SHA256SUMS
python recompute_synergy_metrics.py .
```
"""


def archive_directory(directory: Path, archive: Path) -> None:
    with archive.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for path in sorted(directory.rglob("*")):
                    if not path.is_file():
                        continue
                    info = tar.gettarinfo(
                        str(path),
                        arcname=(
                            Path(CAPSULE_ID) / path.relative_to(directory)
                        ).as_posix(),
                    )
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)


def build(
    source: Path, core: Path, output: Path, version_doi: str
) -> dict[str, object]:
    archive = Path(f"{output}.tar.gz")
    if output.exists() or archive.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {archive}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "data").mkdir(parents=True)
        (temporary / "metadata").mkdir(parents=True)
        shutil.copyfile(
            source / "all_fold_predictions.csv",
            temporary / "data/all_fold_predictions.csv",
        )
        shutil.copyfile(
            source / "summary.json", temporary / "metadata/replay_summary.json"
        )
        shutil.copyfile(
            core / "experiments/synergy/legacy_split_seed0.json",
            temporary / "metadata/split_candidate.json",
        )
        shutil.copyfile(CHECKER, temporary / CHECKER.name)
        (temporary / "README.md").write_text(
            render_readme(version_doi), encoding="utf-8"
        )
        release = {
            "schema_version": 1,
            "capsule_id": CAPSULE_ID,
            "version_doi": version_doi,
            "source_builder_revision": git_revision(),
            "core_replay_revision": "23e560e11a2ce06c19cc6eb6fe1dd0c0ee9b21f5",
            "historical_boundary": "high-confidence seed-0 candidate; not proven exact 2025 membership",
            "rows": 2371,
            "privacy_boundary": [
                "no exact FICI values",
                "no raw molecule identifiers or structures",
                "no embeddings, checkpoints, optimizer state, or absolute author paths",
            ],
        }
        (temporary / "metadata/release.json").write_text(
            json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files = []
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        (temporary / "MANIFEST.json").write_text(
            json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        checksum_paths = sorted(item for item in temporary.rglob("*") if item.is_file())
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--core", type=Path, default=ROOT / "modules/core")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version-doi", default="pending_until_reserved")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.source.resolve(),
                args.core.resolve(),
                args.output.resolve(),
                args.version_doi,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
