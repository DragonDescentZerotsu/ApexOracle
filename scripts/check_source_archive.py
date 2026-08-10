#!/usr/bin/env python3
"""Validate a built ApexOracle source-only archive without Git metadata."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import PurePosixPath


REQUIRED = {
    "ApexOracle-source/README.md",
    "ApexOracle-source/SOURCE_ARCHIVE_MANIFEST.json",
    "ApexOracle-source/modules/core/pyproject.toml",
    "ApexOracle-source/modules/dlm_pretrain/pyproject.toml",
    "ApexOracle-source/modules/mdlm/pyproject.toml",
    "ApexOracle-source/modules/evo2/pyproject.toml",
    "ApexOracle-source/modules/generation/main.py",
}
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".npz", ".whl", ".zip"
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()
    errors: list[str] = []
    with tarfile.open(args.archive, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        names = {member.name for member in members}
        missing = sorted(REQUIRED - names)
        if missing:
            errors.append(f"missing required files: {missing}")
        forbidden = sorted(
            name for name in names
            if ".git" in PurePosixPath(name).parts
            or PurePosixPath(name).suffix.lower() in FORBIDDEN_SUFFIXES
        )
        if forbidden:
            errors.append(f"forbidden files: {forbidden[:20]}")
        manifest_member = archive.getmember(
            "ApexOracle-source/SOURCE_ARCHIVE_MANIFEST.json"
        )
        manifest = json.load(archive.extractfile(manifest_member))
        if len(manifest.get("modules", [])) != 5:
            errors.append("source manifest does not contain five modules")
    result = {
        "schema_version": 1,
        "status": "failed" if errors else "passed",
        "file_count": len(members),
        "root_commit": manifest.get("root_commit"),
        "module_commits": {
            item["id"]: item["commit"] for item in manifest.get("modules", [])
        },
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
