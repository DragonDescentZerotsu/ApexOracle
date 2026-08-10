#!/usr/bin/env python3
"""Validate that the active super-repo is compact and legacy-recoverable."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMMIT = "2f29dee9cf6b7750425414f66c1a2d67998cb87f"
LEGACY_BRANCH = "legacy-monorepo"
LEGACY_TAG = "legacy-monorepo-snapshot-2026-08-10"
MAX_ROOT_FILE_BYTES = 5 * 1024 * 1024
FORBIDDEN_ACTIVE_ROOTS = (
    "ApexOracle",
    "DLM_pretrain",
    "PepLink",
    "discrete-diffusion-guidance",
    "mdlm",
    "assets",
)
FORBIDDEN_SUFFIXES = (".ckpt", ".pth", ".pt", ".bin", ".xlsx", ".png", ".pdf")
REQUIRED = (
    ".gitmodules",
    "AGENTS.md",
    "CITATION.cff",
    "LICENSE",
    "NOTICE",
    "README.md",
    "docs/LEGACY_MONOREPO.md",
    "docs/RELEASE_STATUS.md",
    "manifests/modules.lock.yaml",
    "scripts/check_module_locks.py",
)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def active_files() -> list[Path]:
    result = git("ls-files", "-co", "--exclude-standard", "-z")
    return sorted(
        path
        for raw in result.stdout.split("\0")
        if raw and (path := ROOT / raw).is_file()
    )


def main() -> None:
    errors = []
    files = active_files()
    relative = {path.relative_to(ROOT).as_posix() for path in files}
    for required in REQUIRED:
        if required not in relative:
            errors.append(f"missing required path: {required}")
    for root in FORBIDDEN_ACTIVE_ROOTS:
        if (ROOT / root).exists():
            errors.append(f"legacy root remains active: {root}")
    forbidden_files = sorted(
        path for path in relative if Path(path).suffix.lower() in FORBIDDEN_SUFFIXES
    )
    if forbidden_files:
        errors.append(f"binary/data files remain active: {forbidden_files}")
    oversized = sorted(
        (path.relative_to(ROOT).as_posix(), path.stat().st_size)
        for path in files
        if not path.as_posix().startswith((ROOT / "modules").as_posix())
        and path.stat().st_size > MAX_ROOT_FILE_BYTES
    )
    if oversized:
        errors.append(f"root files exceed {MAX_ROOT_FILE_BYTES} bytes: {oversized}")

    branch = git("rev-parse", LEGACY_BRANCH)
    if branch.returncode:
        branch = git("rev-parse", f"origin/{LEGACY_BRANCH}")
    tag_type = git("cat-file", "-t", LEGACY_TAG)
    tag_commit = git("rev-parse", f"{LEGACY_TAG}^{{commit}}")
    if branch.returncode or branch.stdout.strip() != LEGACY_COMMIT:
        errors.append("legacy recovery branch is missing or moved")
    if tag_type.returncode or tag_type.stdout.strip() != "tag":
        errors.append("legacy recovery tag is missing or not annotated")
    if tag_commit.returncode or tag_commit.stdout.strip() != LEGACY_COMMIT:
        errors.append("legacy recovery tag does not resolve to the frozen commit")

    summary = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "active_file_count": len(files),
        "legacy_commit": LEGACY_COMMIT,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
