#!/usr/bin/env python3
"""Validate active ApexOracle gitlinks against the module lock manifest."""

from __future__ import annotations

import configparser
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "manifests" / "modules.lock.yaml"


def git(*args: str, cwd: Path = ROOT, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def load_gitmodules() -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(ROOT / ".gitmodules")
    result = {}
    for section in parser.sections():
        path = parser[section]["path"]
        result[path] = parser[section]["url"]
    return result


def main() -> None:
    manifest = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    modules = manifest["modules"]
    ready = {item["path"]: item for item in modules if item["status"] == "ready"}
    pending = {item["path"]: item for item in modules if item["status"] == "pending"}
    gitmodules = load_gitmodules()
    errors = []

    if set(gitmodules) != set(ready):
        errors.append(
            f".gitmodules paths {sorted(gitmodules)} != ready locks {sorted(ready)}"
        )
    for path, item in ready.items():
        expected = item["commit"]
        if not isinstance(expected, str) or len(expected) != 40:
            errors.append(f"ready module lacks full commit: {path}")
            continue
        if gitmodules.get(path) != item["url"]:
            errors.append(f"URL mismatch for {path}")
        module_path = ROOT / path
        if not module_path.is_dir():
            errors.append(f"module is not initialized: {path}")
            continue
        actual = git("rev-parse", "HEAD", cwd=module_path, check=False)
        if actual != expected:
            errors.append(f"commit mismatch for {path}: {actual} != {expected}")
    for path, item in pending.items():
        if item["commit"] is not None:
            errors.append(f"pending module has a commit: {path}")
        if path in gitmodules:
            errors.append(f"pending module appears in .gitmodules: {path}")

    summary = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "ready_modules": sorted(ready),
        "pending_modules": sorted(pending),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
