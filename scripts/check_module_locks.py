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


def indexed_gitlink(path: str) -> str | None:
    """Return the commit recorded by the super-repository index."""
    line = git("ls-files", "--stage", "--", path, check=False)
    if not line:
        return None
    metadata, recorded_path = line.split("\t", 1)
    mode, commit, stage = metadata.split()
    if mode != "160000" or stage != "0" or recorded_path != path:
        return None
    return commit


def initialized_checkout_head(path: str) -> str | None:
    module_path = ROOT / path
    if not module_path.is_dir():
        return None
    top_level = git("rev-parse", "--show-toplevel", cwd=module_path, check=False)
    if not top_level or Path(top_level).resolve() != module_path.resolve():
        return None
    return git("rev-parse", "HEAD", cwd=module_path, check=False) or None


def main() -> None:
    manifest = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    modules = manifest["modules"]
    ready = {item["path"]: item for item in modules if item["status"] == "ready"}
    pending = {item["path"]: item for item in modules if item["status"] == "pending"}
    gitmodules = load_gitmodules()
    errors = []
    uninitialized = []

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
        gitlink = indexed_gitlink(path)
        if gitlink != expected:
            errors.append(f"gitlink mismatch for {path}: {gitlink} != {expected}")
        checkout_head = initialized_checkout_head(path)
        if checkout_head is None:
            uninitialized.append(path)
        elif checkout_head != expected:
            errors.append(
                f"checkout mismatch for {path}: {checkout_head} != {expected}"
            )
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
        "uninitialized_modules": sorted(uninitialized),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
