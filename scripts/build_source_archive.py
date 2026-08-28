#!/usr/bin/env python3
"""Build a deterministic source archive from all locked submodules."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "manifests/modules.lock.yaml"
ARCHIVE_ROOT = "ApexOracle-source"


def run(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan() -> dict:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    modules = payload["modules"]
    if any(item["status"] != "ready" or not item["commit"] for item in modules):
        raise ValueError("All modules must be ready and commit-locked")
    root_commit = run("git", "rev-parse", "HEAD")
    for item in modules:
        actual = run("git", "-C", item["path"], "rev-parse", "HEAD")
        if actual != item["commit"]:
            raise ValueError(f"{item['path']} is {actual}, expected {item['commit']}")
    return {
        "schema_version": 1,
        "root_commit": root_commit,
        "archive_root": ARCHIVE_ROOT,
        "modules": [
            {key: item[key] for key in ("id", "path", "url", "commit")}
            for item in modules
        ],
    }


def add_git_archive(repo: Path, commit: str, destination: Path, prefix: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar") as handle:
        subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", f"--prefix={prefix}", commit, "-o", handle.name],
            check=True,
        )
        with tarfile.open(handle.name, "r") as archive:
            archive.extractall(destination, filter="data")


def build(output: Path) -> dict:
    plan = load_plan()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="apexoracle-source-") as temporary:
        staging = Path(temporary)
        add_git_archive(ROOT, plan["root_commit"], staging, f"{ARCHIVE_ROOT}/")
        for module in plan["modules"]:
            add_git_archive(
                ROOT / module["path"],
                module["commit"],
                staging,
                f"{ARCHIVE_ROOT}/{module['path']}/",
            )
        source_root = staging / ARCHIVE_ROOT
        manifest_path = source_root / "SOURCE_ARCHIVE_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        subprocess.run(
            [
                "tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0",
                "--numeric-owner", "-czf", str(output), "-C", str(staging), ARCHIVE_ROOT,
            ],
            check=True,
        )
    result = plan | {
        "archive": output.name,
        "size_bytes": output.stat().st_size,
        "sha256": sha256(output),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{result['sha256']}  {output.name}\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        result = load_plan()
    else:
        if args.output is None:
            parser.error("--output is required unless --plan-only is used")
        result = build(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
