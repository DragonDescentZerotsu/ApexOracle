#!/usr/bin/env python3
"""Fail on unreviewed growth, generated artifacts, or large exact duplicates."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "manifests" / "repository_size_policy.json"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=repo, check=False, capture_output=True
    )


def active_files(repo: Path) -> tuple[list[Path], set[str]]:
    result = git(repo, "ls-files", "-co", "--exclude-standard", "-z")
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    tracked_result = git(repo, "ls-files", "-z")
    if tracked_result.returncode:
        raise RuntimeError(tracked_result.stderr.decode("utf-8", errors="replace"))
    tracked = {
        item.decode("utf-8", errors="surrogateescape")
        for item in tracked_result.stdout.split(b"\0")
        if item
    }
    paths = []
    for item in result.stdout.split(b"\0"):
        if not item:
            continue
        relative = item.decode("utf-8", errors="surrogateescape")
        path = repo / relative
        if path.is_file() or path.is_symlink():
            paths.append(path)
    return sorted(paths), tracked


def file_bytes(path: Path) -> bytes:
    if path.is_symlink():
        return os.readlink(path).encode("utf-8")
    return path.read_bytes()


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    default_max = int(policy["default_max_file_bytes"])
    duplicate_min = int(policy["duplicate_scan_min_bytes"])
    forbidden_suffixes = set(policy["forbidden_suffixes"])
    forbidden_parts = set(policy["forbidden_path_parts"])
    allowlist = policy["large_file_allowlist"]

    errors: list[str] = []
    repository_summaries: dict[str, dict[str, object]] = {}
    hashes: dict[str, list[dict[str, object]]] = defaultdict(list)

    for repo_id, repo_policy in policy["repositories"].items():
        repo = (ROOT / repo_policy["path"]).resolve()
        paths, tracked = active_files(repo)
        total_bytes = 0
        largest: list[tuple[int, str]] = []
        untracked: list[str] = []

        for path in paths:
            relative = path.relative_to(repo).as_posix()
            scoped = f"{repo_id}/{relative}"
            payload = file_bytes(path)
            size = len(payload)
            total_bytes += size
            largest.append((size, relative))
            if relative not in tracked:
                untracked.append(relative)

            parts = set(Path(relative).parts)
            if parts & forbidden_parts:
                errors.append(f"{scoped}: generated/cache path is forbidden")
            if path.suffix.lower() in forbidden_suffixes:
                errors.append(f"{scoped}: model/cache suffix is forbidden")

            allowed = allowlist.get(scoped)
            if size > default_max and allowed is None:
                errors.append(
                    f"{scoped}: {size} bytes exceeds default {default_max} without allowlist"
                )
            if allowed is not None and size > int(allowed["max_bytes"]):
                errors.append(
                    f"{scoped}: {size} bytes exceeds allowlisted maximum {allowed['max_bytes']}"
                )

            if size >= duplicate_min:
                digest = hashlib.sha256(payload).hexdigest()
                hashes[digest].append(
                    {"repository": repo_id, "path": relative, "size_bytes": size}
                )

        count = len(paths)
        if count > int(repo_policy["max_file_count"]):
            errors.append(
                f"{repo_id}: {count} files exceeds {repo_policy['max_file_count']}"
            )
        if total_bytes > int(repo_policy["max_total_bytes"]):
            errors.append(
                f"{repo_id}: {total_bytes} bytes exceeds {repo_policy['max_total_bytes']}"
            )
        repository_summaries[repo_id] = {
            "file_count": count,
            "tracked_bytes": total_bytes,
            "untracked_nonignored": sorted(untracked),
            "largest_files": [
                {"path": path, "size_bytes": size}
                for size, path in sorted(largest, reverse=True)[:5]
            ],
            "limits": {
                "max_file_count": int(repo_policy["max_file_count"]),
                "max_total_bytes": int(repo_policy["max_total_bytes"]),
            },
        }

    duplicate_groups = [
        {"sha256": digest, "files": files}
        for digest, files in sorted(hashes.items())
        if len(files) > 1
    ]
    for group in duplicate_groups:
        locations = [
            f"{item['repository']}/{item['path']}" for item in group["files"]
        ]
        errors.append(
            f"exact duplicate >= {duplicate_min} bytes: {', '.join(locations)}"
        )

    summary = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "policy": str(POLICY_PATH.relative_to(ROOT)),
        "repositories": repository_summaries,
        "duplicate_groups": duplicate_groups,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
