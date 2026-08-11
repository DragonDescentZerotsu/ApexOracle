#!/usr/bin/env python3
"""Audit repository growth, layout drift, generated artifacts, and duplicates."""

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
    intra_repo_duplicate_min = int(policy["intra_repository_duplicate_min_bytes"])
    source_review_min_lines = int(policy["source_review_min_lines"])
    soft_limit_fraction = float(policy["soft_limit_fraction"])
    forbidden_suffixes = set(policy["forbidden_suffixes"])
    forbidden_parts = set(policy["forbidden_path_parts"])
    allowlist = policy["large_file_allowlist"]

    errors: list[str] = []
    soft_limit_alerts: list[str] = []
    repository_summaries: dict[str, dict[str, object]] = {}
    hashes: dict[str, list[dict[str, object]]] = defaultdict(list)

    for repo_id, repo_policy in policy["repositories"].items():
        repo = (ROOT / repo_policy["path"]).resolve()
        paths, tracked = active_files(repo)
        total_bytes = 0
        largest: list[tuple[int, str]] = []
        large_source_files: list[tuple[int, str]] = []
        top_level_counts: dict[str, int] = defaultdict(int)
        top_level_bytes: dict[str, int] = defaultdict(int)
        untracked: list[str] = []
        allowed_top_level_directories = set(
            repo_policy["allowed_top_level_directories"]
        )

        for path in paths:
            relative = path.relative_to(repo).as_posix()
            scoped = f"{repo_id}/{relative}"
            payload = file_bytes(path)
            size = len(payload)
            total_bytes += size
            largest.append((size, relative))
            relative_parts = Path(relative).parts
            top_level = relative_parts[0]
            top_level_counts[top_level] += 1
            top_level_bytes[top_level] += size
            if relative not in tracked:
                untracked.append(relative)

            if (
                len(relative_parts) > 1
                and top_level not in allowed_top_level_directories
            ):
                errors.append(
                    f"{scoped}: unexpected top-level directory {top_level!r}"
                )

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

            if path.suffix.lower() in {".py", ".sh"}:
                line_count = payload.count(b"\n")
                if payload and not payload.endswith(b"\n"):
                    line_count += 1
                if line_count >= source_review_min_lines:
                    large_source_files.append((line_count, relative))

            if size >= intra_repo_duplicate_min:
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
        file_fraction = count / int(repo_policy["max_file_count"])
        byte_fraction = total_bytes / int(repo_policy["max_total_bytes"])
        if file_fraction >= soft_limit_fraction:
            soft_limit_alerts.append(
                f"{repo_id}: file-count utilization is {file_fraction:.1%}"
            )
        if byte_fraction >= soft_limit_fraction:
            soft_limit_alerts.append(
                f"{repo_id}: byte utilization is {byte_fraction:.1%}"
            )
        repository_summaries[repo_id] = {
            "file_count": count,
            "tracked_bytes": total_bytes,
            "untracked_nonignored": sorted(untracked),
            "largest_files": [
                {"path": path, "size_bytes": size}
                for size, path in sorted(largest, reverse=True)[:5]
            ],
            "large_source_files": [
                {"path": path, "line_count": line_count}
                for line_count, path in sorted(large_source_files, reverse=True)
            ],
            "top_level": {
                name: {
                    "file_count": top_level_counts[name],
                    "tracked_bytes": top_level_bytes[name],
                }
                for name in sorted(top_level_counts)
            },
            "limits": {
                "max_file_count": int(repo_policy["max_file_count"]),
                "max_total_bytes": int(repo_policy["max_total_bytes"]),
            },
            "limit_utilization": {
                "file_count_fraction": round(file_fraction, 6),
                "total_bytes_fraction": round(byte_fraction, 6),
            },
        }

    all_duplicate_groups = [
        {
            "sha256": digest,
            "size_bytes": int(files[0]["size_bytes"]),
            "files": files,
        }
        for digest, files in sorted(hashes.items())
        if len(files) > 1
    ]
    duplicate_groups = [
        group
        for group in all_duplicate_groups
        if int(group["size_bytes"]) >= duplicate_min
    ]
    intra_repository_duplicate_groups = [
        group
        for group in all_duplicate_groups
        if any(
            sum(
                1
                for item in group["files"]
                if item["repository"] == repository
            )
            > 1
            for repository in {item["repository"] for item in group["files"]}
        )
    ]
    informational_cross_repository_duplicates = [
        group
        for group in all_duplicate_groups
        if group not in duplicate_groups
        and group not in intra_repository_duplicate_groups
    ]
    for group in duplicate_groups:
        locations = [
            f"{item['repository']}/{item['path']}" for item in group["files"]
        ]
        errors.append(
            f"exact duplicate >= {duplicate_min} bytes: {', '.join(locations)}"
        )
    for group in intra_repository_duplicate_groups:
        locations = [
            f"{item['repository']}/{item['path']}" for item in group["files"]
        ]
        errors.append(
            "same-repository exact duplicate "
            f">= {intra_repo_duplicate_min} bytes: {', '.join(locations)}"
        )

    summary = {
        "schema_version": 2,
        "status": "passed" if not errors else "failed",
        "policy": str(POLICY_PATH.relative_to(ROOT)),
        "repositories": repository_summaries,
        "duplicate_groups": duplicate_groups,
        "intra_repository_duplicate_groups": intra_repository_duplicate_groups,
        "informational_cross_repository_duplicates": (
            informational_cross_repository_duplicates
        ),
        "soft_limit_alerts": soft_limit_alerts,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
