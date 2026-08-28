#!/usr/bin/env python3
"""Validate that the active super-repo is compact and legacy-recoverable."""

from __future__ import annotations

import hashlib
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
)
FORBIDDEN_SUFFIXES = (".ckpt", ".pth", ".pt", ".bin", ".xlsx", ".png", ".pdf")
FORBIDDEN_README_SUBMODULE_LINKS = tuple(
    f"](modules/{module})"
    for module in ("core", "dlm_pretrain", "mdlm", "evo2", "generation", "virus")
)
REQUIRED_VISUAL_ASSETS = {
    "assets/ApexOracle_1.png": "761da4c0dfbf92bb2e6d4d5f536cc426b8cca159d0946fcd7c798a7e8504b0be",
    "assets/upenn.png": "b2e94cc500d1687a71f3763752571342c4fca1f6fe77db975e8b7d781d5a3f3f",
}
REQUIRED = (
    ".gitmodules",
    "AGENTS.md",
    "CITATION.cff",
    "LICENSE",
    "NOTICE",
    "README.md",
    *REQUIRED_VISUAL_ASSETS,
    "docs/LEGACY_MONOREPO.md",
    "docs/RELEASE_PROVENANCE.md",
    "docs/RELEASE_STATUS.md",
    "manifests/data_assets.yaml",
    "manifests/model_weights.yaml",
    "manifests/modules.lock.yaml",
    "scripts/check_module_locks.py",
)
REQUIRED_MODEL_ASSETS = {
    "apexoracle_molecule_embedding_model",
    "apexoracle_core_mic_single_member_quickstart",
    "apexoracle_generation_compact_baa3170_v1",
}
REQUIRED_DATA_ASSETS = {
    "apexoracle_core_mic_quickstart_input",
    "apexoracle_generation_baa3170_genome_condition",
    "apexoracle_generation_baa3170_text_condition",
    "apexoracle_zenodo_genome_embeddings",
    "apexoracle_zenodo_text_descriptions",
    "apexoracle_zenodo_fixed_mic_reconstruction",
    "apexoracle_zenodo_v3_release_manifest",
    "apexoracle_zenodo_synergy_replay",
    "apexoracle_zenodo_v4_release_manifest",
    "apexoracle_zenodo_model_ready_public_tables",
    "apexoracle_zenodo_v5_release_manifest",
    "apexoracle_core_paper_strain_mapping",
}


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
        path
        for path in relative
        if Path(path).suffix.lower() in FORBIDDEN_SUFFIXES
        and path not in REQUIRED_VISUAL_ASSETS
    )
    if forbidden_files:
        errors.append(f"binary/data files remain active: {forbidden_files}")
    for path, expected_sha256 in REQUIRED_VISUAL_ASSETS.items():
        asset_path = ROOT / path
        if asset_path.is_file():
            observed_sha256 = hashlib.sha256(asset_path.read_bytes()).hexdigest()
            if observed_sha256 != expected_sha256:
                errors.append(
                    f"README visual asset hash mismatch: {path} "
                    f"(expected {expected_sha256}, observed {observed_sha256})"
                )
    oversized = sorted(
        (path.relative_to(ROOT).as_posix(), path.stat().st_size)
        for path in files
        if not path.as_posix().startswith((ROOT / "modules").as_posix())
        and path.stat().st_size > MAX_ROOT_FILE_BYTES
    )
    if oversized:
        errors.append(f"root files exceed {MAX_ROOT_FILE_BYTES} bytes: {oversized}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if broken_links := sorted(
        link for link in FORBIDDEN_README_SUBMODULE_LINKS if link in readme
    ):
        errors.append(
            "README uses relative links that cannot traverse gitlinks: "
            f"{broken_links}"
        )

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

    model_manifest = json.loads((ROOT / "manifests/model_weights.yaml").read_text())
    data_manifest = json.loads((ROOT / "manifests/data_assets.yaml").read_text())
    model_asset_ids = {asset["id"] for asset in model_manifest["assets"]}
    data_asset_ids = {asset["id"] for asset in data_manifest["assets"]}
    if missing := sorted(REQUIRED_MODEL_ASSETS - model_asset_ids):
        errors.append(f"released model assets are missing: {missing}")
    if missing := sorted(REQUIRED_DATA_ASSETS - data_asset_ids):
        errors.append(f"released data assets are missing: {missing}")
    if any(
        "guided-generation example condition assets" == item
        for item in data_manifest["pending"]
    ):
        errors.append(
            "released generation conditions remain incorrectly marked pending"
        )

    summary = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "active_file_count": len(files),
        "legacy_commit": LEGACY_COMMIT,
        "released_model_asset_count": len(model_asset_ids),
        "released_data_asset_count": len(data_asset_ids),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
