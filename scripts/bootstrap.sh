#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git submodule sync --recursive
git submodule update --init --recursive
python scripts/check_release_tree.py
python scripts/check_module_locks.py

echo "ApexOracle release tree and initialized modules match the frozen manifests"
