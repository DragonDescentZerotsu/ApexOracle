import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_tree_checker_passes():
    completed = subprocess.run(
        [sys.executable, "scripts/check_release_tree.py"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["status"] == "passed"
    assert summary["legacy_commit"] == "2f29dee9cf6b7750425414f66c1a2d67998cb87f"
    assert summary["released_model_asset_count"] == 3
    assert summary["released_data_asset_count"] == 6
