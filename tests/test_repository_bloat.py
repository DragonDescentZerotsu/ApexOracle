import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repository_bloat_checker_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_repository_bloat.py"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["status"] == "passed"
    assert summary["duplicate_groups"] == []
    assert set(summary["repositories"]) == {
        "root",
        "core",
        "dlm_pretraining",
        "mdlm",
        "evo2",
        "generation",
    }
    assert all(
        not record["untracked_nonignored"]
        for record in summary["repositories"].values()
    )
