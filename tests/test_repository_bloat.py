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
    assert summary["schema_version"] == 2
    assert summary["status"] == "passed"
    assert summary["duplicate_groups"] == []
    assert summary["intra_repository_duplicate_groups"] == []
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
    assert all(record["top_level"] for record in summary["repositories"].values())
    assert all(
        0 <= record["limit_utilization"]["file_count_fraction"] <= 1
        and 0 <= record["limit_utilization"]["total_bytes_fraction"] <= 1
        for record in summary["repositories"].values()
    )
    assert all(
        len({item["repository"] for item in group["files"]}) > 1
        for group in summary["informational_cross_repository_duplicates"]
    )
