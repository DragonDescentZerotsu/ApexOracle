import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_source_archive_plan_matches_all_five_locks():
    completed = subprocess.run(
        [sys.executable, "scripts/build_source_archive.py", "--plan-only"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    plan = json.loads(completed.stdout)
    assert len(plan["modules"]) == 5
    assert {module["id"] for module in plan["modules"]} == {
        "core", "dlm_pretraining", "mdlm", "evo2", "generation"
    }
