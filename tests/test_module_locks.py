import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_module_lock_checker_passes():
    completed = subprocess.run(
        [sys.executable, "scripts/check_module_locks.py"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["status"] == "passed"
    assert summary["ready_modules"] == [
        "modules/core",
        "modules/dlm_pretrain",
        "modules/evo2",
        "modules/generation",
        "modules/mdlm",
    ]
    assert summary["pending_modules"] == []
