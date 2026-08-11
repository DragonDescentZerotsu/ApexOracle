import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = load_script("recompute_synergy_metrics.py")


def test_binary_metrics_match_perfect_ranking() -> None:
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8]
    assert CHECKER.roc_auc(labels, scores) == pytest.approx(1.0)
    assert CHECKER.average_precision(labels, scores) == pytest.approx(1.0)


def test_tied_binary_metrics_are_supported() -> None:
    labels = [0, 1]
    scores = [0.5, 0.5]
    assert CHECKER.roc_auc(labels, scores) == pytest.approx(0.5)
    assert CHECKER.average_precision(labels, scores) == pytest.approx(0.5)


def test_measurement_identity_scope_includes_fold() -> None:
    first = (0, "pair", 0)
    second = (1, "pair", 0)
    assert first != second
