#!/usr/bin/env python3
"""Verify a fixed-split MIC reconstruction capsule using the Python standard library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


SUBSETS = ("full", "train_seen", "train_unseen")
MODELS = (
    "apexoracle_ensemble",
    "peptide_mean_or_train_mean",
    "training_mean_constant",
    "always_512um_constant",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected True/False, got {value!r}")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            output[index] = average_rank
        start = end
    return output


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0:
        return float("nan")
    return (
        sum(a * b for a, b in zip(left_delta, right_delta, strict=True)) / denominator
    )


def calculate_metrics(
    labels: list[float], predictions: list[float]
) -> dict[str, float]:
    label_mean = mean(labels)
    denominator = sum((value - label_mean) ** 2 for value in labels)
    r2 = (
        1.0
        - sum(
            (label - prediction) ** 2
            for label, prediction in zip(labels, predictions, strict=True)
        )
        / denominator
    )
    return {
        "r2": r2,
        "spearman": pearson(ranks(labels), ranks(predictions)),
        "pearson": pearson(labels, predictions),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_sha256sums(root: Path) -> int:
    count = 0
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative}: {actual} != {expected}")
        count += 1
    return count


def verify_member_average(root: Path, ensemble_rows: list[dict[str, str]]) -> int:
    by_key: dict[str, list[float]] = {}
    member_files = sorted((root / "data/member_predictions").glob("*.csv"))
    for path in member_files:
        for row in read_rows(path):
            by_key.setdefault(row["row_key"], []).append(float(row["prediction"]))
    for row in ensemble_rows:
        predictions = by_key.pop(row["row_key"], None)
        if predictions is None:
            raise ValueError(f"Missing member predictions for {row['row_key']}")
        expected_members = int(row["ensemble_members"])
        if len(predictions) != expected_members:
            raise ValueError(
                f"{row['row_key']} has {len(predictions)} members, expected {expected_members}"
            )
        if not math.isclose(
            mean(predictions), float(row["prediction"]), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"Ensemble mean mismatch for {row['row_key']}")
    if by_key:
        raise ValueError(f"Member predictions contain {len(by_key)} unknown row keys")
    return len(member_files)


def select_subset(rows: list[dict[str, str]], subset: str) -> list[dict[str, str]]:
    if subset == "full":
        return rows
    expected = subset == "train_seen"
    return [
        row for row in rows if parse_bool(row["train_seen_exact_molecule"]) is expected
    ]


def recompute_metric_rows(
    ensemble_rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    groups = sorted({row["group_index"] for row in ensemble_rows}, key=int)
    scopes = [
        (group, [row for row in ensemble_rows if row["group_index"] == group])
        for group in groups
    ]
    scopes.append(("all_groups_pooled", ensemble_rows))
    output: list[dict[str, object]] = []
    high_mic_z = -math.log10(512.0 / 10.0)
    for group, rows in scopes:
        for subset in SUBSETS:
            selected = select_subset(rows, subset)
            labels = [float(row["label_z"]) for row in selected]
            predictors = {
                "apexoracle_ensemble": [float(row["prediction"]) for row in selected],
                "peptide_mean_or_train_mean": [
                    float(row["peptide_mean_baseline_prediction"]) for row in selected
                ],
                "training_mean_constant": [
                    float(row["training_mean_baseline_prediction"]) for row in selected
                ],
                "always_512um_constant": [high_mic_z] * len(selected),
            }
            for model in MODELS:
                row: dict[str, object] = {
                    "protocol": "strain",
                    "group": group,
                    "subset": subset,
                    "model": model,
                    "measurements": len(selected),
                    "unique_molecules": len(
                        {item["molecule_identity"] for item in selected}
                    ),
                    "pathogens": len({item["strain_id"] for item in selected}),
                    "low_mic_16um_measurements": sum(
                        parse_bool(item["low_mic_16um"]) for item in selected
                    ),
                }
                row["low_mic_16um_fraction"] = row["low_mic_16um_measurements"] / len(
                    selected
                )
                row.update(calculate_metrics(labels, predictors[model]))
                output.append(row)
    return output


def verify_metrics(root: Path, ensemble_rows: list[dict[str, str]]) -> int:
    expected_rows = read_rows(root / "data/metrics.csv")
    expected = {
        (row["group"], row["subset"], row["model"]): row for row in expected_rows
    }
    actual_rows = recompute_metric_rows(ensemble_rows)
    for actual in actual_rows:
        key = (str(actual["group"]), str(actual["subset"]), str(actual["model"]))
        frozen = expected.pop(key, None)
        if frozen is None:
            raise ValueError(f"Missing frozen metric row {key}")
        for field in (
            "measurements",
            "unique_molecules",
            "pathogens",
            "low_mic_16um_measurements",
        ):
            if int(frozen[field]) != int(actual[field]):
                raise ValueError(f"Metric count mismatch for {key} field {field}")
        for field in ("low_mic_16um_fraction", "r2", "spearman", "pearson"):
            frozen_value = frozen[field]
            actual_value = float(actual[field])
            if frozen_value == "" and math.isnan(actual_value):
                continue
            if not math.isclose(
                float(frozen_value), actual_value, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(f"Metric mismatch for {key} field {field}")
    if expected:
        raise ValueError(f"Unexpected frozen metric rows: {sorted(expected)}")
    return len(actual_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule_root", type=Path)
    args = parser.parse_args()
    root = args.capsule_root.resolve()
    checked_files = verify_sha256sums(root)
    ensemble_rows = read_rows(root / "data/ensemble_predictions.csv")
    member_files = verify_member_average(root, ensemble_rows)
    metric_rows = verify_metrics(root, ensemble_rows)
    print(
        json.dumps(
            {
                "status": "passed",
                "checked_files": checked_files,
                "ensemble_rows": len(ensemble_rows),
                "member_files": member_files,
                "metric_rows": metric_rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
