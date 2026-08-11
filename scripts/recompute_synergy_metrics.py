#!/usr/bin/env python3
"""Verify the released high-confidence synergy replay capsule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    ordered = sorted(zip(scores, labels), reverse=True)
    true_positives = false_positives = 0
    previous_recall = area = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == score:
            true_positives += ordered[index][1]
            false_positives += 1 - ordered[index][1]
            index += 1
        recall = true_positives / positives
        area += (recall - previous_recall) * (
            true_positives / (true_positives + false_positives)
        )
        previous_recall = recall
    return area


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    ordered = sorted(zip(scores, labels))
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        positive_rank_sum += ((index + 1 + end) / 2) * sum(
            label for _, label in ordered[index:end]
        )
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def verify_sha256sums(root: Path) -> int:
    count = 0
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if sha256_file(root / relative) != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        count += 1
    return count


def verify(root: Path) -> dict[str, object]:
    checked_files = verify_sha256sums(root)
    summary = json.loads((root / "metadata/replay_summary.json").read_text())
    path = root / "data/all_fold_predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    member_columns = [f"member_{index}_probability" for index in range(7)]
    required = {
        "pair_identity",
        "measurement_index",
        "fold",
        "route",
        "strain_id",
        "label",
        *member_columns,
        "ensemble_probability",
    }
    if set(rows[0]) != required:
        raise ValueError("Unexpected prediction schema")
    identities = set()
    for row in rows:
        identity = (
            int(row["fold"]),
            row["pair_identity"],
            int(row["measurement_index"]),
        )
        if identity in identities:
            raise ValueError(f"Duplicate measurement identity: {identity}")
        identities.add(identity)
        expected = mean(float(row[column]) for column in member_columns)
        if not math.isclose(
            expected, float(row["ensemble_probability"]), rel_tol=0, abs_tol=1e-15
        ):
            raise ValueError("Ensemble mean mismatch")
    expected_folds = {int(item["fold"]): item for item in summary["folds"]}
    recomputed = []
    for fold in sorted(expected_folds):
        selected = [row for row in rows if int(row["fold"]) == fold]
        labels = [int(row["label"]) for row in selected]
        scores = [float(row["ensemble_probability"]) for row in selected]
        metrics = {
            "auroc": roc_auc(labels, scores),
            "auprc": average_precision(labels, scores),
        }
        frozen = expected_folds[fold]
        if len(selected) != int(frozen["rows"]):
            raise ValueError(f"Fold {fold} row-count mismatch")
        for name, value in metrics.items():
            if not math.isclose(
                value, float(frozen["metrics"][name]), rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(f"Fold {fold} {name} mismatch")
        recomputed.append({"fold": fold, "rows": len(selected), **metrics})
    return {
        "status": "passed",
        "checked_files": checked_files,
        "rows": len(rows),
        "member_columns": len(member_columns),
        "folds": recomputed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.capsule_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
