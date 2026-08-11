#!/usr/bin/env python3
"""Recompute Fig. 1b classification metrics from a released capsule."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METHODS = (
    "apexoracle_strict_zero_shot",
    "apexoracle_finetuned_10member",
    "chemprop_10member",
)


def average_precision(labels: list[int], scores: list[float]) -> float:
    """Match sklearn's non-interpolated binary average precision."""

    positives = sum(labels)
    if positives == 0:
        raise ValueError("average precision requires at least one positive label")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == score:
            if ordered[index][1] == 1:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def roc_auc(labels: list[int], scores: list[float]) -> float:
    """Compute binary AUROC via average ranks, including tied scores."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"molecule_id", "label", "prediction", "group", "fold"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(
                f"{path}: expected columns {sorted(required)}, got {reader.fieldnames}"
            )
        records: dict[str, dict[str, str]] = {}
        for row in reader:
            molecule_id = row["molecule_id"]
            if molecule_id in records:
                raise ValueError(f"{path}: duplicate molecule_id {molecule_id}")
            records[molecule_id] = row
    return records


def metrics(records: list[dict[str, str]]) -> dict[str, float | int]:
    labels = [int(float(record["label"])) for record in records]
    scores = [float(record["prediction"]) for record in records]
    return {
        "rows": len(records),
        "positives": sum(labels),
        "auprc": average_precision(labels, scores),
        "auroc": roc_auc(labels, scores),
    }


def compute_capsule_metrics(capsule_root: Path) -> dict[str, object]:
    predictions: dict[int, dict[str, dict[str, dict[str, str]]]] = {}
    for group in range(3):
        predictions[group] = {}
        for method in METHODS:
            predictions[group][method] = load_predictions(
                capsule_root / "predictions" / method / f"group_{group}.csv"
            )

    groups: dict[str, object] = {}
    for group, methods in predictions.items():
        common_ids = set.intersection(
            *(set(records) for records in methods.values())
        )
        method_results: dict[str, object] = {}
        for method, records in methods.items():
            full_records = list(records.values())
            common_records = [records[molecule_id] for molecule_id in common_ids]
            fold_metrics = []
            for fold in range(5):
                fold_records = [
                    record
                    for record in common_records
                    if int(record["fold"]) == fold
                ]
                fold_metrics.append({"fold": fold, **metrics(fold_records)})
            method_results[method] = {
                "all_available": metrics(full_records),
                "common_comparison": metrics(common_records),
                "fold_metrics_on_common_comparison": fold_metrics,
                "fold_mean_on_common_comparison": {
                    metric: mean(record[metric] for record in fold_metrics)
                    for metric in ("auprc", "auroc")
                },
                "fold_sample_sd_on_common_comparison": {
                    metric: stdev(record[metric] for record in fold_metrics)
                    for metric in ("auprc", "auroc")
                },
            }

        reference = methods[METHODS[0]]
        for method, records in methods.items():
            for molecule_id in common_ids:
                if records[molecule_id]["label"] != reference[molecule_id]["label"]:
                    raise ValueError(
                        f"group {group}: label mismatch for {molecule_id} in {method}"
                    )
                if records[molecule_id]["fold"] != reference[molecule_id]["fold"]:
                    raise ValueError(
                        f"group {group}: fold mismatch for {molecule_id} in {method}"
                    )
        groups[str(group)] = {
            "common_comparison_rows": len(common_ids),
            "common_comparison_ids_sha256": _ids_sha256(common_ids),
            "methods": method_results,
        }

    return {"schema_version": 1, "groups": groups}


def _ids_sha256(ids: set[str]) -> str:
    import hashlib

    payload = "".join(f"{item}\n" for item in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capsule-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compute_capsule_metrics(args.capsule_root.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
