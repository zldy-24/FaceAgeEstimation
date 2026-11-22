"""Independently recompute the formal metrics from exported records."""

import argparse
import json
import math
import statistics
from pathlib import Path

from age_estimation.locking import DEFAULT_LOCK, PROJECT, verify_dataset_lock
from scripts.evaluate_formal import MODELS, canonical_sha256, file_sha256, read_csv, read_json


def require_equal(expected, actual, context):
    if not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{context}: expected {expected}, found {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, default=PROJECT / "docs/phase4/selection.json")
    parser.add_argument("--splits", type=Path, default=PROJECT / "data/processed/utkface")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()

    folder = args.experiment / "test"
    recorded = read_json(folder / "evaluation.json")
    selection = read_json(args.selection_lock)
    if recorded["selection_sha256"] != canonical_sha256(selection):
        raise ValueError("Test evaluation does not reference the archived validation selection")
    if verify_dataset_lock(args.splits, args.lock) != selection["dataset_lock_id"]:
        raise ValueError("Test dataset differs from the selection")
    if recorded["completed_utc"] <= selection["selected_utc"]:
        raise ValueError("Test evaluation was not completed after validation selection")

    truth = read_csv(args.splits / "test.csv")
    overall = {row["strategy"]: row for row in read_csv(folder / "metrics.csv")}
    bands = {(row["strategy"], row["age_group"]): row for row in read_csv(folder / "age_group_metrics.csv")}
    latency = read_csv(folder / "latency_samples.csv")
    shortlisted = read_csv(folder / "outlier_candidates.csv")
    checked = {}
    for name in MODELS:
        predicted = read_csv(folder / f"predictions_{name}.csv")
        if len(predicted) != int(overall[name]["samples"]):
            raise ValueError(f"Incorrect total count for {name}")
        all_errors, group_errors = [], {}
        for prediction, label in zip(predicted, truth, strict=True):
            if prediction["path"] != label["path"] or prediction["age"] != label["age"]:
                raise ValueError(f"Test prediction does not match the manifest for {name}")
            error = float(prediction["prediction"]) - int(label["age"])
            require_equal(error, float(prediction["error"]), f"{name} signed sample error")
            require_equal(abs(error), float(prediction["absolute_error"]), f"{name} absolute sample error")
            all_errors.append(error)
            group_errors.setdefault(label["age_group"], []).append(error)
        sample_count = len(all_errors)
        mae = math.fsum(abs(error) for error in all_errors) / sample_count
        rmse = math.sqrt(math.fsum(error ** 2 for error in all_errors) / sample_count)
        for metric, value in (("mae", mae), ("rmse", rmse)):
            require_equal(value, float(overall[name][metric]), f"{name} {metric}")
        for group, errors in group_errors.items():
            row = bands[name, group]
            if len(errors) != int(row["samples"]):
                raise ValueError(f"Incorrect age group count for {name} {group}")
            require_equal(math.fsum(map(abs, errors)) / len(errors), float(row["mae"]), f"{name} {group} MAE")

        times = [float(row["milliseconds"]) for row in latency if row["strategy"] == name]
        if len(times) != recorded["inference_protocol"]["measured_forwards"] or any(value <= 0 for value in times):
            raise ValueError(f"Incorrect latency sample count or value for {name}")
        require_equal(statistics.mean(times), float(overall[name]["latency_mean_ms"]), f"{name} mean latency")
        require_equal(statistics.median(times), float(overall[name]["latency_median_ms"]), f"{name} median latency")
        require_equal(statistics.quantiles(times, n=100, method="inclusive")[94],
                      float(overall[name]["latency_p95_ms"]), f"{name} p95 latency")
        run_seconds = selection["runs"][name]["training_seconds"]
        warmup_seconds = selection["runs"]["warmup"]["training_seconds"] if name != "cnn" else 0
        require_equal(run_seconds, float(overall[name]["training_seconds"]), f"{name} training time")
        require_equal(warmup_seconds, float(overall[name]["shared_warmup_seconds"]), f"{name} warmup time")
        require_equal(run_seconds + warmup_seconds, float(overall[name]["standalone_training_seconds"]),
                      f"{name} standalone training time")
        top = sorted(range(sample_count), key=lambda index: (-abs(all_errors[index]), index))[:5]
        if [truth[index]["path"] for index in top] != [row["path"] for row in shortlisted if row["strategy"] == name]:
            raise ValueError(f"Outlier shortlist differs from the exported predictions for {name}")
        checked[name] = {
            "predictions": sample_count, "age_groups": len(group_errors),
            "latency_samples": len(times), "mae": mae, "rmse": rmse,
        }
    print(json.dumps({
        "selection_sha256": recorded["selection_sha256"],
        "verified": checked,
        "output_sha256": {path.name: file_sha256(path) for path in sorted(folder.iterdir()) if path.is_file()},
    }, indent=2))


if __name__ == "__main__":
    main()
