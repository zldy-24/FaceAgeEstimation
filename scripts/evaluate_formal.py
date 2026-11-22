"""Seal the validation-based selection, then evaluate the three test models."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from age_estimation.data import AgeDataset
from age_estimation.locking import DEFAULT_LOCK, PROJECT, verify_dataset_lock
from age_estimation.models import load_checkpoint
from scripts.prepare_data import AGE_GROUPS, write_csv
from scripts.train import evaluate, export_predictions


MODELS = ("cnn", "frozen", "finetune")
RUNS = ("cnn", "warmup", "frozen", "finetune")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def file_sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def describe_selection(experiment, protocol):
    training, counts = protocol["training"], protocol["sample_counts"]
    warmup_hash = file_sha256(experiment / "warmup/best.pt")
    runs = {}
    for strategy in RUNS:
        folder = experiment / strategy
        result = read_json(folder / "run.json")
        expected = {
            key: training[key] for key in ("seed", "device", "batch_size", "workers", "optimizer", "lr", "weight_decay")
        }
        expected.update({
            "strategy": strategy, "epochs": training["epochs"][strategy], "loss": "MSE",
            "dataset_lock_id": protocol["dataset_lock_id"],
            "train_samples": counts["train"], "validation_samples": counts["validation"],
            "architecture": "cnn" if strategy == "cnn" else "resnet18",
            "init_checkpoint_sha256": warmup_hash if strategy in ("frozen", "finetune") else None,
            "backbone_lr": training["finetune_backbone_lr"] if strategy == "finetune" else None,
        })
        for setting, value in expected.items():
            if result["config"][setting] != value:
                raise ValueError(f"{strategy} does not match the protocol for {setting}")
        history = read_csv(folder / "history.csv")
        if [int(row["epoch"]) for row in history] != list(range(1, expected["epochs"] + 1)):
            raise ValueError(f"{strategy} did not finish its planned epochs")
        best = min(history, key=lambda row: (float(row["validation_mae"]), int(row["epoch"])))
        validation = result["best_validation"]
        for metric in ("mae", "rmse"):
            if not math.isclose(validation[metric], float(best[f"validation_{metric}"]), rel_tol=1e-6):
                raise ValueError(f"{strategy} has inconsistent saved validation {metric}")
        if not result["reload_verified"] or result["best_epoch"] != int(best["epoch"]):
            raise ValueError(f"{strategy} has an invalid checkpoint selection")
        runs[strategy] = {
            "checkpoint": f"{strategy}/best.pt",
            "sha256": file_sha256(folder / "best.pt"),
            "best_epoch": result["best_epoch"],
            "validation": validation,
            "training_seconds": result["training_seconds"],
            "reload_and_export_seconds": result["reload_and_export_seconds"],
        }
    demo = min(MODELS, key=lambda strategy: runs[strategy]["validation"]["mae"])
    return {
        "dataset_lock_id": protocol["dataset_lock_id"], "protocol_id": canonical_sha256(protocol),
        "demo_strategy": demo, "runs": runs,
    }


def current_device():
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).splitlines()[0].strip()
    return {
        "device": "cuda", "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "driver": driver,
    }


@torch.inference_mode()
def time_single_image(model, tensor, protocol):
    for _ in range(protocol["warmup_forwards"]):
        model(tensor)
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(protocol["measured_forwards"])
    ]
    for before, after in events:
        before.record()
        after.record()
    torch.cuda.synchronize()
    # Record CUDA stream time with the tensor already on the device.
    for before, after in events:
        before.record()
        model(tensor)
        after.record()
    torch.cuda.synchronize()
    return [before.elapsed_time(after) for before, after in events]


def evaluate_test(args, protocol, selection):
    device_info = current_device()
    bench = protocol["inference"]
    for key, value in device_info.items():
        if value != bench[key]:
            raise RuntimeError(f"Timing environment differs for {key}: {value!r}")
    torch.set_num_threads(4)
    device = torch.device(bench["device"])
    test = AgeDataset(args.data, args.splits / "test.csv")
    labels = read_csv(args.splits / "test.csv")
    if len(test) != protocol["sample_counts"]["test"]:
        raise ValueError("Test size differs from the protocol")
    loader = DataLoader(test, batch_size=protocol["training"]["batch_size"])
    probe = test[0][0].unsqueeze(0).to(device)
    group_indices = {}
    for index, label in enumerate(labels):
        group_indices.setdefault(label["age_group"], []).append(index)
    ages = torch.tensor([age for _, age in test.samples], dtype=torch.float64)
    metrics, groups, latency, outliers = [], [], [], []
    results = args.experiment / "test"
    if results.exists():
        raise FileExistsError(f"Test evaluation already exists: {results}")

    with tempfile.TemporaryDirectory(prefix=".test-", dir=args.experiment) as temporary:
        directory = Path(temporary) / "result"
        directory.mkdir()
        for strategy in MODELS:
            run = selection["runs"][strategy]
            weight = args.experiment / run["checkpoint"]
            with weight.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != run["sha256"]:
                    raise ValueError(f"Selected checkpoint changed: {strategy}")
                file.seek(0)
                model, checkpoint = load_checkpoint(file, device)
            if (checkpoint["epoch"] != run["best_epoch"] or
                checkpoint["config"]["strategy"] != strategy or
                checkpoint["config"]["dataset_lock_id"] != selection["dataset_lock_id"]):
                raise ValueError(f"Bad selected checkpoint metadata: {strategy}")
            errors, predictions = evaluate(model, loader, device)
            export_predictions(directory / f"predictions_{strategy}.csv", test.samples, predictions)
            absolute = (predictions.double() - ages).abs()
            for _, band in AGE_GROUPS:
                indices = group_indices[band]
                groups.append({
                    "strategy": strategy, "age_group": band, "samples": len(indices),
                    "mae": absolute[indices].mean().item(),
                })
            for rank, index in enumerate(sorted(range(len(test)), key=lambda i: (-absolute[i].item(), i))[:5], 1):
                path, age = test.samples[index]
                outliers.append({
                    "strategy": strategy, "rank": rank, "path": path.as_posix(),
                    "age": age, "prediction": predictions[index].item(), "absolute_error": absolute[index].item(),
                })
            times = time_single_image(model, probe, bench)
            latency.extend({"strategy": strategy, "repeat": index, "milliseconds": value}
                           for index, value in enumerate(times, 1))
            warmup = selection["runs"]["warmup"]["training_seconds"] if strategy != "cnn" else 0.0
            metrics.append({
                "strategy": strategy, "samples": len(test), **errors,
                "validation_mae": run["validation"]["mae"], "best_epoch": run["best_epoch"],
                "training_seconds": run["training_seconds"], "shared_warmup_seconds": warmup,
                "standalone_training_seconds": run["training_seconds"] + warmup,
                "latency_mean_ms": statistics.mean(times), "latency_median_ms": statistics.median(times),
                "latency_p95_ms": statistics.quantiles(times, n=100, method="inclusive")[94],
            })
            print(f"{strategy} test MAE {errors['mae']:.4f}, RMSE {errors['rmse']:.4f}", flush=True)
            del model, checkpoint
        for filename, rows in (
            ("metrics.csv", metrics), ("age_group_metrics.csv", groups),
            ("outlier_candidates.csv", outliers), ("latency_samples.csv", latency),
        ):
            write_csv(directory / filename, rows, tuple(rows[0]))
        report = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": canonical_sha256(selection),
            "device": device_info, "inference_probe": test.samples[0][0].as_posix(), "inference_protocol": bench,
            "unique_training_seconds": sum(run["training_seconds"] for run in selection["runs"].values()),
        }
        (directory / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        directory.rename(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "test"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROJECT / "docs/phase4/protocol.json")
    parser.add_argument("--selection-lock", type=Path, default=PROJECT / "docs/phase4/selection.json")
    parser.add_argument("--data", type=Path, default=PROJECT / "data/raw/UTKFace")
    parser.add_argument("--splits", type=Path, default=PROJECT / "data/processed/utkface")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if protocol != read_json(args.experiment / "protocol.json"):
        raise ValueError("The experiment snapshot differs from the archived protocol")
    if verify_dataset_lock(args.splits, args.lock) != protocol["dataset_lock_id"]:
        raise ValueError("Dataset lock differs from the experiment protocol")
    fresh = describe_selection(args.experiment, protocol)
    selection_path = args.selection_lock
    if selection_path.resolve().is_relative_to(args.experiment.resolve()):
        raise ValueError("The validation selection must be archived outside the run directory")
    if args.action == "select":
        selection = {**fresh, "selected_utc": datetime.now(timezone.utc).isoformat()}
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        with selection_path.open("x", encoding="utf-8") as file:
            json.dump(selection, file, indent=2)
            file.write("\n")
        print(f"Validation selection sealed. Demo strategy: {fresh['demo_strategy']}")
    else:
        selection = read_json(selection_path)
        if {key: selection[key] for key in fresh} != fresh:
            raise ValueError("Training results or weights changed after selection")
        evaluate_test(args, protocol, selection)


if __name__ == "__main__":
    main()
