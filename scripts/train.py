"""Train an age regressor and select its checkpoint by validation MAE."""

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from age_estimation.data import AgeDataset, preprocessing_config
from age_estimation.locking import DEFAULT_LOCK, PROJECT, verify_dataset_lock
from age_estimation.models import CHECKPOINT_VERSION, RESNET18_WEIGHTS, build_model, load_checkpoint


STRATEGIES = ("cnn", "warmup", "frozen", "finetune")
FIXED_BACKBONE = ("warmup", "frozen")
HISTORY_COLUMNS = (
    "epoch", "train_mse", "validation_mae", "validation_rmse",
    "train_seconds", "validation_seconds", "checkpoint_seconds",
)


def create_model(args, lock_id):
    source_hash = None
    if args.init_checkpoint:
        with args.init_checkpoint.open("rb") as file:
            source_hash = hashlib.file_digest(file, "sha256").hexdigest()
            file.seek(0)
            model, checkpoint = load_checkpoint(file)
        source = checkpoint["config"]
        if source["architecture"] != "resnet18" or source["strategy"] != "warmup":
            raise ValueError("Frozen and finetune runs must start from a ResNet18 warmup checkpoint")
        if source["dataset_lock_id"] != lock_id:
            raise ValueError("Warmup checkpoint belongs to a different dataset lock")
        return model, source_hash
    architecture = "cnn" if args.strategy == "cnn" else "resnet18"
    return build_model(architecture, pretrained=args.strategy == "warmup"), source_hash


def build_optimizer(model, args):
    if args.strategy in FIXED_BACKBONE:
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    groups = [{"params": [parameter for parameter in model.parameters() if parameter.requires_grad]}]
    if args.strategy == "finetune":
        head = list(model.fc.parameters())
        head_ids = {id(parameter) for parameter in head}
        groups = [
            {"params": [parameter for parameter in model.parameters() if id(parameter) not in head_ids],
             "lr": args.backbone_lr},
            {"params": head},
        ]
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def load_split(args, split, limit):
    dataset = AgeDataset(args.data, args.splits / f"{split}.csv", training=split == "train")
    if limit is not None:
        if limit > len(dataset):
            raise ValueError(f"Requested {limit} {split} images, found {len(dataset)}")
        indices = sorted(random.Random(args.seed).sample(range(len(dataset)), limit))
        dataset.samples = [dataset.samples[index] for index in indices]
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split == "train",
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )


def train_epoch(model, loader, optimizer, criterion, device, *, freeze_backbone):
    if freeze_backbone:
        # In eval mode the frozen ResNet BatchNorm buffers are fixed as well.
        model.eval()
        model.fc.train()
    else:
        model.train()
    total_loss = 0.0
    for images, ages in loader:
        images, ages = images.to(device), ages.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images).squeeze(1), ages)
        batch_mse = loss.item()
        if not math.isfinite(batch_mse):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total_loss += batch_mse * ages.numel()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    predictions, ages = [], []
    for images, labels in loader:
        predictions.append(model(images.to(device)).squeeze(1).cpu())
        ages.append(labels)
    predictions = torch.cat(predictions)
    if not torch.isfinite(predictions).all():
        raise RuntimeError("Non-finite validation prediction")
    errors = predictions.double() - torch.cat(ages).double()
    return {
        "mae": errors.abs().mean().item(),
        "rmse": errors.square().mean().sqrt().item(),
    }, predictions


def save_checkpoint(output, model, config, epoch, metrics):
    destination = output / "best.pt"
    temporary = destination.with_suffix(".tmp")
    torch.save({
        "version": CHECKPOINT_VERSION, "model": model.state_dict(),
        "config": config, "epoch": epoch, "metrics": metrics,
    }, temporary)
    temporary.replace(destination)


def export_predictions(path, samples, predictions):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("path", "age", "prediction", "error", "absolute_error"))
        for (image, age), prediction in zip(samples, predictions.tolist(), strict=True):
            writer.writerow((image.as_posix(), age, prediction, prediction - age, abs(prediction - age)))


def run(args):
    lock_id = verify_dataset_lock(args.splits, args.lock)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.hub.set_dir(str(PROJECT / "artifacts/torch/hub"))
    device = torch.device(args.device)
    train = load_split(args, "train", args.train_limit)
    validation = load_split(args, "validation", args.validation_limit)
    started = time.perf_counter()
    model, source_hash = create_model(args, lock_id)
    model.to(device)
    optimizer = build_optimizer(model, args)
    criterion = nn.MSELoss()
    config = {
        "architecture": "cnn" if args.strategy == "cnn" else "resnet18",
        "strategy": args.strategy, "seed": args.seed, "device": str(device),
        "pretrained_weights": None if args.strategy == "cnn" else RESNET18_WEIGHTS.name,
        "preprocessing": preprocessing_config(), "dataset_lock_id": lock_id,
        "init_checkpoint_sha256": source_hash, "loss": "MSE", "optimizer": args.optimizer,
        "lr": args.lr, "backbone_lr": args.backbone_lr if args.strategy == "finetune" else None,
        "weight_decay": args.weight_decay, "epochs": args.epochs, "batch_size": args.batch_size,
        "workers": args.workers, "train_samples": len(train.dataset),
        "validation_samples": len(validation.dataset),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    best_mae = math.inf
    with (args.output / "history.csv").open("w", encoding="utf-8", newline="") as file:
        history = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        history.writeheader()
        for epoch in range(1, args.epochs + 1):
            before_train = time.perf_counter()
            mse = train_epoch(model, train, optimizer, criterion, device,
                              freeze_backbone=args.strategy in FIXED_BACKBONE)
            train_seconds = time.perf_counter() - before_train
            before_validation = time.perf_counter()
            metrics, epoch_predictions = evaluate(model, validation, device)
            validation_seconds = time.perf_counter() - before_validation
            checkpoint_seconds = 0.0
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_predictions = epoch_predictions
                before_save = time.perf_counter()
                save_checkpoint(args.output, model, config, epoch, metrics)
                checkpoint_seconds = time.perf_counter() - before_save
            row = {
                "epoch": epoch, "train_mse": mse,
                "validation_mae": metrics["mae"], "validation_rmse": metrics["rmse"],
                "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                "checkpoint_seconds": checkpoint_seconds,
            }
            history.writerow(row)
            file.flush()
            print(f"{args.strategy} epoch {epoch}/{args.epochs} | MSE {mse:.3f} | "
                  f"val MAE {metrics['mae']:.3f} | val RMSE {metrics['rmse']:.3f}", flush=True)
    training_seconds = time.perf_counter() - started
    del optimizer, model
    before_reload = time.perf_counter()
    restored, checkpoint = load_checkpoint(args.output / "best.pt", device)
    restored_metrics, predictions = evaluate(restored, validation, device)
    torch.testing.assert_close(predictions, best_predictions, rtol=1e-5, atol=1e-5)
    for name, value in restored_metrics.items():
        if not math.isclose(value, checkpoint["metrics"][name], rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError(f"Reloaded checkpoint {name} differs from the saved validation metric")
    export_predictions(args.output / "validation_predictions.csv", validation.dataset.samples, predictions)
    report = {
        "config": config, "best_epoch": checkpoint["epoch"],
        "best_validation": restored_metrics, "training_seconds": training_seconds,
        "reload_and_export_seconds": time.perf_counter() - before_reload, "reload_verified": True,
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/raw/UTKFace")
    parser.add_argument("--splits", type=Path, default=PROJECT / "data/processed/utkface")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    args = parser.parse_args()
    if (args.strategy in ("frozen", "finetune")) != bool(args.init_checkpoint):
        parser.error("Supply --init-checkpoint for frozen and finetune only")
    positive = (args.epochs, args.batch_size, args.train_limit, args.validation_limit,
                args.lr, args.backbone_lr)
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in positive):
        parser.error("Epochs, batch size, sample limits and learning rates must be positive")
    if args.workers < 0 or args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        parser.error("Workers and weight decay must be non-negative")
    if args.strategy == "finetune" and args.backbone_lr >= args.lr:
        parser.error("Finetune backbone learning rate must be below the head learning rate")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for CPU diagnostics")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new run directory")
    run(args)


if __name__ == "__main__":
    main()
