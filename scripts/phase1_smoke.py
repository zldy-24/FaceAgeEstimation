"""Verify UTKFace input, pretrained ResNet18, and a full training step."""

import argparse
import csv
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


PROJECT = Path(__file__).resolve().parents[1]
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class FaceSample(Dataset):
    def __init__(self, root, rows):
        self.root = root
        self.rows = rows
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            pixels = self.preprocess(image.convert("RGB"))
        return pixels, torch.tensor(row["age"], dtype=torch.float32)


def inspect_images(root, sample_size, seed):
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    rows = []
    for path in random.Random(seed).sample(paths, min(len(paths), sample_size)):
        row = {"path": path.relative_to(root).as_posix(), "age": "", "status": "ok"}
        age_text, separator, _ = path.name.partition("_")
        if not separator or not age_text.isascii() or not age_text.isdecimal():
            row["status"] = "invalid_age"
        else:
            age = int(age_text)
            row["age"] = age
            if not 0 <= age <= 116:
                row["status"] = "invalid_age"
            else:
                try:
                    with Image.open(path) as image:
                        image.load()
                except (OSError, ValueError, Image.DecompressionBombError):
                    row["status"] = "decode_failed"
        rows.append(row)
    return len(paths), rows


def train_and_measure(loader, device, args):
    use_cuda = device.type == "cuda"
    model = resnet18(weights=WEIGHTS)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    original_stem = model.conv1.weight.detach().clone()

    timings = []
    synchronize = torch.cuda.synchronize if use_cuda else lambda: None
    batches = iter(loader)
    for step in range(args.warmup + args.steps):
        if use_cuda and step == args.warmup:
            torch.cuda.reset_peak_memory_stats()
        synchronize()
        started = time.perf_counter()
        images, ages = next(batches)
        images, ages = images.to(device), ages.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images).squeeze(1), ages)
        loss.backward()
        optimizer.step()
        synchronize()
        elapsed = time.perf_counter() - started
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Non-finite training loss at batch {step + 1}")
        if step >= args.warmup:
            timings.append(elapsed)
    if torch.equal(original_stem, model.conv1.weight):
        raise RuntimeError("The ResNet18 backbone did not update")

    metrics = {
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup,
        "measured_batches": args.steps,
        "last_batch_mse": loss.item(),
        "mean_batch_ms": 1_000 * statistics.mean(timings),
        "median_batch_ms": 1_000 * statistics.median(timings),
        "images_per_second": args.batch_size * len(timings) / sum(timings),
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if use_cuda else None,
        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if use_cuda else None,
    }

    checkpoint = args.output / "smoke-resnet18.pt"
    torch.save({"model": model.state_dict(), "pretrained": WEIGHTS.name}, checkpoint)
    restored = resnet18(weights=None)
    restored.fc = nn.Linear(restored.fc.in_features, 1)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    probe = images[:2].cpu()
    model.cpu().eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/raw/UTKFace")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.warmup < 1:
        parser.error("batch-size, warmup, and steps must be positive")
    if not args.data.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data}")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Check the Windows CUDA build or use --device cpu.")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.hub.set_dir(str(PROJECT / "artifacts/torch/hub"))
    gpu_result = (torch.tensor([2.0], device=device) ** 2).item()
    if gpu_result != 4.0:
        raise RuntimeError("Device tensor operation gave an unexpected result")

    args.output.mkdir(parents=True, exist_ok=True)
    needed = args.batch_size * (args.warmup + args.steps)
    total_images, rows = inspect_images(args.data, needed + 32, args.seed)
    with (args.output / "sample.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "age", "status"])
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row["status"] == "ok"]
    if len(valid) < needed:
        parser.error(f"Need {needed} valid images for this run, found {len(valid)} in the sample")

    loader = DataLoader(FaceSample(args.data, valid[:needed]), batch_size=args.batch_size)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": args.device,
        "device_check": gpu_result,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20
        if device.type == "cuda" else None,
        "images_discovered": total_images,
        "images_sampled": len(rows),
        "images_sampled_valid": len(valid),
        "seed": args.seed,
        "pretrained_weights": WEIGHTS.name,
    }
    report.update(train_and_measure(loader, device, args))
    report["checkpoint_reload_verified"] = True
    text = json.dumps(report, indent=2)
    (args.output / "report.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
