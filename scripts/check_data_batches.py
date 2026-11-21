"""Check the locked split files and exercise the real Windows data loader."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from age_estimation.data import AgeDataset, INPUT_SIZE, build_preprocess
from age_estimation.locking import DEFAULT_LOCK, verify_dataset_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/UTKFace"))
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--against", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)

    lock_id = verify_dataset_lock(args.splits, args.lock)
    if args.against:
        verify_dataset_lock(args.against, args.lock)
        if (args.splits / "summary.json").read_bytes() != (args.against / "summary.json").read_bytes():
            raise ValueError("Rerun summary bytes differ")

    report = {
        "workers": args.workers,
        "dataset_lock_id": lock_id,
        "locked_files_verified": True,
        "rerun_identical": bool(args.against),
    }
    subsets = {}
    for split in ("train", "validation", "test"):
        data = AgeDataset(args.data, args.splits / f"{split}.csv", training=split == "train")
        images, ages = next(iter(DataLoader(data, batch_size=args.batch_size, num_workers=args.workers)))
        batch_count = min(args.batch_size, len(data))
        expected_ages = torch.tensor([age for _, age in data.samples[:batch_count]], dtype=torch.float32)
        if images.shape != (batch_count, 3, *INPUT_SIZE) or ages.shape != (batch_count,):
            raise RuntimeError(f"Unexpected batch dimensions: {split}")
        if images.dtype != torch.float32 or ages.dtype != torch.float32 or not torch.isfinite(images).all():
            raise RuntimeError(f"Unexpected batch data: {split}")
        torch.testing.assert_close(ages, expected_ages, rtol=0, atol=0)
        if split != "train":
            path, _ = data.samples[0]
            with Image.open(args.data / path) as image:
                reference = build_preprocess()(image)
            torch.testing.assert_close(images[0], reference, rtol=0, atol=0)
        else:
            path, _ = data.samples[0]
            with Image.open(args.data / path) as image:
                torch.manual_seed(42)
                first = data.preprocess(image)
                torch.manual_seed(43)
                second = data.preprocess(image)
            if torch.equal(first, second):
                raise RuntimeError("Training augmentation did not vary on the probe")
        subsets[split] = {
            "samples": len(data), "image_shape": list(images.shape), "label_shape": list(ages.shape),
            "image_dtype": str(images.dtype), "label_dtype": str(ages.dtype),
        }
    report["batches"] = subsets
    report["validation_and_test_match_standalone_preprocess"] = True
    report["train_augmentation_varies"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
