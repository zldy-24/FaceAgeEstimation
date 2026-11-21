"""Audit UTKFace and publish reproducible age-stratified split manifests."""

import argparse
import csv
import hashlib
import io
import json
import random
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FILENAME = re.compile(
    r"(?P<age>-?[0-9]+)_[01]_[0-4]_[0-9]{17}(?:\.jpg\.chip)?\.(?:jpe?g|png)",
    re.IGNORECASE,
)
AGE_GROUPS = ((12, "0-12"), (17, "13-17"), (39, "18-39"), (59, "40-59"), (116, "60+"))
SPLITS = ("train", "validation", "test")
SPLIT_FIELDS = ("path", "age", "age_group", "file_sha256", "pixel_sha256")
AUDIT_FIELDS = (*SPLIT_FIELDS, "read_status", "label_status", "status", "duplicate_of")
PAIR_FIELDS = ("path_a", "path_b", "age_a", "age_b", "dhash_distance", "gray_mae")
MAX_DISTANCE = 4
MAX_GRAY_MAE = 12.0


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def scan_images(root):
    rows = []
    thumbnails = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        row = dict.fromkeys(AUDIT_FIELDS, "")
        row["path"] = relative
        label = FILENAME.fullmatch(path.name)
        if label is None:
            row["label_status"] = "invalid_filename"
        else:
            age = int(label["age"])
            row["age"] = age
            if 0 <= age <= 116:
                row["age_group"] = next(group for maximum, group in AGE_GROUPS if age <= maximum)
                row["label_status"] = "ok"
            else:
                row["label_status"] = "out_of_range_age"
        try:
            data = path.read_bytes()
        except OSError:
            row["read_status"] = "read_failed"
        else:
            row["file_sha256"] = sha256(data)
            try:
                with Image.open(io.BytesIO(data)) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    row["read_status"] = "ok"
            except (OSError, ValueError, Image.DecompressionBombError):
                row["read_status"] = "decode_failed"
            else:
                if row["label_status"] == "ok":
                    row["pixel_sha256"] = sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes())
                    gray = rgb.convert("L")
                    grid = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
                    dhash = int.from_bytes(np.packbits(grid[:, 1:] > grid[:, :-1]).tobytes(), "big")
                    thumbnail = np.asarray(gray.resize((16, 16), Image.Resampling.LANCZOS), dtype=np.int16)
                    thumbnails[relative] = (dhash, thumbnail)
        row["status"] = (
            row["read_status"] if row["read_status"] != "ok" else row["label_status"]
        )
        rows.append(row)
    return rows, thumbnails


def remove_exact_duplicates(rows):
    by_pixels = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_pixels[row["pixel_sha256"]].append(row)
    for group in by_pixels.values():
        if len(group) < 2:
            continue
        if len({row["age"] for row in group}) > 1:
            for row in group:
                row["status"] = "conflicting_exact_duplicate_age"
            continue
        kept, *duplicates = group
        for row in duplicates:
            row["status"] = (
                "exact_file_duplicate" if row["file_sha256"] == kept["file_sha256"]
                else "exact_pixel_duplicate"
            )
            row["duplicate_of"] = kept["path"]


def find_near_duplicates(rows, thumbnails):
    # Five disjoint bands ensure hashes at distance four or less share a band.
    bands = defaultdict(list)
    candidates = []
    for row in rows:
        if row["status"] != "ok":
            continue
        dhash, thumbnail = thumbnails[row["path"]]
        keys = [(offset, (dhash >> offset) & 0x1FFF) for offset in range(0, 64, 13)]
        neighbors = {previous for key in keys for previous in bands[key]}
        for previous_path, previous_age in neighbors:
            previous_hash, previous_thumbnail = thumbnails[previous_path]
            distance = (dhash ^ previous_hash).bit_count()
            if distance <= MAX_DISTANCE:
                error = float(np.abs(thumbnail - previous_thumbnail).mean())
                if error <= MAX_GRAY_MAE:
                    candidates.append({
                        "path_a": previous_path, "path_b": row["path"],
                        "age_a": previous_age, "age_b": row["age"],
                        "dhash_distance": distance, "gray_mae": error,
                    })
        for key in keys:
            bands[key].append((row["path"], row["age"]))
    candidates.sort(key=lambda pair: (
        pair["gray_mae"], pair["dhash_distance"], pair["path_a"], pair["path_b"],
    ))
    return candidates


def connected_components(pairs):
    neighbors = defaultdict(set)
    for first, second in pairs:
        neighbors[first].add(second)
        neighbors[second].add(first)
    remaining = set(neighbors)
    for start in sorted(neighbors):
        if start not in remaining:
            continue
        pending = [start]
        component = set()
        while pending:
            current = pending.pop()
            if current in remaining:
                remaining.remove(current)
                component.add(current)
                pending.extend(neighbors[current])
        yield sorted(component)


def apply_review(rows, candidates, review_path):
    if review_path is None:
        return {}
    eligible = {(pair["path_a"], pair["path_b"]) for pair in candidates}
    decisions = {}
    with review_path.open(encoding="utf-8-sig", newline="") as file:
        for item in csv.DictReader(file):
            pair = (item["path_a"], item["path_b"])
            decision = item["decision"]
            if pair not in eligible or pair in decisions or decision not in {"same_photo", "retain"}:
                raise ValueError(f"Invalid review decision: {pair}, {decision}")
            decisions[pair] = decision

    by_path = {row["path"]: row for row in rows}
    confirmed = (pair for pair, decision in decisions.items() if decision == "same_photo")
    for component in connected_components(confirmed):
        kept, *duplicates = component
        if len({by_path[path]["age"] for path in component}) > 1:
            for path in component:
                by_path[path]["status"] = "conflicting_reviewed_duplicate_age"
        else:
            for path in duplicates:
                by_path[path]["status"] = "reviewed_near_duplicate"
                by_path[path]["duplicate_of"] = kept
    return decisions


def stratify(rows, seed):
    by_age = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_age[row["age"]].append(row)
    rng = random.Random(seed)
    splits = {name: [] for name in SPLITS}
    for age in sorted(by_age):
        samples = by_age[age]
        rng.shuffle(samples)
        count = len(samples)
        # A stratum with fewer than three examples cannot populate all three sets.
        holdout = max(1, (15 * count + 50) // 100) if count >= 3 else 0
        train_end = count - 2 * holdout
        splits["train"].extend(samples[:train_end])
        splits["validation"].extend(samples[train_end:train_end + holdout])
        splits["test"].extend(samples[train_end + holdout:])
    return splits


def coassign_near_duplicates(splits, candidates, decisions):
    samples = {row["path"]: row for members in splits.values() for row in members}
    assignments = {row["path"]: name for name, members in splits.items() for row in members}
    pairs = [(pair["path_a"], pair["path_b"]) for pair in candidates]
    components = [
        [path for path in photos if path in samples]
        for photos in connected_components(pair for pair in pairs if decisions.get(pair) != "retain")
    ]
    ages = Counter(row["age"] for row in samples.values())
    counts = Counter(assignments.values())
    targets = counts.copy()
    moved = 0
    group_count = 0
    for group in components:
        if len(group) < 2:
            continue
        group_count += 1
        present = Counter(assignments[path] for path in group)
        if any(ages[samples[path]["age"]] < 3 for path in group):
            destination = "train"
        else:
            destination = max(SPLITS, key=lambda name: (present[name], targets[name] - counts[name]))
        for path in group:
            original = assignments[path]
            if original != destination:
                moved += 1
                counts[original] -= 1
                counts[destination] += 1
                assignments[path] = destination

    for name in SPLITS:
        splits[name] = []
    for path in sorted(samples):
        splits[assignments[path]].append(samples[path])
    if any(len({assignments[path] for path in group}) > 1 for group in components):
        raise RuntimeError("A near-duplicate candidate crosses split boundaries")
    return {"groups_kept_together": group_count, "samples_reassigned": moved, "cross_split_pairs": 0}


def verify_split(rows, splits):
    valid = {row["path"] for row in rows if row["status"] == "ok"}
    combined = [row for members in splits.values() for row in members]
    if len(combined) != len(valid) or {row["path"] for row in combined} != valid:
        raise RuntimeError("Split manifests do not partition the valid samples")
    for field in ("path", "file_sha256", "pixel_sha256"):
        if len({row[field] for row in combined}) != len(combined):
            raise RuntimeError(f"Split manifests contain repeated {field}")
    if any(not members for members in splits.values()):
        raise RuntimeError("At least one split is empty")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_ages(output, age_counts, groups):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 4), layout="constrained")
    FigureCanvasAgg(figure)
    ages, bands = figure.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})
    ages.bar([row["age"] for row in age_counts], [row["total"] for row in age_counts],
             width=0.9, color="#255b87")
    ages.set(xlabel="Labeled age (years)", ylabel="Images", title="Cleaned UTKFace by age", xlim=(-1, 117))
    bars = bands.bar(list(groups), list(groups.values()), color="#2e7067")
    bands.bar_label(bars, fmt="{:,}", padding=3, fontsize=9)
    bands.set(xlabel="Age group (years)", title="Images by evaluation group")
    bands.set_ylim(0, max(groups.values()) * 1.14)
    for axis in (ages, bands):
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=160)


def publish(output, rows, splits, candidates, decisions, grouping, seed):
    by_age = {name: Counter(row["age"] for row in splits[name]) for name in SPLITS}
    age_counts = []
    for age in range(117):
        counts = {name: by_age[name][age] for name in SPLITS}
        age_counts.append({"age": age, **counts, "total": sum(counts.values())})
    groups = {
        name: {group: sum(row["age_group"] == group for row in members) for _, group in AGE_GROUPS}
        for name, members in splits.items()
    }
    total_groups = {group: sum(counts[group] for counts in groups.values()) for _, group in AGE_GROUPS}
    observed = [row["age"] for row in age_counts if row["total"]]
    summary = {
        "seed": seed,
        "split_strategy": "age_stratified_then_coassign_near_duplicate_candidates",
        "target_ratios": dict(zip(SPLITS, (0.70, 0.15, 0.15))),
        "rare_age_rule": "Ages with fewer than 3 examples are assigned to training.",
        "images_scanned": len(rows),
        "read_status": dict(Counter(row["read_status"] for row in rows)),
        "label_status": dict(Counter(row["label_status"] for row in rows)),
        "sample_status": dict(Counter(row["status"] for row in rows)),
        "split_counts": {name: len(members) for name, members in splits.items()},
        "age_groups": {"total": total_groups, **groups},
        "observed_age_range": [min(observed), max(observed)],
        "ages_with_fewer_than_three_images": [
            {"age": row["age"], "count": row["total"]} for row in age_counts if 0 < row["total"] < 3
        ],
        "near_duplicates": {
            "max_dhash_distance": MAX_DISTANCE, "max_gray_mae": MAX_GRAY_MAE,
            "candidates": len(candidates), "review": dict(Counter(decisions.values())),
            **grouping,
        },
        "intersection_counts": {"path": 0, "file_sha256": 0, "pixel_sha256": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temp:
        folder = Path(temp) / "result"
        folder.mkdir()
        write_csv(folder / "manifest.csv", rows, AUDIT_FIELDS)
        write_csv(folder / "exclusions.csv", (row for row in rows if row["status"] != "ok"), AUDIT_FIELDS)
        for name, members in splits.items():
            write_csv(folder / f"{name}.csv", members, SPLIT_FIELDS)
        write_csv(folder / "age_distribution.csv", age_counts, ("age", *SPLITS, "total"))
        write_csv(folder / "near_duplicate_candidates.csv", candidates, PAIR_FIELDS)
        plot_ages(folder / "age_distribution.png", age_counts, total_groups)
        summary["output_sha256"] = {
            path.name: sha256(path.read_bytes()) for path in sorted(folder.iterdir())
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        folder.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/raw/UTKFace")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-review", type=Path)
    args = parser.parse_args()
    if not args.data.is_dir():
        parser.error(f"Image directory does not exist: {args.data}")
    if args.output.exists() or args.output.resolve().is_relative_to(args.data.resolve()):
        parser.error("Output must be a new directory outside the image directory")
    rows, thumbnails = scan_images(args.data)
    remove_exact_duplicates(rows)
    candidates = find_near_duplicates(rows, thumbnails)
    decisions = apply_review(rows, candidates, args.near_review)
    splits = stratify(rows, args.seed)
    grouping = coassign_near_duplicates(splits, candidates, decisions)
    verify_split(rows, splits)
    summary = publish(args.output, rows, splits, candidates, decisions, grouping, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
