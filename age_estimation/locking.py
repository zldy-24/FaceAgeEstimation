"""Verify prepared data against the separately archived dataset lock."""

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = PROJECT / "docs/phase2/dataset-lock.json"


def verify_dataset_lock(splits, lock=DEFAULT_LOCK):
    splits, lock = Path(splits).resolve(), Path(lock).resolve()
    if lock.is_relative_to(splits):
        raise ValueError("The dataset lock must be stored outside the generated split directory")
    archived = json.loads(lock.read_text(encoding="utf-8-sig"))
    for name, expected in archived["output_sha256"].items():
        with (splits / name).open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Dataset lock checksum mismatch: {name}")
    generated = json.loads((splits / "summary.json").read_text(encoding="utf-8-sig"))
    if generated != archived:
        raise ValueError("Generated summary does not match the archived dataset lock")
    # Formatting changes such as Git's Windows line endings do not change the lock ID.
    canonical = json.dumps(archived, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
