"""Reproducible uniform sampling from the same candidate pool as LESS."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def inspect_source(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                raise ValueError(f"Blank JSONL row in {path} at row {count}")
            count += 1
    return {"path": str(path), "sha256": digest.hexdigest(), "rows": count}


def select_random(train_files, output_dir, seed=3, percentage=0.05, reference_file=None):
    sources = [inspect_source(path) for path in train_files]
    if len({source["path"] for source in sources}) != len(sources):
        raise ValueError("Duplicate training sources are not allowed.")
    total = sum(source["rows"] for source in sources)
    if not 0 < percentage <= 1:
        raise ValueError("percentage must be in (0, 1].")
    reference = inspect_source(reference_file) if reference_file else None
    count = reference["rows"] if reference else math.floor(total * percentage)
    if not 0 < count <= total:
        raise ValueError(f"Invalid selection size {count} for {total} candidate rows.")
    selected = set(random.Random(seed).sample(range(total), count))
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    indices = []
    row_index = 0
    temporary = output / "selected.jsonl.tmp"
    with temporary.open("wb") as destination:
        for source_index, source in enumerate(sources):
            digest = hashlib.sha256()
            with Path(source["path"]).open("rb") as stream:
                for source_row, line in enumerate(stream):
                    digest.update(line)
                    if row_index in selected:
                        json.loads(line)
                        destination.write(line.rstrip(b"\r\n") + b"\n")
                        indices.append({"source": source_index, "row": source_row})
                    row_index += 1
            if digest.hexdigest() != source["sha256"]:
                raise ValueError(f"Candidate source changed during selection: {source['path']}")
    temporary.replace(output / "selected.jsonl")
    (output / "indices.json").write_text(json.dumps(indices), encoding="utf-8")
    manifest = {
        "method": "random", "seed": seed, "candidate_count": total,
        "selected_count": count, "percentage": count / total,
        "budget": "reference_file_rows" if reference else "percentage_of_pool",
        "reference_file": reference, "sources": sources,
        "selected_file": inspect_source(output / "selected.jsonl"),
        "row_order": "source_order", "completed": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_files", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--percentage", type=float, default=0.05)
    parser.add_argument("--reference_file", help="Match the exact row budget of an existing LESS subset.")
    args = parser.parse_args()
    manifest = select_random(**vars(args))
    print(json.dumps({key: manifest[key] for key in ("method", "seed", "candidate_count", "selected_count")}))


if __name__ == "__main__":
    main()
