"""Select a global top-k subset from multi-checkpoint Qwen gradient manifests."""

import argparse
import json
import math
from pathlib import Path

from less.data_selection.qwen_gradients import sha256_file


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("completed"):
        raise ValueError(f"Incomplete gradient manifest: {path}")
    manifest["directory"] = path.parent
    return manifest


def batches(manifest, dimension, torch):
    start = 0
    for shard in manifest["shards"][str(dimension)]:
        path = manifest["directory"] / shard["path"]
        if shard["start"] != start or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"Invalid or modified gradient shard: {path}")
        values = torch.load(path, map_location="cpu", weights_only=True).float()
        if list(values.shape) != [shard["count"], dimension] or not torch.isfinite(values).all():
            raise ValueError(f"Invalid gradient tensor: {path}")
        yield start, values
        start += shard["count"]
    if start != len(manifest["rows"]):
        raise ValueError("Gradient row count does not match manifest.")


def select(spec, output_dir):
    import torch
    import torch.nn.functional as functional

    dimension = int(spec["dimension"])
    normalize = spec.get("normalize", True)
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a JSON boolean.")
    checkpoints = spec["checkpoints"]
    weights = [float(checkpoint["weight"]) for checkpoint in checkpoints]
    if not weights or not all(math.isfinite(weight) and weight >= 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Checkpoint weights must be finite, nonnegative, with a positive sum.")
    weights = [weight / sum(weights) for weight in weights]
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite {output}")
    sources, canonical_rows, accumulated, valid_masks = [], [], [], []
    reference_rows = reference_signature = reference_source = None
    limited = False
    gradient_type = None
    for checkpoint_index, (checkpoint, weight) in enumerate(zip(checkpoints, weights)):
        optimizer_signature = None
        target = read_manifest(checkpoint["validation_manifest"])
        if target["gradient_type"] != "sgd" or target["source"].get("task") != spec["task"]:
            raise ValueError("Validation gradients must be SGD for the specified task.")
        if any(not row["valid"] for row in target["rows"]):
            raise ValueError("Invalid validation gradients are not allowed.")
        if checkpoint_index == 0:
            reference_rows, reference_signature, reference_source = target["rows"], target["signature"], target["source"]
        elif target["rows"] != reference_rows or target["signature"] != reference_signature or target["source"] != reference_source:
            raise ValueError("Target rows, source contents or projection signatures changed across checkpoints.")
        subtasks = sorted({row["subtask"] for row in target["rows"]})
        target_features = torch.cat([values for _, values in batches(target, dimension, torch)])
        if normalize:
            if torch.any(target_features.norm(dim=1) == 0):
                raise ValueError("Zero-norm validation feature cannot define a cosine direction.")
            target_features = functional.normalize(target_features, dim=1)
        target_means = torch.stack([target_features[[row["subtask"] == sub for row in target["rows"]]].mean(0)
                                    for sub in subtasks])
        training = checkpoint["training_manifests"]
        if checkpoint_index and len(training) != len(sources):
            raise ValueError("Candidate source count changed across checkpoints.")
        for source_index, path in enumerate(training):
            candidate = read_manifest(path)
            if candidate["source"]["kind"] != "train":
                raise ValueError("Training manifest is not a candidate source.")
            if gradient_type is None:
                gradient_type = candidate["gradient_type"]
            elif gradient_type != candidate["gradient_type"]:
                raise ValueError("Do not mix Adam, SGD or sign features across sources/checkpoints.")
            if gradient_type == "adam":
                candidate_optimizer = candidate.get("optimizer_sha256")
                if not candidate_optimizer:
                    raise ValueError("Adam gradients require an optimizer-state fingerprint; regenerate legacy features.")
                if optimizer_signature is None:
                    optimizer_signature = candidate_optimizer
                elif optimizer_signature != candidate_optimizer:
                    raise ValueError("Candidate sources used different Adam states at the same checkpoint.")
            if (candidate["signature"] != target["signature"]
                    or candidate["checkpoint"] != target["checkpoint"]
                    or candidate["checkpoint_sha256"] != target["checkpoint_sha256"]):
                raise ValueError("Candidate/reference model, parameter layout or projection differs.")
            limited |= candidate["limited"] or target["limited"]
            if checkpoint_index == 0:
                sources.append(candidate["source"])
                canonical_rows.append(candidate["rows"])
                accumulated.append(torch.zeros(len(candidate["rows"]), len(subtasks)))
                valid_masks.append(torch.tensor([row["valid"] for row in candidate["rows"]]))
            else:
                if candidate["source"] != sources[source_index] or candidate["rows"] != canonical_rows[source_index]:
                    raise ValueError("Candidate row identities or validity changed across checkpoints.")
            if sha256_file(candidate["source"]["path"]) != candidate["source"]["sha256"]:
                raise ValueError("Candidate JSONL changed after gradient collection.")
            for start, features in batches(candidate, dimension, torch):
                if normalize:
                    valid_masks[source_index][start:start + len(features)] &= features.norm(dim=1) > 0
                    features = functional.normalize(features, dim=1)
                accumulated[source_index][start:start + len(features)] += weight * (features @ target_means.T)
    if len({source["path"] for source in sources}) != len(sources):
        raise ValueError("Duplicate candidate sources.")
    scores = []
    identities = []
    for index, values in enumerate(accumulated):
        values = values.max(dim=1).values
        values[~valid_masks[index]] = -torch.inf
        scores.append(values)
        identities.extend((index, row["row"]) for row in canonical_rows[index])
    scores = torch.cat(scores)
    if "count" in spec:
        count = int(spec["count"])
    else:
        percentage = float(spec["percentage"])
        if not 0 < percentage <= 1:
            raise ValueError("percentage must be in (0, 1].")
        count = math.floor(len(scores) * percentage)
    if not 0 < count <= int(torch.isfinite(scores).sum()):
        raise ValueError("Selection count exceeds the number of valid finite scores.")
    indices = torch.argsort(scores, descending=True, stable=True)[:count].tolist()
    selected_set = {identities[index] for index in indices}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "selected.jsonl").open("x", encoding="utf-8") as destination:
        for source_index, source in enumerate(sources):
            with Path(source["path"]).open(encoding="utf-8") as stream:
                for row_index, line in enumerate(stream):
                    if (source_index, row_index) in selected_set:
                        destination.write(line.rstrip("\r\n") + "\n")
            if sha256_file(source["path"]) != source["sha256"]:
                raise ValueError("Candidate JSONL changed while writing the selected subset.")
    ranking = [{"source": identities[index][0], "row": identities[index][1], "score": float(scores[index])} for index in indices]
    (output / "indices.json").write_text(json.dumps(ranking), encoding="utf-8")
    manifest = dict(completed=True, task=spec["task"], method="less", selected_count=count,
                    candidate_count=len(scores), valid_count=int(torch.isfinite(scores).sum()),
                    limited=limited, spec=spec, sources=sources,
                    selected_sha256=sha256_file(output / "selected.jsonl"),
                    row_order="source_order", aggregation="checkpoint-weighted mean per subtask then max across subtasks")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    manifest = select(json.loads(Path(args.spec).read_text()), args.output_dir)
    print(json.dumps({key: manifest[key] for key in ("task", "candidate_count", "selected_count", "limited")}))


if __name__ == "__main__":
    main()
