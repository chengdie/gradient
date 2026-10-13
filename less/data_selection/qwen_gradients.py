"""Per-example projected LoRA directions for Qwen LESS selection."""

import argparse
import hashlib
import json
import math
import os
from datetime import timedelta
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def partition_bounds(total, rank, world_size):
    """Disjoint contiguous ranges preserve candidate order without sampler padding."""
    if total < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid source size or distributed rank.")
    return total * rank // world_size, total * (rank + 1) // world_size


def write_manifest(path, manifest):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def merge_rank_manifests(output, world_size):
    """Merge metadata only; projected tensors remain in rank-specific shard files."""
    shared_keys = ("source", "gradient_type", "signature", "checkpoint_sha256", "checkpoint",
                   "parameter_layout", "limited", "normalization", "total_rows", "source_rows",
                   "world_size", "optimizer_sha256", "projection_dimensions", "shard_size")
    common = None
    merged_rows, merged_shards, ranges = [], None, []
    seen_files = set()
    for rank in range(world_size):
        path = output / f"rank{rank}" / "manifest.json"
        worker = json.loads(path.read_text(encoding="utf-8"))
        if not worker.get("completed") or worker.get("rank") != rank or worker.get("world_size") != world_size:
            raise ValueError(f"Incomplete or inconsistent worker manifest: {path}")
        current = {key: worker[key] for key in shared_keys}
        if common is None:
            common = current
            merged_shards = {str(dim): [] for dim in worker["projection_dimensions"]}
        elif common != current:
            raise ValueError(f"Worker source, model, optimizer or projection mismatch: {path}")
        begin, end = partition_bounds(worker["total_rows"], rank, world_size)
        if worker["row_range"] != [begin, end] or len(worker["rows"]) != end - begin:
            raise ValueError(f"Worker row range/count mismatch: {path}")
        if any(row["row"] != begin + offset or not isinstance(row["valid"], bool)
               for offset, row in enumerate(worker["rows"])):
            raise ValueError(f"Duplicate, missing or out-of-order rows: {path}")
        if set(worker["shards"]) != set(merged_shards):
            raise ValueError(f"Projection dimensions differ: {path}")
        for dimension, shards in worker["shards"].items():
            position = begin
            for shard in shards:
                file = (output / shard["path"]).resolve()
                if (shard["start"] != position or shard["count"] <= 0
                        or position + shard["count"] > end or not file.is_file()
                        or not file.is_relative_to((output / f"rank{rank}").resolve())
                        or str(file) in seen_files):
                    raise ValueError(f"Invalid worker feature shard: {file}")
                seen_files.add(str(file))
                position += shard["count"]
            if position != end:
                raise ValueError(f"Missing feature rows for dim{dimension}: {path}")
            merged_shards[dimension].extend(shards)
        merged_rows.extend(worker["rows"])
        ranges.append([begin, end])
    if common is None or len(merged_rows) != common["total_rows"] or not merged_rows:
        raise ValueError("No complete per-example gradient collection to merge.")
    if common["source"]["kind"] == "train" and sha256_file(common["source"]["path"]) != common["source"]["sha256"]:
        raise ValueError("Candidate source changed before gradient merge.")
    result = dict(common, rows=merged_rows, shards=merged_shards, completed=True,
                  execution=dict(world_size=world_size, row_ranges=ranges,
                                 sharding="contiguous ranges, no padding", per_device_batch_size=1,
                                 gradient_synchronization=False))
    write_manifest(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model_path", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train_file")
    source.add_argument("--task", choices=("mmlu", "bbh", "tydiqa"))
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gradient_type", choices=("adam", "sgd", "sign"), default="adam")
    parser.add_argument("--projection_dimensions", nargs="+", type=int, default=[8192])
    parser.add_argument("--projection_seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--shard_size", type=int, default=128)
    args = parser.parse_args()
    if min(args.projection_dimensions) <= 0 or args.max_length <= 0 or args.shard_size <= 0:
        parser.error("Dimensions, max_length and shard_size must be positive.")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max_samples must be positive.")
    if args.task and (args.gradient_type != "sgd" or not args.data_dir):
        parser.error("Target features require --gradient_type sgd and --data_dir.")
    if not (Path(args.model_path) / "adapter_config.json").is_file():
        raise ValueError("Gradient collection requires a warmup LoRA checkpoint, not the base model.")

    import torch
    import torch.distributed as distributed

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if "LOCAL_RANK" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun gradient workers require CUDA.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    if world_size > 1:
        distributed.init_process_group(backend="gloo", timeout=timedelta(hours=24))
    try:
        collect_worker(args, rank, world_size, distributed, torch)
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def collect_worker(args, rank, world_size, distributed, torch):
    from trak.projectors import BasicProjector, ProjectionType
    from less.model_utils import load_model_and_tokenizer
    from less.data_selection.get_training_dataset import encode_with_qwen_chat_format
    from less.data_selection.target_data import load_reference_records

    out = Path(args.output_dir).resolve()
    if rank == 0:
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise FileExistsError(f"Refusing nonempty output {out}")
        out.mkdir(parents=True, exist_ok=True)
        with (out / ".gradient.lock").open("x", encoding="utf-8") as lock:
            lock.write(str(os.getpid()))
    if world_size > 1:
        distributed.barrier()
    worker_out = out / f"rank{rank}"
    worker_out.mkdir(exist_ok=False)
    if args.train_file:
        input_path = Path(args.train_file).resolve()
        digest, source_rows = hashlib.sha256(), 0
        with input_path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    raise ValueError(f"Blank candidate JSONL row at {source_rows}: {input_path}")
                digest.update(line)
                source_rows += 1
        source_hash = digest.hexdigest()
        source_info = {"kind": "train", "path": str(input_path), "sha256": source_hash}
    else:
        refs = load_reference_records(args.task, args.data_dir)
        source_rows = len(refs)
        source_info = {"kind": "reference", "task": args.task, "sha256": fingerprint(refs)}
    total_rows = min(source_rows, args.max_samples) if args.max_samples is not None else source_rows
    if total_rows <= 0:
        raise ValueError("No source records to collect.")
    begin, end = partition_bounds(total_rows, rank, world_size)
    if args.train_file:
        def candidates():
            with input_path.open(encoding="utf-8") as stream:
                for index, line in enumerate(stream):
                    if index >= end:
                        break
                    if index >= begin:
                        yield index, json.loads(line)
        records = candidates()
    else:
        records = ((index, refs[index]) for index in range(begin, end))
    print(f"[rank {rank}/{world_size}] {args.device}: rows [{begin}, {end}) of {total_rows}; no gradient synchronization", flush=True)
    torch.manual_seed(args.projection_seed)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, base_model_path=args.base_model_path, dtype=args.dtype,
        device=args.device, trainable=True)
    model.eval()
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not parameters or any("lora" not in name.lower() for name, _ in parameters):
        raise ValueError("Only LoRA parameters may require gradients during feature extraction.")
    layout = [{"name": name, "shape": list(p.shape)} for name, p in parameters]
    dimension = sum(p.numel() for _, p in parameters)
    means = squares = None
    optimizer_sha256 = None
    if args.gradient_type == "adam":
        checkpoint = Path(args.model_path)
        metadata = json.loads((checkpoint / "selection_meta.json").read_text())
        if (metadata["observed_steps"] <= 0 or not math.isfinite(metadata["mean_learning_rate"])
                or metadata["mean_learning_rate"] <= 0 or not metadata["param_groups"]):
            raise ValueError("Invalid warmup step count or mean learning rate.")
        for group in metadata["param_groups"]:
            if list(group["betas"]) != [0.9, 0.999] or group["weight_decay"] != 0 or group["eps"] != 1e-8:
                raise ValueError("Adam LESS requires betas=(0.9,0.999), weight_decay=0 and eps=1e-8.")
        optimizer_sha256 = sha256_file(checkpoint / "optimizer_named.pt")
        state = torch.load(checkpoint / "optimizer_named.pt", map_location="cpu", weights_only=True)
        if state["parameter_names"] != [name for name, _ in parameters]:
            raise ValueError("Named optimizer parameter order differs from loaded LoRA parameters.")
        for name, parameter in parameters:
            for key in ("exp_avg", "exp_avg_sq"):
                moment = state["state"][name][key]
                if moment.shape != parameter.shape or not torch.isfinite(moment).all():
                    raise ValueError(f"Optimizer moment shape/value mismatch: {name}")
                if key == "exp_avg_sq" and torch.any(moment < 0):
                    raise ValueError(f"Negative second moment: {name}")
        means = torch.cat([state["state"][name]["exp_avg"].flatten() for name, _ in parameters]).to(args.device, torch.float32)
        squares = torch.cat([state["state"][name]["exp_avg_sq"].flatten() for name, _ in parameters]).to(args.device, torch.float32)
        del state
    dimensions = sorted(set(args.projection_dimensions))
    projectors = {dim: BasicProjector(grad_dim=dimension, proj_dim=dim, seed=args.projection_seed,
                                    proj_type=ProjectionType.rademacher, device=torch.device(args.device),
                                    dtype=torch.float32, block_size=min(dim, 16)) for dim in dimensions}
    signature = dict(parameters=fingerprint(layout), projection="trak-basic-rademacher-fp32-block16",
                     projection_seed=args.projection_seed, max_length=args.max_length,
                     encoding="qwen-native-assistant-v1", dtype=args.dtype,
                     tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                     template_sha256=fingerprint(tokenizer.get_chat_template()),
                     base_model_path=str(Path(args.base_model_path).resolve()))
    checkpoint_files = sorted(Path(args.model_path).glob("adapter_model.*"))
    if not checkpoint_files:
        raise ValueError("No adapter weights found in the warmup checkpoint.")
    checkpoint_files.append(Path(args.model_path) / "adapter_config.json")
    signature["base_config_sha256"] = sha256_file(Path(args.base_model_path) / "config.json")
    checkpoint_sha256 = fingerprint({path.name: sha256_file(path) for path in checkpoint_files})
    common = dict(source=source_info, gradient_type=args.gradient_type, signature=signature,
                  checkpoint_sha256=checkpoint_sha256, checkpoint=str(Path(args.model_path).resolve()),
                  parameter_layout=layout, optimizer_sha256=optimizer_sha256,
                  limited=args.max_samples is not None, normalization="none", total_rows=total_rows,
                  source_rows=source_rows, world_size=world_size, projection_dimensions=dimensions,
                  shard_size=args.shard_size)
    if world_size > 1:
        worker_signatures = [None] * world_size
        distributed.all_gather_object(worker_signatures, fingerprint(common))
        if len(set(worker_signatures)) != 1:
            raise ValueError("Gradient workers loaded different source, model, optimizer, tokenizer or projection settings.")
    buffers, shards = {dim: [] for dim in dimensions}, {str(dim): [] for dim in dimensions}
    rows = []
    def flush():
        if not buffers[dimensions[0]]:
            return
        start = begin + len(rows) - len(buffers[dimensions[0]])
        for dim in dimensions:
            directory = worker_out / f"dim{dim}"
            directory.mkdir(exist_ok=True)
            file = directory / f"grads-{start:08d}.pt"
            torch.save(torch.stack(buffers[dim]), file)
            shards[str(dim)].append({"path": str(file.relative_to(out)), "start": start,
                                     "count": len(buffers[dim]), "sha256": sha256_file(file)})
            buffers[dim].clear()
    for index, record in records:
        if args.task:
            reference_text = tokenizer.apply_chat_template(
                record["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
            if len(tokenizer.encode(reference_text, add_special_tokens=False)) > args.max_length:
                raise ValueError(f"Reference {record['id']} exceeds --max_length; increase it rather than truncate the answer.")
        encoded = encode_with_qwen_chat_format(record, tokenizer, args.max_length)
        valid = any(token != -100 for token in encoded["labels"][1:])
        if args.task and not valid:
            raise ValueError(f"Reference answer truncated: {record['id']}; increase --max_length.")
        model.zero_grad(set_to_none=True)
        if valid:
            batch = {name: torch.tensor([value], dtype=torch.long, device=args.device) for name, value in encoded.items()}
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at source row {index}")
            loss.backward()
            gradient = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().float()
                                  for _, p in parameters])
            if args.gradient_type == "adam":
                gradient = (0.9 * means + 0.1 * gradient) / (0.999 * squares + 0.001 * gradient.square() + 1e-8).sqrt()
            elif args.gradient_type == "sign":
                gradient = gradient.sign()
            if not torch.isfinite(gradient).all():
                raise ValueError(f"Non-finite gradient at source row {index}")
            for dim, projector in projectors.items():
                feature = projector.project(gradient.unsqueeze(0), model_id=0).squeeze(0).detach().float().cpu()
                if not torch.isfinite(feature).all():
                    raise ValueError(f"Non-finite projected gradient at source row {index}")
                buffers[dim].append(feature)
            del gradient, loss, batch
        else:
            for dim in dimensions:
                buffers[dim].append(torch.zeros(dim, dtype=torch.float32))
        rows.append({"row": index, "id": str(record.get("id", index)), "subtask": record.get("subtask"), "valid": valid})
        model.zero_grad(set_to_none=True)
        if len(buffers[dimensions[0]]) >= args.shard_size:
            flush()
        if len(rows) == 1 or len(rows) % 100 == 0:
            print(f"[rank {rank}] collected {len(rows)}/{end - begin} examples", flush=True)
    flush()
    if len(rows) != end - begin:
        raise ValueError(f"Worker {rank} collected {len(rows)} rows; expected {end - begin}.")
    if args.train_file and sha256_file(input_path) != source_hash:
        raise ValueError("Candidate source changed during gradient collection.")
    if fingerprint({path.name: sha256_file(path) for path in checkpoint_files}) != checkpoint_sha256:
        raise ValueError("Warmup adapter weights changed during gradient collection.")
    if optimizer_sha256 is not None and sha256_file(Path(args.model_path) / "optimizer_named.pt") != optimizer_sha256:
        raise ValueError("Warmup optimizer state changed during gradient collection.")
    write_manifest(worker_out / "manifest.json",
                   dict(common, completed=True, rank=rank, row_range=[begin, end], rows=rows, shards=shards))
    if world_size > 1:
        distributed.barrier()
    if rank == 0:
        manifest = merge_rank_manifests(out, world_size)
        print(f"Completed {len(manifest['rows'])} per-example gradients using {world_size} worker(s): {out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
