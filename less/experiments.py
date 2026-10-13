"""Plan or run Qwen baselines, task-transfer experiments, and LESS ablations.

Default operation only prints a plan. --execute is required for training/evaluation.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

from less.data_selection.random_subset import inspect_source
from less.model_utils import QWEN_PROJECTIONS

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_semantic_doc/user_awardcheng/qwen3.5-4b"
PREFIXES = {"mmlu": "mmlu-chat", "bbh": "bbh-icl", "tydiqa": "tydiqa"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def source_files(data_dir):
    return [data_dir / "train" / "processed" / name / f"{name}_data.jsonl"
            for name in ("flan_v2", "cot", "dolly", "oasst1")]


def completed(output, marker):
    path = output / marker
    if not path.is_file():
        return False
    if marker in {"metrics.json", "manifest.json"}:
        return bool(json.loads(path.read_text()).get("completed"))
    return (output / "adapter_config.json").is_file() and (
        (output / "adapter_model.safetensors").is_file() or (output / "adapter_model.bin").is_file())


def step(args, command, output, marker, context):
    command = [str(item) for item in command]
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    record = dict(command=command, context=context, output=str(output), marker=marker)
    signature = fingerprint(record)
    sidecar = output.parent / (output.name + ".run.json")
    if sidecar.exists():
        saved = json.loads(sidecar.read_text())
        if saved.get("fingerprint") != signature:
            raise ValueError(f"Run configuration changed: {sidecar}")
        if completed(output, marker):
            print(f"Reusing completed stage: {output}", flush=True)
            return
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        hint = "archive the failed stage directory or choose a new --work-dir/--stage-output" if args.action == "single" else "choose a new --run-tag or archive it manually"
        raise FileExistsError(f"Incomplete/nonempty stage {output}; {hint}. Completed upstream stages need not be rerun.")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(dict(fingerprint=signature, **record), indent=2), encoding="utf-8")
    environment = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    if args.action == "single":
        log_path = output.parent / (output.name + ".console.log")
        print(f"Full stage log: {log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\nCOMMAND: " + shlex.join(command) + "\n")
            log.flush()
            with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                return_code = process.wait()
            if return_code:
                raise RuntimeError(f"Only this stage failed (exit {return_code}). Full traceback: {log_path}")
    else:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    if not completed(output, marker):
        raise RuntimeError(f"Stage ended without a valid completion artifact: {output / marker}")


def launch_module(module, nproc):
    if nproc == 1:
        return [sys.executable, "-m", module]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={nproc}", "-m", module]


def training_command(args, files, output, train_seed, selection_seed, warmup=False):
    argv = launch_module("less.train.train", args.train_nproc) + ["--model_name_or_path", args.model_path,
            "--local_files_only", "True", "--torch_dtype", "bfloat16", "--attn_implementation", "sdpa",
            "--use_fast_tokenizer", "True", "--train_files", *map(str, files), "--chat_format", "qwen",
            "--percentage", str(args.warmup_percentage if warmup else 1.0),
            "--sample_data_seed", str(selection_seed), "--preprocessing_num_workers", str(args.preprocessing_workers),
            "--max_seq_length", str(args.max_length), "--output_dir", str(output),
            "--do_train", "True", "--eval_strategy", "no",
            "--num_train_epochs", str(args.warmup_epochs if warmup else args.epochs),
            "--max_steps", str(args.warmup_max_steps if warmup else args.max_steps),
            "--per_device_train_batch_size", str(args.train_batch_size), "--gradient_accumulation_steps", str(args.gradient_accumulation),
            "--gradient_checkpointing", str(args.gradient_checkpointing), "--gradient_checkpointing_kwargs", '{"use_reentrant":false}',
            "--ddp_find_unused_parameters", "False", "--ddp_broadcast_buffers", "False",
            "--learning_rate", str(args.learning_rate), "--lr_scheduler_type", "linear", "--warmup_steps", "0.03",
            "--weight_decay", "0.0", "--optim", "adamw_torch", "--bf16", "True", "--fp16", "False",
            "--lora", "True", "--lora_r", str(args.lora_r), "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", "0.1", "--lora_target_modules", *QWEN_PROJECTIONS,
            "--save_strategy", "epoch", "--logging_steps", "1", "--report_to", "none", "--seed", str(train_seed),
            "--dataloader_num_workers", str(args.dataloader_workers), "--dataloader_pin_memory", "True",
            "--export_optimizer_state", str(warmup)]
    if not warmup:
        argv.extend(["--save_total_limit", "2"])
    return argv


def evaluate(args, model, task, output, context):
    argv = launch_module("evaluation.eval.qwen", args.eval_nproc) + ["--model_path", str(model),
            "--base_model_path", args.model_path, "--task", task, "--data_dir", str(args.data_dir),
            "--output_dir", str(output), "--seed", str(args.eval_seed),
            "--max_length", str(args.eval_max_length), "--max_new_tokens", str(args.eval_max_new_tokens),
            "--batch_size", str(args.eval_batch_size), "--device", args.device, "--dtype", "bfloat16"]
    if args.eval_max_per_subtask is not None:
        argv.extend(["--max_per_subtask", str(args.eval_max_per_subtask)])
    step(args, argv, output, "metrics.json", context)


def less_variants(args):
    default = dict(gradient_type="adam", dimension=args.projection_dim, normalize=True, checkpoints="all")
    variants = [("default", default)]
    if args.suite in {"ablations", "all"}:
        variants += [(kind, dict(default, gradient_type=kind)) for kind in ("sgd", "sign")]
        variants += [("unnormalized", dict(default, normalize=False)), ("first-checkpoint", dict(default, checkpoints="first"))]
        variants += [(f"dim{dim}", dict(default, dimension=dim)) for dim in (1024, 2048, 4096, 8192) if dim != args.projection_dim]
    return variants


def prepare_less(args, sources, task, selection_seed, variant_name, variant, count, shared_context):
    warmup_config = dict(shared_context, selection_seed=selection_seed,
                         epochs=args.warmup_epochs, percentage=args.warmup_percentage,
                         max_steps=args.warmup_max_steps, seed=args.warmup_train_seed)
    warmup_root = args.output_dir / "warmup" / f"s{selection_seed}-{fingerprint(warmup_config)}"
    step(args, training_command(args, sources, warmup_root, args.warmup_train_seed, selection_seed, True),
         warmup_root, "train_results.json", warmup_config)
    if args.execute:
        checkpoints = sorted(warmup_root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if not checkpoints:
            raise ValueError(f"Warmup produced no checkpoints: {warmup_root}")
        for checkpoint in checkpoints:
            if not all((checkpoint / name).is_file() for name in ("optimizer_named.pt", "selection_meta.json")):
                raise ValueError(f"Warmup selection state missing in {checkpoint}")
    else:
        checkpoints = [warmup_root / "checkpoint-<actual-epoch-step>"]
    if variant["checkpoints"] == "first":
        checkpoints = checkpoints[:1]
    dimensions = sorted({item["dimension"] for _, item in less_variants(args)})
    selection_config = dict(warmup=warmup_config, task=task, variant=variant, count=count,
                            gradient_max_length=args.gradient_max_length, projection_seed=args.projection_seed,
                            gradient_nproc=args.gradient_nproc, gradient_sharding="contiguous-v1",
                            max_samples=args.gradient_max_samples)
    selection_root = args.output_dir / "selected" / f"{task}-s{selection_seed}-{variant_name}-{fingerprint(selection_config)}"
    if args.execute and completed(selection_root, "manifest.json"):
        saved = json.loads((selection_root / "experiment.json").read_text())
        if saved != selection_config:
            raise ValueError("Existing selection configuration does not match.")
        return selection_root / "selected.jsonl"
    specs = []
    for checkpoint in checkpoints:
        grad_config = dict(warmup=warmup_config, checkpoint=str(checkpoint), dimensions=dimensions,
                           max_length=args.gradient_max_length, projection_seed=args.projection_seed,
                           gradient_nproc=args.gradient_nproc, sharding="contiguous-v1",
                           max_samples=args.gradient_max_samples)
        grad_root = args.output_dir / "grads" / fingerprint(grad_config)
        common = launch_module("less.data_selection.qwen_gradients", args.gradient_nproc) + ["--model_path", str(checkpoint),
                  "--base_model_path", args.model_path, "--projection_dimensions", *map(str, dimensions),
                  "--max_length", str(args.gradient_max_length), "--projection_seed", str(args.projection_seed),
                  "--device", args.device]
        training_manifests = []
        for file in sources:
            output = grad_root / variant["gradient_type"] / file.parent.name
            command = common + ["--train_file", str(file), "--gradient_type", variant["gradient_type"], "--output_dir", str(output)]
            if args.gradient_max_samples is not None:
                command += ["--max_samples", str(args.gradient_max_samples)]
            step(args, command, output, "manifest.json", grad_config)
            training_manifests.append(str(output / "manifest.json"))
        target = grad_root / "reference" / task
        step(args, common + ["--task", task, "--data_dir", str(args.data_dir), "--gradient_type", "sgd", "--output_dir", str(target)],
             target, "manifest.json", grad_config)
        weight = json.loads((checkpoint / "selection_meta.json").read_text())["mean_learning_rate"] if args.execute else 1.0
        specs.append(dict(weight=weight, training_manifests=training_manifests, validation_manifest=str(target / "manifest.json")))
    spec = dict(task=task, dimension=variant["dimension"], normalize=variant["normalize"], count=count, checkpoints=specs)
    spec_file = selection_root.parent / (selection_root.name + ".spec.json")
    if args.execute:
        spec_file.parent.mkdir(parents=True, exist_ok=True)
        spec_file.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    else:
        print(f"Selection spec (actual checkpoint weights resolved at runtime): {json.dumps(spec)}")
    step(args, [sys.executable, "-m", "less.data_selection.qwen_select", "--spec", str(spec_file), "--output_dir", str(selection_root)],
         selection_root, "manifest.json", selection_config)
    if args.execute:
        (selection_root / "experiment.json").write_text(json.dumps(selection_config, indent=2), encoding="utf-8")
    return selection_root / "selected.jsonl"


def run(args):
    if args.suite == "all":
        args.tasks, args.methods, args.selection_seeds = list(PREFIXES), ["base", "random", "transfer", "less"], [3, 6, 9]
    elif args.suite == "ablations":
        args.methods = ["less"]
    sources = source_files(args.data_dir)
    source_info = [inspect_source(path) for path in sources]
    total = sum(source["rows"] for source in source_info)
    config_path = Path(args.model_path) / "config.json"
    if args.execute and not config_path.is_file():
        raise FileNotFoundError(config_path)
    if args.preflight:
        from less.data_selection.target_data import load_eval_data
        for task in args.tasks:
            records, protocol = load_eval_data(task, args.data_dir, args.eval_seed, args.eval_max_per_subtask)
            print(json.dumps(dict(task=task, records=len(records), protocol=protocol)))
        print(json.dumps(dict(model_config_exists=config_path.is_file(), candidate_count=total,
                              sources=source_info, gpu_checked=False)))
        return
    hyperparameters = dict(epochs=args.epochs, max_length=args.max_length, learning_rate=args.learning_rate,
                           lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                           train_nproc=args.train_nproc, train_batch_size=args.train_batch_size,
                           gradient_accumulation=args.gradient_accumulation,
                           effective_batch_size=args.train_nproc * args.train_batch_size * args.gradient_accumulation,
                           gradient_checkpointing=args.gradient_checkpointing,
                           preprocessing_workers=args.preprocessing_workers, dataloader_workers=args.dataloader_workers,
                           max_steps=args.max_steps)
    eval_sources = {str(path.relative_to(args.data_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted((args.data_dir / "eval").rglob("*"))
                    if path.is_file() and path.suffix in {".csv", ".json", ".txt"}}
    shared = dict(model_path=args.model_path, model_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else "unavailable-dry-run",
                  sources=source_info, eval_sources=eval_sources,
                  hyperparameters=hyperparameters, run_tag=args.run_tag)
    evaluation = dict(seed=args.eval_seed, max_length=args.eval_max_length,
                      max_new_tokens=args.eval_max_new_tokens, max_per_subtask=args.eval_max_per_subtask,
                      nproc=args.eval_nproc, batch_size=args.eval_batch_size)
    for task in args.tasks:
        if "base" in args.methods and args.stage != "train":
            context = dict(shared, method="base", task=task, evaluation=evaluation)
            evaluate(args, args.model_path, task, args.output_dir / "runs" / f"{task}-base-{fingerprint(context)}" / "eval", context)
        for selection_seed in args.selection_seeds:
            reference = args.data_dir / "selected_data" / f"{PREFIXES[task]}_adam_sim_trainp0.05_seed{selection_seed}_p0.05.jsonl"
            reference_info = inspect_source(reference) if reference.exists() else None
            if "transfer" in args.methods:
                if reference_info is None:
                    raise FileNotFoundError(reference)
                if args.percentage != 0.05:
                    raise ValueError("Preselected transfer files only represent 5%; omit transfer for other percentages.")
            count = reference_info["rows"] if reference_info and args.percentage == 0.05 else math.floor(total * args.percentage)
            if args.gradient_max_samples is not None and "less" in args.methods:
                raise ValueError("Use qwen_gradients directly for limited smoke features; benchmark matrix requires the full candidate pool.")
            for method in (method for method in args.methods if method != "base"):
                variants = less_variants(args) if method == "less" else [("default", {})]
                for variant_name, variant in variants:
                    selected_file = reference
                    if method == "random":
                        random_config = dict(sources=source_info, count=count, selection_seed=selection_seed, reference=reference_info)
                        random_root = args.output_dir / "selected" / f"random-s{selection_seed}-{fingerprint(random_config)}"
                        selected_file = random_root / "selected.jsonl"
                        if args.stage != "eval":
                            argv = [sys.executable, "-m", "less.data_selection.random_subset", "--train_files", *map(str, sources),
                                    "--output_dir", str(random_root), "--seed", str(selection_seed), "--percentage", str(args.percentage)]
                            if reference_info and args.percentage == 0.05:
                                argv += ["--reference_file", str(reference)]
                            step(args, argv, random_root, "manifest.json", random_config)
                    if method == "less" and args.stage != "eval":
                        selected_file = prepare_less(args, sources, task, selection_seed, variant_name, variant, count, shared)
                    for train_seed in args.training_seeds:
                        context = dict(shared, evaluation=evaluation, task=task, method=method, variant=variant,
                                       selection_seed=selection_seed, train_seed=train_seed, count=count,
                                       reference=reference_info, warmup_epochs=args.warmup_epochs,
                                       warmup_percentage=args.warmup_percentage, warmup_train_seed=args.warmup_train_seed,
                                       warmup_max_steps=args.warmup_max_steps, projection_seed=args.projection_seed,
                                       gradient_max_length=args.gradient_max_length)
                        if method == "less":
                            context.update(gradient_nproc=args.gradient_nproc, gradient_sharding="contiguous-v1")
                        name = f"{task}-{method}-{variant_name}-s{selection_seed}-t{train_seed}-{fingerprint(context)}"
                        directory = args.output_dir / "runs" / name
                        if args.stage != "eval":
                            step(args, training_command(args, [selected_file], directory / "model", train_seed, selection_seed),
                                 directory / "model", "train_results.json", context)
                        if args.stage != "train":
                            if args.execute and not completed(directory / "model", "train_results.json"):
                                raise FileNotFoundError(f"Train this configuration first: {directory / 'model'}")
                            evaluate(args, directory / "model", task, directory / "eval", context)


def single_stage(args):
    """Run exactly one explicitly requested unit; never start its prerequisites."""
    if len(args.tasks) != 1 or len(args.selection_seeds) != 1 or len(args.training_seeds) != 1:
        raise ValueError("single requires one task, one selection seed, and one training seed.")
    if args.suite != "main" or args.preflight:
        raise ValueError("single does not expand suites or run preflight; select one --only stage.")
    task, selection_seed, train_seed = args.tasks[0], args.selection_seeds[0], args.training_seeds[0]
    root = args.work_dir.resolve()
    stage = args.only
    if not stage:
        raise ValueError("single requires --only before|warmup|train-grad|reference-grad|select|finetune|after")
    context = dict(mode="single", only=stage, task=task)
    output = None
    marker = "manifest.json"
    if stage in {"before", "after"}:
        if stage == "after" and args.adapter_path is None:
            raise ValueError("--only after requires --adapter-path; training is never launched automatically.")
        model = args.model_path if stage == "before" else str(args.adapter_path.resolve())
        if stage == "after" and args.execute and not (Path(model) / "adapter_config.json").is_file():
            raise FileNotFoundError(f"No saved adapter at {model}")
        output = args.stage_output or root / "eval" / f"{task}-{stage}"
        context.update(eval_seed=args.eval_seed)
        evaluate(args, model, task, output.resolve(), context)
        return
    if stage == "warmup":
        files = source_files(args.data_dir)
        context.update(sources=[inspect_source(path) for path in files], selection_seed=selection_seed)
        output = args.stage_output or root / "warmup" / f"s{selection_seed}"
        command = training_command(args, files, output.resolve(), args.warmup_train_seed, selection_seed, True)
        marker = "train_results.json"
    elif stage in {"train-grad", "reference-grad"}:
        if args.checkpoint_path is None:
            raise ValueError("Gradient stages require --checkpoint-path pointing to an existing warmup checkpoint.")
        checkpoint = args.checkpoint_path.resolve()
        if args.execute and not (checkpoint / "adapter_config.json").is_file():
            raise FileNotFoundError(checkpoint / "adapter_config.json")
        context.update(checkpoint=str(checkpoint))
        if (checkpoint / "adapter_config.json").is_file():
            context["adapter_config_sha256"] = hashlib.sha256((checkpoint / "adapter_config.json").read_bytes()).hexdigest()
        command = launch_module("less.data_selection.qwen_gradients", args.gradient_nproc) + [
            "--model_path", str(checkpoint), "--base_model_path", args.model_path,
            "--projection_dimensions", str(args.projection_dim), "--projection_seed", str(args.projection_seed),
            "--max_length", str(args.gradient_max_length), "--device", args.device]
        if stage == "train-grad":
            if args.source is None:
                raise ValueError("train-grad requires one --source: flan_v2, cot, dolly, or oasst1.")
            file = args.data_dir / "train" / "processed" / args.source / f"{args.source}_data.jsonl"
            context["source"] = inspect_source(file)
            output = args.stage_output or root / "grads" / checkpoint.name / args.gradient_kind / args.source
            command += ["--train_file", str(file), "--gradient_type", args.gradient_kind]
            if args.gradient_max_samples is not None:
                raise ValueError("single stages retain full source budgets; use qwen_gradients directly for a limited diagnostic run.")
        else:
            output = args.stage_output or root / "grads" / checkpoint.name / "reference" / task
            command += ["--task", task, "--data_dir", str(args.data_dir), "--gradient_type", "sgd"]
        command += ["--output_dir", str(output.resolve())]
    elif stage == "select":
        if not args.checkpoint_paths:
            raise ValueError("select requires --checkpoint-paths listing the warmup checkpoints to combine.")
        checkpoints = [path.resolve() for path in args.checkpoint_paths]
        if len(set(checkpoints)) != len(checkpoints) or len({path.name for path in checkpoints}) != len(checkpoints):
            raise ValueError("Checkpoint paths must be distinct and from one warmup trajectory.")
        if len({path.parent for path in checkpoints}) != 1:
            raise ValueError("All selected checkpoints must belong to the same warmup directory.")
        spec = dict(task=task, dimension=args.projection_dim, normalize=not args.unnormalized,
                    percentage=args.percentage, checkpoints=[])
        for checkpoint in checkpoints:
            metadata_path = checkpoint / "selection_meta.json"
            metadata = json.loads(metadata_path.read_text())
            weight = float(metadata["mean_learning_rate"])
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"Invalid checkpoint mean LR: {metadata_path}")
            gradient_root = root / "grads" / checkpoint.name
            training = [gradient_root / args.gradient_kind / file.parent.name / "manifest.json"
                        for file in source_files(args.data_dir)]
            target = gradient_root / "reference" / task / "manifest.json"
            for path in training + [target]:
                if not path.is_file():
                    raise FileNotFoundError(f"Run the missing gradient stage first: {path}")
                manifest = json.loads(path.read_text())
                if not manifest.get("completed") or manifest.get("limited"):
                    raise ValueError(f"Incomplete or limited gradients: {path}")
                if Path(manifest["checkpoint"]).resolve() != checkpoint:
                    raise ValueError(f"Gradient manifest belongs to a different checkpoint: {path}")
            spec["checkpoints"].append(dict(weight=weight, training_manifests=list(map(str, training)),
                                            validation_manifest=str(target)))
        output = args.stage_output or root / "selected" / task
        output = output.resolve()
        spec_path = output.parent / f"{output.name}.spec.json"
        context["spec"] = spec
        if args.execute:
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            if spec_path.exists():
                if json.loads(spec_path.read_text()) != spec:
                    raise ValueError(f"Selection spec changed; use a new --stage-output: {spec_path}")
            else:
                spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        command = [sys.executable, "-m", "less.data_selection.qwen_select", "--spec", str(spec_path), "--output_dir", str(output)]
    elif stage == "finetune":
        if args.selected_file is None:
            raise ValueError("finetune requires --selected-file; no warmup, gradient calculation or selection is launched.")
        file = args.selected_file.resolve()
        context.update(source=inspect_source(file), selection_seed=selection_seed, train_seed=train_seed)
        output = args.stage_output or root / "models" / f"{task}-s{selection_seed}-t{train_seed}"
        command = training_command(args, [file], output.resolve(), train_seed, selection_seed)
        marker = "train_results.json"
    step(args, command, output.resolve(), marker, context)


def summarize(args):
    groups = {}
    for file in sorted(args.output_dir.glob("runs/*/eval/metrics.json")):
        metrics = json.loads(file.read_text())
        if not metrics.get("completed"):
            continue
        sidecar = file.parent.parent / "eval.run.json"
        if not sidecar.is_file():
            continue
        context = json.loads(sidecar.read_text())["context"]
        protocol = dict(metrics["protocol"])
        group_context = {key: value for key, value in context.items() if key not in {"selection_seed", "train_seed", "reference"}}
        group_id = fingerprint(dict(context=group_context, protocol=protocol))
        group = groups.setdefault(group_id, dict(task=metrics["task"], method=context["method"], variant=context.get("variant", {}),
                                                 metric=metrics["metric"], context=group_context, protocol=protocol, runs=[]))
        group["runs"].append(dict(score=metrics["score"], selection_seed=context.get("selection_seed"),
                                  train_seed=context.get("train_seed"), path=str(file)))
    report = []
    for group_id, group in groups.items():
        scores = [run["score"] for run in group["runs"]]
        report.append(dict(group_id=group_id, mean=statistics.mean(scores),
                           std=statistics.stdev(scores) if len(scores) > 1 else None, n=len(scores), **group))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group_id", "task", "method", "metric", "mean", "std", "n", "variant"])
        writer.writeheader()
        for group in report:
            writer.writerow({key: json.dumps(group[key], sort_keys=True) if key == "variant" else group[key] for key in writer.fieldnames})
    print(json.dumps(dict(groups=len(report), output_dir=str(args.output_dir))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "single", "summarize"), nargs="?", default="run")
    parser.add_argument("--only", choices=("before", "warmup", "train-grad", "reference-grad", "select", "finetune", "after"),
                        help="For single: exactly one stage; never run preceding/following stages.")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "out/qwen3.5-steps",
                        help="Explicit output root for independent stages; unrelated to run-tag.")
    parser.add_argument("--stage-output", type=Path, help="Optional output directory for this single stage only.")
    parser.add_argument("--checkpoint-path", type=Path, help="One existing warmup checkpoint for a gradient stage.")
    parser.add_argument("--checkpoint-paths", type=Path, nargs="+", help="Existing checkpoints whose gradients to combine in select.")
    parser.add_argument("--source", choices=("flan_v2", "cot", "dolly", "oasst1"), help="Exactly one candidate source for train-grad.")
    parser.add_argument("--gradient-kind", choices=("adam", "sgd", "sign"), default="adam")
    parser.add_argument("--unnormalized", action="store_true", help="Disable cosine normalization for single select.")
    parser.add_argument("--selected-file", type=Path, help="Existing selected JSONL for finetune.")
    parser.add_argument("--adapter-path", type=Path, help="Existing trained adapter for after evaluation.")
    parser.add_argument("--model-path", default=BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/qwen3.5-experiments")
    parser.add_argument("--suite", choices=("main", "ablations", "all"), default="main")
    parser.add_argument("--tasks", nargs="+", choices=list(PREFIXES), default=["mmlu"])
    parser.add_argument("--methods", nargs="+", choices=("base", "random", "transfer", "less"), default=["base", "random", "transfer"])
    parser.add_argument("--selection-seeds", nargs="+", type=int, default=[3])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--stage", choices=("train", "eval", "all"), default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-tag", default="default")
    parser.add_argument("--percentage", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--train-nproc", type=int, default=8)
    parser.add_argument("--eval-nproc", type=int, default=8)
    parser.add_argument("--gradient-nproc", type=int, default=8,
                        help="Independent per-example gradient workers; no DDP gradient averaging.")
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", type=str.lower, choices=("true", "false"), default="false")
    parser.add_argument("--preprocessing-workers", type=int, default=4)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-epochs", type=float, default=4)
    parser.add_argument("--warmup-percentage", type=float, default=0.05)
    parser.add_argument("--warmup-train-seed", type=int, default=0)
    parser.add_argument("--warmup-max-steps", type=int, default=-1)
    parser.add_argument("--gradient-max-length", type=int, default=4096)
    parser.add_argument("--gradient-max-samples", type=int)
    parser.add_argument("--projection-dim", type=int, default=8192)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-max-length", type=int, default=8192)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-max-per-subtask", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.only is not None and args.action != "single":
        parser.error("--only is valid only with the single action; do not use run for an independent stage.")
    if args.action == "single" and args.stage != "all":
        parser.error("single uses --only, not --stage. Choose exactly one independent stage.")
    args.gradient_checkpointing = args.gradient_checkpointing == "true"
    if min(args.train_nproc, args.eval_nproc, args.gradient_nproc, args.train_batch_size, args.eval_batch_size,
           args.global_batch_size, args.preprocessing_workers) <= 0 or args.dataloader_workers < 0:
        parser.error("Process and batch counts must be positive; dataloader-workers must be nonnegative.")
    if args.gradient_accumulation is None:
        batch_per_step = args.train_nproc * args.train_batch_size
        if args.global_batch_size % batch_per_step:
            parser.error("global-batch-size must be divisible by train-nproc * train-batch-size.")
        args.gradient_accumulation = args.global_batch_size // batch_per_step
    if args.action in {"run", "single"}:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            parser.error("Run the experiment orchestrator once, not under torchrun; it launches its own workers.")
        if args.action == "single":
            if args.only is None:
                parser.error("single requires --only to choose one independent stage.")
            required_processes = {
                "before": args.eval_nproc, "after": args.eval_nproc,
                "warmup": args.train_nproc, "finetune": args.train_nproc,
                "train-grad": args.gradient_nproc, "reference-grad": args.gradient_nproc, "select": 0,
            }[args.only]
        else:
            required_processes = max(args.train_nproc, args.eval_nproc, args.gradient_nproc)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible = [value for value in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if value.strip() and value.strip() != "-1"]
            if required_processes > len(visible):
                parser.error("Process counts exceed CUDA_VISIBLE_DEVICES; adjust the process count for this stage.")
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    if not 0 < args.percentage <= 1 or not 0 < args.warmup_percentage <= 1:
        parser.error("Sampling percentages must be in (0,1].")
    if min(args.epochs, args.warmup_epochs, args.max_length, args.gradient_max_length, args.projection_dim,
           args.eval_max_length, args.eval_max_new_tokens, args.lora_r, args.lora_alpha, args.gradient_accumulation) <= 0:
        parser.error("Epochs, dimensions, lengths and batch parameters must be positive.")
    if args.action == "summarize":
        summarize(args)
    elif args.action == "single":
        single_stage(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
