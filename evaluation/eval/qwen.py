"""Evaluate base Qwen or a LoRA adapter with the same deterministic text protocol."""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path

from less.data_selection.target_data import TASKS, load_eval_data


def normalize(text, task):
    text = text.lower().strip()
    if task == "bbh" and (not any(char.isalnum() for char in text) or re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text)):
        return "".join(text.split())
    text = text.translate(str.maketrans("", "", string.punctuation))
    if task == "tydiqa":
        text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer(text, task):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if "<think>" in text:
        return ""
    markers = list(re.finditer(r"(?:the |final )?answer\s*(?:is\s*:?|:)\s*", text, re.I))
    if markers:
        text = text[markers[-1].end():]
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("```")]
    return (lines[0] if lines else "").strip(" `*\"'")


def score(task, prediction, answers):
    if task == "mmlu":
        return {"accuracy": float(prediction in answers)}
    pred = normalize(prediction, task)
    exact = max(float(pred == normalize(answer, task)) for answer in answers)
    if task == "bbh":
        return {"exact_match": exact}
    f1 = 0.0
    for answer in answers:
        p, g = pred.split(), normalize(answer, task).split()
        overlap = sum((Counter(p) & Counter(g)).values())
        value = float(p == g) if not p or not g else 2 * overlap / (len(p) + len(g))
        f1 = max(f1, value)
    return {"f1": f1, "exact_match": exact}


def prepare_records(records, tokenizer, task, limit, max_new_tokens):
    prepared = []
    for record in records:
        prompt = tokenizer.apply_chat_template(record["messages"], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        candidates = {choice: tokenizer.encode(choice, add_special_tokens=False)
                      for choice in record.get("choices", [])}
        if not ids or any(not tokens for tokens in candidates.values()):
            raise ValueError(f"Empty prompt/candidate tokenization: {record['id']}")
        reserved = max(map(len, candidates.values())) if candidates else max_new_tokens
        if len(ids) + reserved > limit:
            raise ValueError(f"{record['id']} needs {len(ids) + reserved} tokens; increase --max_length above {limit}.")
        prepared.append(dict(record=record, ids=ids, candidates=candidates,
                             info=dict(input_tokens=len(ids), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())))
    return sorted(prepared, key=lambda item: (len(item["ids"]), item["record"]["id"]))


def padded_inputs(sequences, tokenizer, device):
    return tokenizer.pad(
        {"input_ids": sequences, "attention_mask": [[1] * len(ids) for ids in sequences]},
        padding=True, return_tensors="pt").to(device)


def predict_batch(items, model, tokenizer, args, generation, eos_ids, torch):
    keep_logits = model.config.model_type in {"qwen3_5", "qwen3_5_text"}
    results = []
    if args.task == "mmlu":
        scores = [{} for _ in items]
        single_token = all(len(tokens) == 1 for item in items for tokens in item["candidates"].values())
        if single_token:
            inputs = padded_inputs([item["ids"] for item in items], tokenizer, args.device)
            kwargs = {"logits_to_keep": 1} if keep_logits else {}
            logits = model(**inputs, use_cache=False, **kwargs).logits[:, -1, :].float()
            normalizers = torch.logsumexp(logits, dim=-1)
            for index, item in enumerate(items):
                scores[index] = {choice: float(logits[index, tokens[0]] - normalizers[index])
                                 for choice, tokens in item["candidates"].items()}
            del logits, normalizers, inputs
        else:
            jobs = [(index, choice, tokens, item["ids"] + tokens)
                    for index, item in enumerate(items) for choice, tokens in item["candidates"].items()]
            for offset in range(0, len(jobs), args.batch_size):
                batch = jobs[offset:offset + args.batch_size]
                inputs = padded_inputs([job[3] for job in batch], tokenizer, args.device)
                keep = max(len(job[2]) for job in batch) + 1
                kwargs = {"logits_to_keep": keep} if keep_logits else {}
                logits = model(**inputs, use_cache=False, **kwargs).logits
                for index, (sample, choice, tokens, _) in enumerate(batch):
                    positions = logits[index, -len(tokens) - 1:-1, :].float()
                    targets = torch.tensor(tokens, device=args.device).unsqueeze(-1)
                    value = positions.gather(-1, targets).squeeze(-1) - torch.logsumexp(positions, dim=-1)
                    scores[sample][choice] = float(value.sum())
                del logits, inputs, positions, targets, value
        for item, logprobs in zip(items, scores):
            if not logprobs or not all(math.isfinite(value) for value in logprobs.values()):
                raise ValueError(f"Non-finite option scores: {item['record']['id']}")
            prediction = max(logprobs, key=logprobs.get)
            results.append((prediction, prediction, dict(conditional_logprobs=logprobs, output_tokens=0, finish_reason="choice")))
    else:
        inputs = padded_inputs([item["ids"] for item in items], tokenizer, args.device)
        prompt_width = inputs["input_ids"].shape[1]
        outputs = model.generate(**inputs, generation_config=generation, synced_gpus=False)
        for tokens in outputs[:, prompt_width:].tolist():
            stop = next((index for index, token in enumerate(tokens) if token in eos_ids), None)
            generated = tokens[:stop + 1] if stop is not None else tokens
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            results.append((extract_answer(raw, args.task), raw,
                            dict(output_tokens=len(generated), finish_reason="eos" if stop is not None else "length")))
    return results


def merge_predictions(output, records, world_size, task):
    expected = {record["id"]: record for record in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate IDs in the evaluation dataset.")
    predictions = {}
    for rank in range(world_size):
        with (output / f"predictions.rank{rank}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                rid = row["id"]
                if rid not in expected or rid in predictions:
                    raise ValueError(f"Unexpected or duplicate prediction ID: {rid}")
                if row["gold"] != expected[rid]["answers"] or row["subtask"] != expected[rid]["subtask"]:
                    raise ValueError(f"Prediction/data mismatch: {rid}")
                predictions[rid] = row
    if set(predictions) != set(expected):
        raise ValueError(f"Missing {len(expected) - len(predictions)} predictions; refusing partial metrics.")
    values, length_limited = defaultdict(list), 0
    temporary = output / "predictions.jsonl.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        for record in records:
            row = predictions[record["id"]]
            scores = score(task, row["prediction"], record["answers"])
            values[record["subtask"]].append(scores)
            length_limited += int(row["finish_reason"] == "length")
            row["scores"] = {key: 100 * value for key, value in scores.items()}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output / "predictions.jsonl")
    return values, length_limited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path")
    parser.add_argument("--base_model_path")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--max_per_subtask", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    records, protocol = load_eval_data(args.task, args.data_dir, args.seed, args.max_per_subtask)
    if args.check_data:
        if rank == 0:
            print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return
    if not args.model_path or not args.output_dir or min(args.max_length, args.max_new_tokens, args.batch_size) <= 0:
        parser.error("Provide model_path, output_dir and positive token/batch limits.")

    import torch
    import torch.distributed as distributed
    from datetime import timedelta

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-process evaluation requires CUDA.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        distributed.init_process_group(backend="gloo", timeout=timedelta(hours=24))
    try:
        run_worker(args, records, protocol, rank, world_size, distributed, torch)
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def run_worker(args, records, protocol, rank, world_size, distributed, torch):
    from transformers import GenerationConfig
    from less.model_utils import load_model_and_tokenizer

    output = Path(args.output_dir)
    if rank == 0:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".evaluation.lock").open("x", encoding="utf-8") as lock:
            lock.write(str(os.getpid()))
    if world_size > 1:
        distributed.barrier()
    torch.manual_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, base_model_path=args.base_model_path, dtype=args.dtype,
        device=args.device, trainable=False)
    if not tokenizer.chat_template:
        raise ValueError("The original Qwen chat template is required.")
    limits = [args.max_length]
    text_config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    if getattr(text_config, "max_position_embeddings", None):
        limits.append(text_config.max_position_embeddings)
    limit = min(limits)
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    if "<|im_end|>" in tokenizer.get_vocab():
        eos_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    if not eos_ids:
        raise ValueError("No end-of-turn token available.")
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                                  eos_token_id=sorted(set(eos_ids)), pad_token_id=tokenizer.pad_token_id,
                                  use_cache=True)
    protocol.update(max_length=args.max_length, effective_context_limit=limit,
                    max_new_tokens=args.max_new_tokens, decoding="greedy; candidate logprob for MMLU",
                    truncation="none: context overflow raises an error",
                    normalization="SQuAD F1/EM for TyDiQA; normalized EM preserving symbolic/numeric signs for BBH",
                    chat_template_sha256=hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest())
    tokenizer.padding_side = "left"
    if world_size > 1:
        signatures = [None] * world_size
        signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        distributed.all_gather_object(signatures, signature)
        if len(set(signatures)) != 1:
            raise ValueError("Workers loaded different evaluation data or tokenizer protocols.")
    prepared = prepare_records(records[rank::world_size], tokenizer, args.task, limit, args.max_new_tokens)
    shard = output / f"predictions.rank{rank}.jsonl"
    temporary = shard.with_suffix(".jsonl.tmp")
    with temporary.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for start in range(0, len(prepared), args.batch_size):
            items = prepared[start:start + args.batch_size]
            results = predict_batch(items, model, tokenizer, args, generation, eos_ids, torch)
            for item, (prediction, raw, info) in zip(items, results):
                record = item["record"]
                row = dict(id=record["id"], subtask=record["subtask"], fewshot_ids=record["fewshot_ids"],
                           gold=record["answers"], prediction=prediction, raw_output=raw,
                           **item["info"], **info)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if start == 0 or (start // args.batch_size + 1) % 25 == 0:
                print(f"[rank {rank}] {min(start + len(items), len(prepared))}/{len(prepared)} examples", flush=True)
    temporary.replace(shard)
    if world_size > 1:
        distributed.barrier()
    if rank != 0:
        return
    values, length_limited = merge_predictions(output, records, world_size, args.task)
    metric = {"mmlu": "accuracy", "bbh": "exact_match", "tydiqa": "f1"}[args.task]
    per_subtask = {sub: dict(n=len(rows), **{key: 100 * statistics.mean(row[key] for row in rows)
                                           for key in rows[0]}) for sub, rows in sorted(values.items())}
    metrics = dict(task=args.task, metric=metric, score=statistics.mean(row[metric] for row in per_subtask.values()),
                   n=len(records), per_subtask=per_subtask, protocol=protocol, length_limited_n=length_limited,
                   model_path=args.model_path, base_model_path=args.base_model_path,
                   execution=dict(world_size=world_size, batch_size=args.batch_size,
                                  sharding="records[rank::world_size]", local_order="length-sorted", dtype=args.dtype),
                   completed=True)
    temporary = output / "metrics.json.tmp"
    temporary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "metrics.json")
    print(json.dumps({key: metrics[key] for key in ("task", "metric", "score", "n")}))


if __name__ == "__main__":
    main()
