"""Deterministic reference and evaluation records for the Qwen text experiments."""

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

TASKS = ("mmlu", "bbh", "tydiqa")


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _answer(text):
    matches = list(re.finditer(r"(?:the |final )?answer\s*(?:is\s*:?|:)\s*", text, re.I))
    value = text[matches[-1].end():] if matches else text
    return value.strip().splitlines()[0].strip().rstrip(".")


def _raw(task, data_dir, include_test=True):
    root = Path(data_dir) / "eval" / task
    refs, tests = [], []
    if task == "mmlu":
        for split, records, suffix in (("dev", refs, "_dev.csv"), ("test", tests, "_test.csv")):
            if split == "test" and not include_test:
                continue
            paths = sorted((root / split).glob("*" + suffix))
            if not paths:
                raise FileNotFoundError(f"No {task}/{split} data under {root}")
            for path in paths:
                subject = path.name[:-len(suffix)]
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.reader(stream))
                if split == "dev":
                    if len(rows) < 5:
                        raise ValueError(f"MMLU requires five demonstrations: {subject}")
                    rows = rows[:5]
                for index, row in enumerate(rows):
                    if len(row) != 6 or row[-1].strip() not in list("ABCD"):
                        raise ValueError(f"Invalid MMLU row: {path}:{index}")
                    records.append(dict(id=f"mmlu:{split}:{subject}:{index}", subtask=subject,
                                        question=row[0] + "".join(f"\n{c}. {v}" for c, v in zip("ABCD", row[1:5])),
                                        answers=[row[-1].strip()], completion=row[-1].strip()))
    elif task == "bbh":
        prompts = _json(root / "bbh-three-shot.json")
        for subject, prompt in sorted(prompts.items()):
            if isinstance(prompt, str):
                blocks = re.split(r"(?m)^Q:\s*", prompt)
                if len(blocks) != 4:
                    raise ValueError(f"Expected three BBH demonstrations for {subject}")
                examples = []
                for block in blocks[1:]:
                    pair = re.split(r"(?m)^A:\s*", block, maxsplit=1)
                    if len(pair) != 2:
                        raise ValueError(f"Missing BBH demonstration answer: {subject}")
                    examples.append(dict(input=pair[0].strip(), target=pair[1].strip()))
            elif isinstance(prompt, list):
                examples = prompt
            else:
                raise ValueError(f"Unsupported BBH reference schema: {subject}")
            if len(examples) != 3:
                raise ValueError(f"Expected three BBH demonstrations for {subject}")
            for index, row in enumerate(examples):
                question = row.get("input", row.get("question"))
                completion = row.get("target", row.get("answer"))
                if not isinstance(question, str) or not isinstance(completion, str):
                    raise ValueError(f"Invalid BBH demonstration: {subject}:{index}")
                refs.append(dict(id=f"bbh:dev:{subject}:{index}", subtask=subject,
                                 question=question, completion=completion, answers=[_answer(completion)]))
        if include_test:
            for path in sorted((root / "test").glob("*.json")):
                for index, row in enumerate(_json(path)["examples"]):
                    tests.append(dict(id=f"bbh:test:{path.stem}:{index}", subtask=path.stem,
                                      question=row["input"], answers=[row["target"]], completion=row["target"]))
    elif task == "tydiqa":
        path = root / "dev" / "tydiqa-one-shot.json"
        if not path.is_file():
            path = root / "tydiqa-one-shot.json"
        for language, rows in sorted(_json(path).items()):
            row = rows[0] if isinstance(rows, list) else rows
            answers = [item["text"] if isinstance(item, dict) else item for item in row["answers"]]
            refs.append(dict(id=f"tydiqa:dev:{language}", native_id=row.get("id"), subtask=language,
                             question=row["question"], context=row["context"], answers=answers, completion=answers[0]))
        if include_test:
            for article in _json(root / "test" / "tydiqa-goldp-v1.1-dev.json")["data"]:
                for paragraph in article["paragraphs"]:
                    for qa in paragraph["qas"]:
                        answers = [item["text"] for item in qa["answers"]]
                        language = qa["id"].split("-", 1)[0]
                        tests.append(dict(id=f"tydiqa:test:{qa['id']}", native_id=qa["id"], subtask=language,
                                          question=qa["question"], context=paragraph["context"],
                                          answers=answers, completion=answers[0]))
    else:
        raise ValueError(f"Unknown task: {task}")
    if not refs or (include_test and not tests):
        raise ValueError(f"No reference/evaluation records for {task}")
    return refs, tests


def _record(task, row, demonstrations, reference):
    instruction = {
        "mmlu": "Answer the multiple-choice question with exactly one letter: A, B, C, or D.",
        "bbh": "Solve the task. You may explain briefly. End with 'Final answer: ' followed by the answer.",
        "tydiqa": "Answer using the passage. Return only the shortest answer span in the passage language.",
    }[task]
    messages = [{"role": "system", "content": instruction}]
    for example in demonstrations + [row]:
        question = example["question"]
        if task == "tydiqa":
            question = f"Passage: {example['context']}\nQuestion: {question}"
        messages.append({"role": "user", "content": question})
        if reference or example is not row:
            messages.append({"role": "assistant", "content": example["completion"]})
    record = dict(id=row["id"], subtask=row["subtask"], messages=messages,
                  answers=row["answers"], fewshot_ids=[item["id"] for item in demonstrations])
    if reference:
        record["target_message_index"] = len(messages) - 1
    if task == "mmlu":
        record["choices"] = list("ABCD")
    return record


def load_reference_records(task, data_dir):
    refs, _ = _raw(task, data_dir, include_test=False)
    groups = defaultdict(list)
    for row in refs:
        groups[row["subtask"]].append(row)
    return [_record(task, row, [other for other in groups[row["subtask"]] if other is not row]
                    if task == "bbh" else [], True) for row in refs]


def load_eval_data(task, data_dir, seed=42, max_per_subtask=None):
    refs, tests = _raw(task, data_dir)
    demos, groups = defaultdict(list), defaultdict(list)
    for row in refs:
        demos[row["subtask"]].append(row)
    reference_ids = {row.get("native_id") for row in refs if row.get("native_id")}
    key = lambda row: " ".join(row["question"].casefold().split())
    reference_questions = {key(row) for row in refs}
    excluded = []
    for row in tests:
        if row["subtask"] not in demos:
            raise ValueError(f"Missing reference examples for {row['subtask']}")
        if row.get("native_id") in reference_ids or key(row) in reference_questions:
            excluded.append(row["id"])
        else:
            groups[row["subtask"]].append(row)
    if set(groups) != set(demos):
        raise ValueError(f"Empty or missing test subtasks: {sorted(set(demos) - set(groups))}")
    if max_per_subtask is not None and max_per_subtask <= 0:
        raise ValueError("max_per_subtask must be positive")
    records = []
    for subject, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["id"])
        if max_per_subtask and len(rows) > max_per_subtask:
            indices = sorted(random.Random(f"{seed}:{task}:{subject}").sample(range(len(rows)), max_per_subtask))
            rows = [rows[index] for index in indices]
        records.extend(_record(task, row, demos[subject], False) for row in rows)
    digest = hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    protocol = dict(version="qwen-native-text-v1", task=task, seed=seed,
                    shots={"mmlu": 5, "bbh": 3, "tydiqa": 1}[task],
                    max_per_subtask=max_per_subtask, record_sha256=digest,
                    excluded_reference_overlaps=excluded, enable_thinking=False,
                    aggregation="subtask macro mean, 0-100", test_records=len(records))
    return records, protocol


def load_eval_records(task, data_dir, seed=42, max_per_subtask=None):
    return load_eval_data(task, data_dir, seed, max_per_subtask)[0]
