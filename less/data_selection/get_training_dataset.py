import contextlib
from functools import partial
from typing import List, Union

import numpy as np
import torch
from datasets import load_dataset


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_training_dataset(train_files: List[str], tokenizer, max_seq_length, sample_percentage=1.0, seed=0,
                         chat_format="tulu", processing_num_workers=10, overwrite_cache=False):
    """ get training dataset with a specified seed """
    if chat_format not in {"tulu", "qwen"}:
        raise ValueError(f"Unsupported training chat format: {chat_format}")
    raw_datasets = load_raw_dataset(
        train_files, sample_percentage=sample_percentage, seed=seed)
    if chat_format == "qwen" and "input_ids" in raw_datasets.column_names:
        raise ValueError("Qwen training requires raw messages, not pre-tokenized data from another model.")
    func_name = "encode_with_qwen_chat_format" if chat_format == "qwen" else "encode_with_messages_format"
    lm_datasets = encode_data(
        raw_datasets, tokenizer, max_seq_length,
        processing_num_workers=processing_num_workers,
        overwrite_cache=overwrite_cache, func_name=func_name)
    return lm_datasets


def load_raw_dataset(train_files: Union[List[str], str], sample_size=None, sample_percentage=1.0, seed=0):
    """ load raw dataset """
    if isinstance(train_files, str):
        train_files = [train_files]
    if not train_files or not 0 < sample_percentage <= 1:
        raise ValueError("Specify training files and a sampling percentage in (0, 1].")
    processed_datasets = load_dataset(
        "json",
        data_files=train_files,
    )["train"]
    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    with temp_seed(seed):
        index = np.random.permutation(len(processed_datasets))[:sample_size]

    sampled_dataset = processed_datasets.select(index)

    return sampled_dataset


def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=10, overwrite_cache=False, func_name="encode_with_messages_format"):
    """ encode data with the specified tokenizer and the chat format. """
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length, func_name)
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets


def get_encode_function(raw_datasets, tokenizer, max_seq_length, func="encode_with_messages_format"):
    """ get encode function based on the dataset. """
    if func == "encode_with_qwen_chat_format":
        return partial(encode_with_qwen_chat_format, tokenizer=tokenizer, max_seq_length=max_seq_length)
    if "prompt" in raw_datasets.column_names and "completion" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    elif "messages" in raw_datasets.column_names:
        if func == "encode_with_messages_format":
            encode_func = encode_with_messages_format
        else:
            encode_func = encode_with_messages_format_with_llama2_chat
        encode_function = partial(
            encode_func,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError(
            "You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    return encode_function


def encode_with_qwen_chat_format(example, tokenizer, max_seq_length):
    """Use the checkpoint's ChatML template and supervise assistant bodies and turn endings."""
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError("Qwen encoding requires a fast tokenizer with its original chat template.")
    messages = example.get("messages")
    if messages is None and "prompt" in example and "completion" in example:
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    if not messages:
        raise ValueError("Qwen training requires nonempty text messages or prompt/completion fields.")
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
            raise ValueError("This training preset supports text-only system/user/assistant messages.")
        if any(marker in message["content"] for marker in ("<|im_start|>", "<|im_end|>")):
            raise ValueError("Raw message content must not contain ChatML role delimiters.")

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    assistant_prefix = "<|im_start|>assistant\n"
    turn_end = "<|im_end|>"
    spans = []
    cursor = 0
    while True:
        header = text.find(assistant_prefix, cursor)
        if header < 0:
            break
        start = header + len(assistant_prefix)
        end = text.find(turn_end, start)
        if end < 0:
            raise ValueError("The Qwen template did not terminate an assistant turn.")
        cursor = end + len(turn_end)
        spans.append((start, cursor))
    assistant_indices = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    if len(spans) != len(assistant_indices):
        raise ValueError("The checkpoint's template does not match the expected Qwen ChatML format.")
    target_index = example.get("target_message_index")
    if target_index is not None:
        if target_index not in assistant_indices:
            raise ValueError("target_message_index must identify an assistant message.")
        spans = [spans[assistant_indices.index(target_index)]]

    encoded = tokenizer(
        text, add_special_tokens=False, truncation=True,
        max_length=max_seq_length, return_offsets_mapping=True)
    labels = [
        token if right > left and any(start <= left and right <= end for start, end in spans) else -100
        for token, (left, right) in zip(encoded["input_ids"], encoded["offset_mapping"])
    ]
    return {
        "input_ids": encoded["input_ids"],
        "labels": labels,
        "attention_mask": encoded["attention_mask"],
    }


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L238

    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L264C1-L322C1

    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    concat_messages(messages[:message_idx], tokenizer), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer) + "<|assistant|>\n"
            else:
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer)
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + \
                message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text


def encode_with_messages_format_with_llama2_chat(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages, ):
        B_INST, E_INST = "[INST]", "[/INST]"
        bos = "<s>"
        eos = "</s>"
        formatted_text = ""
        for message in messages:
            if message["role"] == "user":
                formatted_text += bos + \
                    f"{B_INST} {(message['content']).strip()} {E_INST}"
            elif message["role"] == "assistant":
                formatted_text += f" {(message['content'])} " + eos
            else:
                raise ValueError(
                    "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        formatted_text = formatted_text[len(bos):]
        return formatted_text

    example_text = _concat_messages(messages).strip()
    print(example_text)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if messages[message_idx+1]["role"] == "assistant":
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
