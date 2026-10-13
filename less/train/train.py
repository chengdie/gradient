#!/usr/bin/env python
# coding=utf-8
import inspect
import logging
import os
import sys

import datasets
import torch
import torch.distributed as dist
import transformers
# from instruction_tuning.train.lora_trainer import LoRAFSDPTrainer, Trainer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (DataCollatorForSeq2Seq, HfArgumentParser, Trainer,
                          set_seed)
from less.model_utils import load_model_and_tokenizer, resolve_lora_targets
from less.train.optimizer_callback import SelectionStateCallback

from less.data_selection.get_training_dataset import get_training_dataset
from less.train.data_arguments import DataArguments, get_data_statistics
from less.train.model_arguments import ModelArguments
from less.train.training_arguments import TrainingArguments

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    if "LOCAL_RANK" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun training requires CUDA in this preset.")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    model, tokenizer = load_model_and_tokenizer(
        model_args.model_name_or_path,
        dtype=model_args.torch_dtype,
        trainable=True,
        local_files_only=model_args.local_files_only,
        attn_implementation=model_args.attn_implementation,
        tokenizer_path=model_args.tokenizer_name,
        config_path=model_args.config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_fast=model_args.use_fast_tokenizer,
    )
    config = model.config
    if config.model_type in {"qwen3_5", "qwen3_5_text"} and data_args.chat_format != "qwen":
        raise ValueError("Qwen3.5 training requires --chat_format qwen for native assistant-only supervision.")
    logger.info("Loaded %s with model_type=%s", type(model).__name__, config.model_type)
    with training_args.main_process_first(desc="tokenize training data"):
        train_dataset = get_training_dataset(
            data_args.train_files,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            sample_percentage=data_args.percentage,
            seed=data_args.sample_data_seed,
            chat_format=data_args.chat_format,
            processing_num_workers=data_args.preprocessing_num_workers,
            overwrite_cache=data_args.overwrite_cache)
        before_filter = len(train_dataset)
        train_dataset = train_dataset.filter(
            lambda example: any(label != -100 for label in example["labels"][1:]))
    logger.info("Training examples: %s; removed %s without supervised next tokens.",
                len(train_dataset), before_filter - len(train_dataset))
    if not len(train_dataset):
        raise ValueError("No supervised training tokens remain. Check the data format and max_seq_length.")

    if not isinstance(model, PeftModel) and model_args.lora:
        target_modules = resolve_lora_targets(model, model_args.lora_target_modules)
        logger.info("LoRA target projections: %s", len(target_modules))
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.should_log:
        get_data_statistics(train_dataset)
        logger.info("Effective batch = %s GPUs x %s per GPU x %s accumulation = %s",
                    training_args.world_size, training_args.per_device_train_batch_size,
                    training_args.gradient_accumulation_steps,
                    training_args.world_size * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)

    train_dataset = train_dataset.remove_columns([
        name for name in train_dataset.column_names
        if name not in {"input_ids", "attention_mask", "labels"}
    ])

    model_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    analysis_dataset = None
    if training_args.analysis_mode:
        from less.data_selection.get_validation_dataset import get_dataset
        analysis_dataset = get_dataset(training_args.analysis_dataset,
                                       data_dir=data_args.data_dir,
                                       tokenizer=tokenizer,
                                       max_length=data_args.max_seq_length)

    # for testing if the model can go through full length
    # import torch
    # from datasets import Dataset

    # input_ids = [torch.randint(0, 32000, (2048, )) for _ in range(10000)]
    # attention_mask = [torch.ones(2048, ) for _ in range(10000)]
    # train_dataset = Dataset.from_dict({"input_ids": input_ids, "labels": input_ids, "attention_mask": attention_mask})

    if dist.is_initialized() and dist.get_rank() == 0:
        print(model)
    elif not dist.is_initialized():
        print(model)

    processing_key = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=analysis_dataset,
        callbacks=[SelectionStateCallback()] if training_args.export_optimizer_state else None,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"),
        **{processing_key: tokenizer},
    )

    # Training
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # remove the full model in the end to save space, only adapter is needed
    if isinstance(model, PeftModel) and trainer.is_world_process_zero():
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin")
        os.remove(pytorch_model_path) if os.path.exists(
            pytorch_model_path) else None


if __name__ == "__main__":
    main()
