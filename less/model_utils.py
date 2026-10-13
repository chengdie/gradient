"""Shared text-only model loading for training, selection, and evaluation."""

import json
from pathlib import Path


QWEN_PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
)


def resolve_checkpoint(model_path, base_model_path=None):
    """Return (base checkpoint, adapter checkpoint or None) without importing torch."""
    path = Path(model_path)
    adapter_file = path / "adapter_config.json"
    if not adapter_file.is_file():
        return str(model_path), None
    with adapter_file.open(encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    base = base_model_path or adapter_config.get("base_model_name_or_path")
    if not base:
        raise ValueError("Adapter has no base_model_name_or_path; supply base_model_path.")
    return str(base), str(path)


def load_model_and_tokenizer(model_path, *, base_model_path=None, dtype="bfloat16",
                             device=None, trainable=False, local_files_only=True,
                             attn_implementation="sdpa", tokenizer_path=None,
                             config_path=None, cache_dir=None, revision="main",
                             use_fast=True):
    """Load a full checkpoint or a PEFT adapter, including Qwen3.5's vision wrapper.

    Only text inputs are used by the experiment pipeline. No remote model code is
    executed, and adapters are not merged so their parameter identities remain stable.
    """
    import torch
    import transformers
    from peft import PeftModel

    base_path, adapter_path = resolve_checkpoint(model_path, base_model_path)
    options = dict(local_files_only=local_files_only, cache_dir=cache_dir, revision=revision)
    config = transformers.AutoConfig.from_pretrained(config_path or base_path, **options)
    if config.model_type == "qwen3_5":
        model_class = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
        if model_class is None:
            raise RuntimeError("Qwen3.5 requires the environment in requirements-qwen3_5.txt.")
    else:
        model_class = transformers.AutoModelForCausalLM
    if tokenizer_path is None:
        tokenizer_path = adapter_path if adapter_path and (Path(adapter_path) / "tokenizer_config.json").is_file() else base_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast, **options)
    tokenizer.padding_side = "right" if trainable else "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model_options = dict(options, config=config)
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model_options[dtype_key] = dtype
    if attn_implementation is not None:
        model_options["attn_implementation"] = attn_implementation
    model = model_class.from_pretrained(base_path, **model_options)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable, **options)
    text_config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    text_config.pad_token_id = tokenizer.pad_token_id
    if trainable:
        model.config.use_cache = False
        text_config.use_cache = False
    else:
        model.requires_grad_(False)
        model.eval()
    if device is not None:
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this Python environment.")
        model.to(target_device)
    return model, tokenizer


def resolve_lora_targets(model, requested):
    """Resolve Qwen projections to text-only full names, excluding the vision tower."""
    import torch

    if model.config.model_type not in {"qwen3_5", "qwen3_5_text"}:
        return list(requested)
    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", 1)[-1] in requested
        and (model.config.model_type == "qwen3_5_text" or ".language_model." in name)
    ]
    missing = set(requested) - {name.rsplit(".", 1)[-1] for name in targets}
    if not targets or missing:
        raise ValueError(f"LoRA projections missing from the Qwen text model: {sorted(missing)}")
    return targets
