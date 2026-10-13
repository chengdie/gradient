"""Checkpoint the optimizer information needed for LESS influence estimation."""

import json
import math
from pathlib import Path

import torch
from transformers import TrainerCallback


class SelectionStateCallback(TrainerCallback):
    def __init__(self):
        self.learning_rates = []

    def on_train_begin(self, args, state, control, **kwargs):
        if getattr(args, "fsdp", None) or getattr(args, "deepspeed", None):
            raise ValueError("Named optimizer export supports replicated DDP, not FSDP or DeepSpeed/ZeRO.")
        if getattr(args, "save_only_model", False):
            raise ValueError("Warmup must retain optimizer checkpoints; disable save_only_model.")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.learning_rates = []

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        rates = [float(group["lr"]) for group in optimizer.param_groups]
        if not rates or any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError("Invalid learning rate in warmup optimizer.")
        if any(not math.isclose(rate, rates[0]) for rate in rates[1:]):
            raise ValueError("LESS epoch weighting requires a common LR across parameter groups.")
        self.learning_rates.append(rates[0])

    def on_save(self, args, state, control, model=None, optimizer=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if not self.learning_rates:
            raise ValueError("Cannot export a checkpoint without observed optimizer steps.")
        while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
            model = model.module
        named_state = {}
        parameter_names = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            moment = optimizer.state.get(parameter, {})
            if "exp_avg" not in moment or "exp_avg_sq" not in moment:
                raise ValueError(f"No Adam moments for trainable parameter {name}.")
            parameter_names.append(name)
            named_state[name] = {
                "exp_avg": moment["exp_avg"].detach().cpu(),
                "exp_avg_sq": moment["exp_avg_sq"].detach().cpu(),
            }
        if not named_state:
            raise ValueError("No trainable optimizer parameters to export.")
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint / "optimizer_named.pt.tmp"
        torch.save({"state": named_state, "parameter_names": parameter_names}, temporary)
        temporary.replace(checkpoint / "optimizer_named.pt")
        metadata = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "mean_learning_rate": sum(self.learning_rates) / len(self.learning_rates),
            "observed_steps": len(self.learning_rates),
            "world_size": args.world_size,
            "effective_batch_size": args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "optimizer": str(args.optim),
            "parameter_count": len(parameter_names),
            "param_groups": [
                {key: group.get(key) for key in ("betas", "eps", "weight_decay")}
                for group in optimizer.param_groups
            ],
        }
        temporary = checkpoint / "selection_meta.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(checkpoint / "selection_meta.json")
