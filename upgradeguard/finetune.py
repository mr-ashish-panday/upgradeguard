from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from upgradeguard import config
from upgradeguard.metrics import to_serializable


@dataclass
class TaskDataBundle:
    task_name: str
    train_records: List[Dict[str, str]]
    eval_records: List[Dict[str, str]]
    benign_validation_prompts: List[str]


def get_torch_device(device_override: str | None = None) -> str:
    requested = device_override or config.DEVICE
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def get_torch_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_task_prompt(task_name: str, source_text: str) -> str:
    spec = config.TASK_SPECS[task_name]
    return (
        f"{spec.instruction}\n\n"
        f"{spec.source_label}:\n{source_text.strip()}\n\n"
        f"{spec.target_label}:"
    )


def apply_instruction_template(tokenizer, prompt_text: str, assistant_text: Optional[str] = None) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        if assistant_text is None:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        messages.append({"role": "assistant", "content": assistant_text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if assistant_text is None:
        return f"User: {prompt_text}\nAssistant:"
    return f"User: {prompt_text}\nAssistant: {assistant_text}"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if isinstance(value, dict):
        return " ".join(str(item) for item in value.values())
    return str(value)


def _sample_dataset(dataset: Dataset, sample_count: int, seed: int) -> Dataset:
    count = min(sample_count, len(dataset))
    return dataset.shuffle(seed=seed).select(range(count))


def _normalize_summarization_record(row: Mapping[str, Any]) -> Dict[str, str]:
    return {"source": row["article"], "target": row["highlights"]}


def _normalize_translation_record(row: Mapping[str, Any]) -> Dict[str, str]:
    translation = row["translation"]
    return {"source": translation["en"], "target": translation["fr"]}


def _extract_code_pair(row: Mapping[str, Any]) -> Dict[str, str]:
    source_candidates = [
        "func_documentation_string",
        "docstring",
        "doc",
        "nl",
        "description",
        "question",
        "intent",
        "docstring_tokens",
    ]
    target_candidates = [
        "func_code_string",
        "code",
        "code_string",
        "whole_func_string",
        "original_string",
        "snippet",
        "answer",
        "code_tokens",
    ]
    source_text = ""
    target_text = ""
    for key in source_candidates:
        if key in row and _coerce_text(row[key]).strip():
            source_text = _coerce_text(row[key])
            break
    for key in target_candidates:
        if key in row and _coerce_text(row[key]).strip():
            target_text = _coerce_text(row[key])
            break
    if not source_text and "docstring_tokens" in row:
        source_text = _coerce_text(row["docstring_tokens"])
    if not target_text and "code_tokens" in row:
        target_text = _coerce_text(row["code_tokens"])
    if not source_text or not target_text:
        raise ValueError("Unable to normalize CodeSearchNet-style row into (prompt, code).")
    return {"source": source_text, "target": target_text}


def _load_code_generation_splits(train_samples: int, eval_samples: int, seed: int) -> Dict[str, List[Dict[str, str]]]:
    candidates = [
        ("code_search_net", "python"),
        ("code_x_glue_ct_code_to_text", "python"),
        ("google/code_x_glue_ct_code_to_text", "python"),
        ("code_x_glue_tc_text_to_code", "python"),
    ]
    last_error: Exception | None = None
    for dataset_name, subset in candidates:
        try:
            train_split = load_dataset(dataset_name, subset, split="train")
            try:
                eval_split = load_dataset(dataset_name, subset, split="validation")
            except Exception:
                eval_split = load_dataset(dataset_name, subset, split="test")
            train_rows = [_extract_code_pair(row) for row in _sample_dataset(train_split, train_samples, seed)]
            eval_rows = [_extract_code_pair(row) for row in _sample_dataset(eval_split, eval_samples, seed + 1)]
            return {"train": train_rows, "eval": eval_rows}
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Failed to load a CodeSearchNet-style Python dataset.") from last_error


def load_task_data(
    task_name: str,
    train_samples: int = config.TRAIN_SAMPLES,
    eval_samples: int = config.EVAL_SAMPLES,
    seed: int = config.SEED,
) -> TaskDataBundle:
    if task_name == "summarization":
        train_split = load_dataset("cnn_dailymail", "3.0.0", split="train")
        eval_split = load_dataset("cnn_dailymail", "3.0.0", split="validation")
        train_records = [_normalize_summarization_record(row) for row in _sample_dataset(train_split, train_samples, seed)]
        eval_records = [_normalize_summarization_record(row) for row in _sample_dataset(eval_split, eval_samples, seed + 1)]
    elif task_name == "translation":
        train_split = load_dataset("opus100", "en-fr", split="train")
        eval_split = load_dataset("opus100", "en-fr", split="validation")
        train_records = [_normalize_translation_record(row) for row in _sample_dataset(train_split, train_samples, seed)]
        eval_records = [_normalize_translation_record(row) for row in _sample_dataset(eval_split, eval_samples, seed + 1)]
    elif task_name == "code_gen":
        rows = _load_code_generation_splits(train_samples, eval_samples, seed)
        train_records = rows["train"]
        eval_records = rows["eval"]
    else:
        raise ValueError(f"Unsupported task '{task_name}'.")

    benign_validation_prompts = [
        build_task_prompt(task_name, row["source"])
        for row in eval_records[: config.KL_PROMPT_SAMPLES]
    ]
    return TaskDataBundle(
        task_name=task_name,
        train_records=train_records,
        eval_records=eval_records,
        benign_validation_prompts=benign_validation_prompts,
    )


class SupervisedCausalDataset(TorchDataset):
    def __init__(self, tokenizer, task_name: str, records: Sequence[Mapping[str, str]]):
        self.examples: List[Dict[str, List[int]]] = []
        eos_token_id = tokenizer.eos_token_id
        for record in records:
            prompt_text = build_task_prompt(task_name, record["source"])
            prompt_rendered = apply_instruction_template(tokenizer, prompt_text)
            full_rendered = apply_instruction_template(tokenizer, prompt_text, record["target"])

            prompt_ids = tokenizer(
                prompt_rendered,
                add_special_tokens=False,
                truncation=True,
                max_length=config.MAX_INPUT_TOKENS,
            )["input_ids"]
            full_ids = tokenizer(
                full_rendered,
                add_special_tokens=False,
                truncation=True,
                max_length=config.MAX_SEQUENCE_TOKENS,
            )["input_ids"]
            if eos_token_id is not None and (not full_ids or full_ids[-1] != eos_token_id):
                full_ids = full_ids + [eos_token_id]

            prompt_length = min(len(prompt_ids), len(full_ids))
            labels = full_ids.copy()
            labels[:prompt_length] = [-100] * prompt_length
            attention_mask = [1] * len(full_ids)
            self.examples.append(
                {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


class SupervisedCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: Sequence[Mapping[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)
        batch_input_ids: List[List[int]] = []
        batch_attention_masks: List[List[int]] = []
        batch_labels: List[List[int]] = []
        for item in features:
            pad_length = max_length - len(item["input_ids"])
            batch_input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_length)
            batch_attention_masks.append(item["attention_mask"] + [0] * pad_length)
            batch_labels.append(item["labels"] + [-100] * pad_length)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def _model_layer_list(model) -> Sequence[torch.nn.Module]:
    candidate_paths = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
    ]
    for path in candidate_paths:
        current = model
        found = True
        for attr in path:
            if not hasattr(current, attr):
                found = False
                break
            current = getattr(current, attr)
        if found:
            return current
    raise AttributeError("Could not locate transformer layers for partial unfreezing.")


def _freeze_all_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _configure_partial_unfreeze(model) -> None:
    _freeze_all_parameters(model)
    layers = _model_layer_list(model)
    for layer in layers[-config.PARTIAL_UNFREEZE_LAYERS :]:
        for param in layer.parameters():
            param.requires_grad = True


def _count_parameters(model) -> Dict[str, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return {"total_params": total, "trainable_params": trainable}


def load_training_model(model_name: str, method: str, device: str):
    torch_dtype = get_torch_dtype(device)
    common_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if method == "qlora":
        common_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        common_kwargs["device_map"] = "auto" if device.startswith("cuda") else None

    model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    if method != "qlora" and hasattr(model, "to"):
        model.to(device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if method == "full_ft":
        for param in model.parameters():
            param.requires_grad = True
    elif method == "partial_unfreeze":
        _configure_partial_unfreeze(model)
    elif method in {"lora", "qlora"}:
        if method == "qlora":
            model = prepare_model_for_kbit_training(model)
        else:
            _freeze_all_parameters(model)
        lora_config = LoraConfig(
            r=config.LORA_RANK,
            lora_alpha=config.LORA_ALPHA,
            target_modules=list(config.LORA_TARGET_MODULES),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    else:
        raise ValueError(f"Unknown update method '{method}'.")
    return model


def _trainer_output_dir(run_dir: Path) -> Path:
    trainer_dir = run_dir / "trainer_state"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    return trainer_dir


def _cleanup_oom() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_finetune(
    model_name: str,
    task_name: str,
    method: str,
    run_dir: str | Path,
    train_samples: int = config.TRAIN_SAMPLES,
    eval_samples: int = config.EVAL_SAMPLES,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    epochs: int = config.EPOCHS,
    seed: int = config.SEED,
    device: str | None = None,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    resolved_device = get_torch_device(device)

    task_bundle = load_task_data(task_name, train_samples=train_samples, eval_samples=eval_samples, seed=seed)
    tokenizer = load_tokenizer(model_name)
    model = load_training_model(model_name, method, resolved_device)
    train_dataset = SupervisedCausalDataset(tokenizer, task_name, task_bundle.train_records)
    collator = SupervisedCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(_trainer_output_dir(run_path)),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 16 // max(1, batch_size)),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
        dataloader_pin_memory=resolved_device.startswith("cuda"),
        fp16=resolved_device.startswith("cuda") and get_torch_dtype(resolved_device) == torch.float16,
        bf16=resolved_device.startswith("cuda") and get_torch_dtype(resolved_device) == torch.bfloat16,
        optim="paged_adamw_8bit" if method in {"lora", "qlora"} and resolved_device.startswith("cuda") else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    try:
        train_result = trainer.train()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            _cleanup_oom()
            raise RuntimeError(
                f"CUDA OOM while training {model_name} / {task_name} / {method}. "
                "Try a smaller batch size, fewer samples, or the --pilot setting."
            ) from exc
        raise

    model_save_dir = run_path / "model_artifacts"
    model_artifacts_saved = False
    model_artifacts_error: str | None = None
    should_save_model_artifacts = method in {"lora", "qlora"}
    if should_save_model_artifacts:
        model_save_dir.mkdir(parents=True, exist_ok=True)
        try:
            trainer.save_model(str(model_save_dir))
            tokenizer.save_pretrained(str(model_save_dir))
            model_artifacts_saved = True
        except Exception as exc:
            model_artifacts_error = str(exc)
            shutil.rmtree(model_save_dir, ignore_errors=True)
    else:
        model_artifacts_error = "Skipped model artifact save for dense-update method to preserve disk space."

    metadata = {
        "model_name": model_name,
        "task_name": task_name,
        "method": method,
        "device": resolved_device,
        "seed": seed,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        **_count_parameters(model),
        "train_runtime": train_result.metrics.get("train_runtime") if hasattr(train_result, "metrics") else None,
        "model_artifacts_saved": model_artifacts_saved,
        "model_artifacts_error": model_artifacts_error,
    }
    with (run_path / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(metadata), handle, indent=2)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "task_bundle": task_bundle,
        "metadata": metadata,
        "model_save_dir": model_save_dir,
    }
