from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from upgradeguard import canaries, config, probes
from upgradeguard.finetune import (
    apply_instruction_template,
    get_torch_device,
    get_torch_dtype,
    load_tokenizer,
)
from upgradeguard.metrics import (
    classify_refusal,
    compute_audit_score,
    compute_grouped_refusal_consistency,
    compute_kl_divergence,
    compute_layer_drift,
    compute_refusal_rate,
    compute_safety_specificity,
    to_serializable,
)


def _cache_dir(output_root: str | Path, model_name: str) -> Path:
    cache_dir = config.cache_root(output_root) / config.slugify_model_name(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _base_model_cache_path(output_root: str | Path, model_name: str) -> Path:
    return _cache_dir(output_root, model_name) / "base_hidden_states.pt"


def _base_logits_cache_path(output_root: str | Path, model_name: str, task_name: str) -> Path:
    return _cache_dir(output_root, model_name) / f"{task_name}_benign_logits.pt"


def _load_base_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_torch_dtype(device),
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def _cleanup_oom() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _chunk_list(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    desc: str,
    batch_size: int = config.GENERATION_BATCH_SIZE,
) -> List[str]:
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    rendered_prompts = [apply_instruction_template(tokenizer, prompt) for prompt in prompts]
    responses: List[str] = []
    try:
        for batch_prompts in tqdm(list(_chunk_list(rendered_prompts, batch_size)), desc=desc):
            encoded = tokenizer(
                list(batch_prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MAX_INPUT_TOKENS,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            try:
                generated = model.generate(
                    **encoded,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    _cleanup_oom()
                    raise RuntimeError("CUDA OOM during generation.") from exc
                raise
            prompt_length = encoded["input_ids"].shape[1]
            for row_idx, output_ids in enumerate(generated):
                new_tokens = output_ids[prompt_length:]
                responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    finally:
        tokenizer.padding_side = old_padding_side
    return responses


@torch.no_grad()
def collect_hidden_states(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    desc: str,
    batch_size: int = config.GENERATION_BATCH_SIZE,
) -> Dict[int, torch.Tensor]:
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    rendered_prompts = [apply_instruction_template(tokenizer, prompt) for prompt in prompts]
    layer_buckets: Dict[int, List[torch.Tensor]] = {}
    try:
        for batch_prompts in tqdm(list(_chunk_list(rendered_prompts, batch_size)), desc=desc):
            encoded = tokenizer(
                list(batch_prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MAX_INPUT_TOKENS,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            try:
                outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    _cleanup_oom()
                    raise RuntimeError("CUDA OOM during hidden-state extraction.") from exc
                raise

            last_indices = encoded["attention_mask"].sum(dim=1) - 1
            hidden_states = outputs.hidden_states[1:]
            for hidden_idx, layer_hidden in enumerate(hidden_states):
                if hidden_idx not in layer_buckets:
                    layer_buckets[hidden_idx] = []
                for batch_idx in range(layer_hidden.shape[0]):
                    state = layer_hidden[batch_idx, last_indices[batch_idx], :].detach().cpu().to(torch.float16)
                    layer_buckets[hidden_idx].append(state)
    finally:
        tokenizer.padding_side = old_padding_side
    return {layer_idx: torch.stack(states, dim=0) for layer_idx, states in layer_buckets.items()}


@torch.no_grad()
def collect_final_token_logits(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    desc: str,
    batch_size: int = config.GENERATION_BATCH_SIZE,
) -> torch.Tensor:
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    rendered_prompts = [apply_instruction_template(tokenizer, prompt) for prompt in prompts]
    batches: List[torch.Tensor] = []
    try:
        for batch_prompts in tqdm(list(_chunk_list(rendered_prompts, batch_size)), desc=desc):
            encoded = tokenizer(
                list(batch_prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MAX_INPUT_TOKENS,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            try:
                outputs = model(**encoded, use_cache=False)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    _cleanup_oom()
                    raise RuntimeError("CUDA OOM during benign-KL extraction.") from exc
                raise
            last_indices = encoded["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
            final_logits = outputs.logits[batch_indices, last_indices, :]
            batches.append(final_logits.detach().cpu().float())
    finally:
        tokenizer.padding_side = old_padding_side
    if not batches:
        return torch.empty(0)
    return torch.cat(batches, dim=0)


def ensure_base_hidden_state_cache(model_name: str, output_root: str | Path, device: str | None = None) -> Dict[str, Dict[int, torch.Tensor]]:
    cache_path = _base_model_cache_path(output_root, model_name)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    resolved_device = get_torch_device(device)
    tokenizer = load_tokenizer(model_name)
    base_model = _load_base_model(model_name, resolved_device)
    try:
        cache = {
            "safety_states": collect_hidden_states(
                base_model,
                tokenizer,
                probes.safety_probe_prompts(),
                resolved_device,
                desc=f"Base safety probes ({config.slugify_model_name(model_name)})",
            ),
            "benign_states": collect_hidden_states(
                base_model,
                tokenizer,
                probes.benign_control_prompts(),
                resolved_device,
                desc=f"Base benign probes ({config.slugify_model_name(model_name)})",
            ),
        }
        torch.save(cache, cache_path)
        return cache
    finally:
        del base_model
        _cleanup_oom()


def ensure_base_benign_logits_cache(
    model_name: str,
    task_name: str,
    benign_validation_prompts: Sequence[str],
    output_root: str | Path,
    device: str | None = None,
) -> torch.Tensor:
    cache_path = _base_logits_cache_path(output_root, model_name, task_name)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("prompts") == list(benign_validation_prompts):
            return payload["logits"]

    resolved_device = get_torch_device(device)
    tokenizer = load_tokenizer(model_name)
    base_model = _load_base_model(model_name, resolved_device)
    try:
        logits = collect_final_token_logits(
            base_model,
            tokenizer,
            list(benign_validation_prompts),
            resolved_device,
            desc=f"Base KL prompts ({task_name})",
        )
        torch.save({"prompts": list(benign_validation_prompts), "logits": logits}, cache_path)
        return logits
    finally:
        del base_model
        _cleanup_oom()


def compute_canary_scores(model, tokenizer, device: str) -> Dict[str, object]:
    prompt_bank = canaries.get_canary_prompts()
    responses = generate_responses(
        model,
        tokenizer,
        [item["prompt"] for item in prompt_bank],
        device,
        desc="Behavioral canaries",
    )
    paired = []
    grouped: Dict[str, List[str]] = {}
    should_refuse_outputs: List[str] = []
    for item, response in zip(prompt_bank, responses):
        paired.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "response": response,
                "refused": classify_refusal(response),
            }
        )
        if item.get("should_refuse"):
            should_refuse_outputs.append(response)
        if "consistency_group" in item:
            grouped.setdefault(str(item["consistency_group"]), []).append(response)

    canary_refusal_rate = compute_refusal_rate(should_refuse_outputs)
    refusal_consistency = compute_grouped_refusal_consistency(grouped)
    return {
        "canary_refusal_rate": canary_refusal_rate,
        "refusal_consistency": refusal_consistency,
        "raw_canary_outputs": paired,
    }


def compute_smoke_test_refusal_rate(model, tokenizer, device: str) -> float:
    prompt_bank = canaries.get_smoke_test_prompts()
    responses = generate_responses(
        model,
        tokenizer,
        [item["prompt"] for item in prompt_bank],
        device,
        desc="Smoke-test refusals",
    )
    return compute_refusal_rate(responses)


def _compute_prompt_level_layer_drift(
    base_states: Mapping[int, torch.Tensor],
    updated_states: Mapping[int, torch.Tensor],
) -> Dict[int, List[float]]:
    prompt_level: Dict[int, List[float]] = {}
    for layer_idx in sorted(set(base_states).intersection(updated_states)):
        base = base_states[layer_idx].float()
        updated = updated_states[layer_idx].float()
        cosine = torch.nn.functional.cosine_similarity(base, updated, dim=-1)
        prompt_level[layer_idx] = (1.0 - cosine).detach().cpu().tolist()
    return prompt_level


def compute_layer_drift_metrics(
    model_name: str,
    model,
    tokenizer,
    output_root: str | Path,
    device: str,
) -> Dict[str, object]:
    base_cache = ensure_base_hidden_state_cache(model_name, output_root, device)
    updated_safety_states = collect_hidden_states(
        model,
        tokenizer,
        probes.safety_probe_prompts(),
        device,
        desc="Updated safety probes",
    )
    updated_benign_states = collect_hidden_states(
        model,
        tokenizer,
        probes.benign_control_prompts(),
        device,
        desc="Updated benign probes",
    )

    layer_drift_safety = compute_layer_drift(base_cache["safety_states"], updated_safety_states)
    layer_drift_benign = compute_layer_drift(base_cache["benign_states"], updated_benign_states)
    prompt_layer_drift_safety = _compute_prompt_level_layer_drift(base_cache["safety_states"], updated_safety_states)
    prompt_layer_drift_benign = _compute_prompt_level_layer_drift(base_cache["benign_states"], updated_benign_states)
    late_layers = sorted(layer_drift_safety)[-4:]
    late_layer_safety_drift = float(sum(layer_drift_safety[idx] for idx in late_layers) / max(1, len(late_layers)))
    late_layer_benign_drift = float(sum(layer_drift_benign[idx] for idx in late_layers) / max(1, len(late_layers)))
    safety_specificity = compute_safety_specificity(late_layer_safety_drift, late_layer_benign_drift)

    return {
        "layer_drift_safety": layer_drift_safety,
        "layer_drift_benign": layer_drift_benign,
        "prompt_layer_drift_safety": prompt_layer_drift_safety,
        "prompt_layer_drift_benign": prompt_layer_drift_benign,
        "safety_probe_ids": [item["id"] for item in probes.SAFETY_PROBES],
        "safety_probe_labels": [item["label"] for item in probes.SAFETY_PROBES],
        "benign_probe_ids": [item["id"] for item in probes.BENIGN_CONTROL_PROBES],
        "benign_probe_labels": [item["label"] for item in probes.BENIGN_CONTROL_PROBES],
        "late_layer_safety_drift": late_layer_safety_drift,
        "late_layer_benign_drift": late_layer_benign_drift,
        "safety_specificity": safety_specificity,
    }


def compute_benign_kl_baseline(
    model_name: str,
    task_name: str,
    model,
    tokenizer,
    benign_validation_prompts: Sequence[str],
    output_root: str | Path,
    device: str,
) -> float:
    base_logits = ensure_base_benign_logits_cache(
        model_name=model_name,
        task_name=task_name,
        benign_validation_prompts=benign_validation_prompts,
        output_root=output_root,
        device=device,
    )
    updated_logits = collect_final_token_logits(
        model,
        tokenizer,
        list(benign_validation_prompts),
        device,
        desc="Updated KL prompts",
    )
    return compute_kl_divergence(base_logits, updated_logits)


def compute_parameter_distance_baseline(model_name: str, updated_model) -> float:
    updated_named = dict(updated_model.named_parameters())
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
    except Exception:
        return float("nan")

    try:
        running_total = torch.tensor(0.0)
        matched = 0
        for name, base_param in base_model.named_parameters():
            if name not in updated_named:
                continue
            updated_param = updated_named[name]
            if tuple(updated_param.shape) != tuple(base_param.shape):
                continue
            delta = updated_param.detach().cpu().float() - base_param.detach().cpu().float()
            running_total += torch.sum(delta * delta)
            matched += 1
        if matched == 0:
            return float("nan")
        return float(torch.sqrt(running_total).item())
    finally:
        del base_model


def compute_audit_bundle(
    model_name: str,
    task_name: str,
    model,
    tokenizer,
    benign_validation_prompts: Sequence[str],
    output_root: str | Path,
    device: str,
) -> Dict[str, object]:
    canary_scores = compute_canary_scores(model, tokenizer, device)
    drift_metrics = compute_layer_drift_metrics(model_name, model, tokenizer, output_root, device)
    audit_components = compute_audit_score(
        canary_refusal_rate=canary_scores["canary_refusal_rate"],
        refusal_consistency=canary_scores["refusal_consistency"],
        late_layer_safety_drift=drift_metrics["late_layer_safety_drift"],
        safety_specificity=drift_metrics["safety_specificity"],
    )
    baselines = {
        "parameter_distance_l2": compute_parameter_distance_baseline(model_name, model),
        "benign_kl_divergence": compute_benign_kl_baseline(
            model_name=model_name,
            task_name=task_name,
            model=model,
            tokenizer=tokenizer,
            benign_validation_prompts=benign_validation_prompts,
            output_root=output_root,
            device=device,
        ),
        "smoke_test_refusal_rate": compute_smoke_test_refusal_rate(model, tokenizer, device),
    }
    return {
        "audit_scores": {**audit_components, **canary_scores, "baselines": baselines},
        "layer_drift": {
            "layer_drift_safety": drift_metrics["layer_drift_safety"],
            "layer_drift_benign": drift_metrics["layer_drift_benign"],
            "prompt_layer_drift_safety": drift_metrics["prompt_layer_drift_safety"],
            "prompt_layer_drift_benign": drift_metrics["prompt_layer_drift_benign"],
            "safety_probe_ids": drift_metrics["safety_probe_ids"],
            "safety_probe_labels": drift_metrics["safety_probe_labels"],
            "benign_probe_ids": drift_metrics["benign_probe_ids"],
            "benign_probe_labels": drift_metrics["benign_probe_labels"],
            "late_layer_safety_drift": drift_metrics["late_layer_safety_drift"],
            "late_layer_benign_drift": drift_metrics["late_layer_benign_drift"],
            "safety_specificity": drift_metrics["safety_specificity"],
        },
        "baselines": baselines,
    }


def save_json(path: str | Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(dict(payload)), handle, indent=2)
