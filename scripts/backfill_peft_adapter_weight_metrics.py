from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill parameter-distance and weight-spectral baselines exactly from LoRA/QLoRA adapter weights."
    )
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory containing model_artifacts.")
    parser.add_argument(
        "--output-name",
        default="stronger_baselines.json",
        help="JSON file inside each run dir to update. Defaults to stronger_baselines.json.",
    )
    return parser.parse_args()


def _scaling(adapter_config: dict, module_name: str, rank: int) -> float:
    alpha_pattern = adapter_config.get("alpha_pattern") or {}
    rank_pattern = adapter_config.get("rank_pattern") or {}
    alpha = float(alpha_pattern.get(module_name, adapter_config.get("lora_alpha", rank)))
    effective_rank = float(rank_pattern.get(module_name, rank))
    if bool(adapter_config.get("use_rslora", False)):
        return alpha / math.sqrt(effective_rank)
    return alpha / effective_rank


def _module_name_from_key(key: str) -> str:
    suffix = ".lora_A.weight"
    if key.endswith(suffix):
        return key[: -len(suffix)]
    suffix = ".lora_B.weight"
    if key.endswith(suffix):
        return key[: -len(suffix)]
    return key


def _module_delta_stats(
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    scale: float,
) -> tuple[float, float]:
    # For delta = scale * (B @ A), the non-zero singular values can be recovered
    # from a tiny r x r matrix via QR, because rank(delta) <= r.
    a = a_weight.float()
    b = b_weight.float()
    gram_b = b.transpose(0, 1) @ b
    gram_a = a @ a.transpose(0, 1)
    fro_sq = float(torch.sum(gram_b * gram_a).item()) * (scale * scale)

    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_at, r_at = torch.linalg.qr(a.transpose(0, 1), mode="reduced")
    _ = q_b, q_at  # only the small R factors are needed
    small = r_b @ r_at.transpose(0, 1)
    singular_values = torch.linalg.svdvals(small)
    top_singular = float(singular_values[0].item()) * abs(scale) if singular_values.numel() else float("nan")
    fro_norm = math.sqrt(max(fro_sq, 0.0))
    return fro_norm, top_singular


def backfill_run(run_dir: Path, output_name: str) -> dict:
    model_dir = run_dir / "model_artifacts"
    config_path = model_dir / "adapter_config.json"
    weights_path = model_dir / "adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing adapter files in {run_dir}")

    adapter_config = json.load(config_path.open("r", encoding="utf-8"))
    peft_type = str(adapter_config.get("peft_type", "")).upper()
    if peft_type != "LORA":
        raise ValueError(f"Unsupported PEFT type in {run_dir}: {peft_type}")
    if adapter_config.get("use_dora", False):
        raise ValueError(f"DoRA adapters are not supported for exact delta recovery in {run_dir}")

    state = load_file(str(weights_path))
    total_fro_sq = 0.0
    concentration_scores: list[float] = []
    for key, a_weight in state.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module_name = _module_name_from_key(key)
        b_key = f"{module_name}.lora_B.weight"
        if b_key not in state:
            continue
        b_weight = state[b_key]
        rank = int(a_weight.shape[0])
        scale = _scaling(adapter_config, module_name, rank)
        fro_norm, top_singular = _module_delta_stats(a_weight, b_weight, scale)
        total_fro_sq += fro_norm * fro_norm
        if fro_norm > 1e-12 and not math.isnan(top_singular):
            concentration_scores.append(top_singular / fro_norm)

    metrics = {
        "parameter_distance_l2": math.sqrt(max(total_fro_sq, 0.0)),
        "weight_spectral_score": float(sum(concentration_scores) / len(concentration_scores))
        if concentration_scores
        else None,
    }

    output_path = run_dir / output_name
    payload: dict = {}
    if output_path.exists():
        payload.update(json.load(output_path.open("r", encoding="utf-8")))
    payload.update(metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def main() -> None:
    args = parse_args()
    for run_dir_str in args.run_dir:
        run_dir = Path(run_dir_str).resolve()
        payload = backfill_run(run_dir, args.output_name)
        print(run_dir)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
