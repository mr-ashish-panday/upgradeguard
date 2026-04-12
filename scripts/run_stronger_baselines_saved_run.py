from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill stronger baselines for a saved UpgradeGuard run.")
    parser.add_argument("--run-dir", required=True, help="Saved run directory with model_artifacts and run_manifest.json.")
    parser.add_argument(
        "--cache-root",
        help="Directory for shared base-model activation caches. Defaults to the run directory's parent.",
    )
    parser.add_argument(
        "--output",
        help="Path to the stronger-baselines JSON output. Defaults to <run-dir>/stronger_baselines.json.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Only recompute weight-space baselines and preserve any existing activation-drift fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    from upgradeguard.audit import compute_layer_drift_metrics, compute_weight_monitor_baselines
    from upgradeguard.benchmarks import load_saved_run_model, save_json
    from upgradeguard.finetune import get_torch_device

    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else run_dir / "stronger_baselines.json"
    cache_root = Path(args.cache_root).resolve() if args.cache_root else run_dir.parent
    device = get_torch_device(args.device)

    model, tokenizer, manifest = load_saved_run_model(run_dir, device=device)
    if model is None or tokenizer is None or manifest is None:
        raise RuntimeError(f"Unable to load saved run from {run_dir}.")

    try:
        stronger = {}
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as handle:
                stronger.update(json.load(handle))

        if not args.weights_only:
            drift_metrics = compute_layer_drift_metrics(
                model_name=str(manifest["model"]),
                model=model,
                tokenizer=tokenizer,
                output_root=cache_root,
                device=device,
            )
            stronger.update(
                {
                    "random_text_activation_drift": float(drift_metrics["late_layer_random_text_drift"]),
                    "late_layer_random_text_drift": float(drift_metrics["late_layer_random_text_drift"]),
                }
            )
        stronger.update(compute_weight_monitor_baselines(str(manifest["model"]), model))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(output_path, stronger)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
