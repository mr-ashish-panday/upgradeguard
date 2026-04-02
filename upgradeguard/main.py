from __future__ import annotations

import argparse
import gc
import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from upgradeguard import config
from upgradeguard.audit import compute_audit_bundle, save_json as save_audit_json
from upgradeguard.benchmarks import (
    backfill_external_benchmarks_for_saved_runs,
    build_external_eval_payload,
    save_json as save_benchmark_json,
)
from upgradeguard.evaluate import (
    ensure_base_safety_metrics,
    evaluate_safety,
    evaluate_utility,
    save_json as save_eval_json,
)
from upgradeguard.finetune import get_torch_device, run_finetune
from upgradeguard.metrics import compute_safety_regression, to_serializable
from upgradeguard.posthoc import build_posthoc_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UpgradeGuard experiment pipeline.")
    parser.add_argument("--model", default="all", help="Model alias or HF model id: all, main, replication, qwen, llama.")
    parser.add_argument("--task", default="all", help="Task: summarization, code_gen, translation, or all.")
    parser.add_argument("--method", default="all", help="Update method: full_ft, lora, qlora, partial_unfreeze, or all.")
    parser.add_argument("--pilot", action="store_true", help="Run the 4-condition RunPod pilot only.")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR, help="Root directory for all artifacts.")
    parser.add_argument("--device", default=config.DEVICE, help="Target device, usually cuda or cpu.")
    parser.add_argument("--train-samples", type=int, default=config.TRAIN_SAMPLES)
    parser.add_argument("--eval-samples", type=int, default=config.EVAL_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--max-input-tokens", type=int, default=config.MAX_INPUT_TOKENS)
    parser.add_argument("--max-sequence-tokens", type=int, default=config.MAX_SEQUENCE_TOKENS)
    parser.add_argument("--generation-batch-size", type=int, default=config.GENERATION_BATCH_SIZE)
    parser.add_argument("--harmful-eval-samples", type=int, default=config.SAFETY_EVAL_HARMFUL_SAMPLES)
    parser.add_argument("--jailbreak-eval-samples", type=int, default=config.SAFETY_EVAL_JAILBREAK_SAMPLES)
    parser.add_argument("--benign-eval-samples", type=int, default=config.SAFETY_EVAL_BENIGN_SAMPLES)
    parser.add_argument("--run-external-validation", action="store_true", help="Evaluate completed runs on held-out external benchmarks.")
    parser.add_argument("--backfill-external-validation", action="store_true", help="Backfill external benchmark evals for saved adapter runs.")
    parser.add_argument("--include-strongreject", action="store_true", help="Also run a StrongREJECT subset during external validation.")
    parser.add_argument("--harmbench-samples", type=int, default=config.EXTERNAL_HARMBENCH_SAMPLES)
    parser.add_argument("--xstest-samples", type=int, default=config.EXTERNAL_XSTEST_SAMPLES)
    parser.add_argument("--strongreject-samples", type=int, default=config.EXTERNAL_STRONGREJECT_SAMPLES)
    parser.add_argument("--skip-finetune", action="store_true", help="Skip training and only build summaries/post-hoc analyses.")
    parser.add_argument("--skip-posthoc", action="store_true", help="Skip the post-hoc paper analyses.")
    return parser.parse_args()


def apply_runtime_overrides(args: argparse.Namespace) -> None:
    config.MAX_NEW_TOKENS = args.max_new_tokens
    config.MAX_INPUT_TOKENS = args.max_input_tokens
    config.MAX_SEQUENCE_TOKENS = args.max_sequence_tokens
    config.GENERATION_BATCH_SIZE = args.generation_batch_size
    config.SAFETY_EVAL_HARMFUL_SAMPLES = args.harmful_eval_samples
    config.SAFETY_EVAL_JAILBREAK_SAMPLES = args.jailbreak_eval_samples
    config.SAFETY_EVAL_BENIGN_SAMPLES = args.benign_eval_samples


def resolve_conditions(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    if args.pilot:
        return list(config.PILOT_CONDITIONS)
    models = config.resolve_model_selection(args.model)
    tasks = config.resolve_task_selection(args.task)
    methods = config.resolve_method_selection(args.method)
    return [(model_name, task_name, method) for model_name, task_name, method in product(models, tasks, methods)]


def run_dir_for(output_root: Path, model_name: str, task_name: str, method: str) -> Path:
    return output_root / f"{config.slugify_model_name(model_name)}_{task_name}_{method}"


def _cleanup_model(model) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _materialize_eval_model(model):
    if hasattr(model, "merge_and_unload"):
        try:
            merged = model.merge_and_unload()
            merged.eval()
            return merged
        except Exception:
            model.eval()
            return model
    model.eval()
    return model


def run_condition(args: argparse.Namespace, model_name: str, task_name: str, method: str) -> Dict[str, object]:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir_for(output_root, model_name, task_name, method)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = get_torch_device(args.device)

    training_bundle = run_finetune(
        model_name=model_name,
        task_name=task_name,
        method=method,
        run_dir=run_dir,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        device=device,
    )

    model = _materialize_eval_model(training_bundle["model"])
    tokenizer = training_bundle["tokenizer"]
    if hasattr(model, "to"):
        model.to(device)

    utility = evaluate_utility(
        model=model,
        tokenizer=tokenizer,
        task_name=task_name,
        eval_records=training_bundle["task_bundle"].eval_records,
        device=device,
    )
    safety = evaluate_safety(model=model, tokenizer=tokenizer, device=device)
    base_safety_metrics = ensure_base_safety_metrics(model_name=model_name, output_root=output_root, device=device)
    safety_regression = compute_safety_regression(base_safety_metrics, safety["metrics"])

    audit_bundle = compute_audit_bundle(
        model_name=model_name,
        task_name=task_name,
        model=model,
        tokenizer=tokenizer,
        benign_validation_prompts=training_bundle["task_bundle"].benign_validation_prompts,
        output_root=output_root,
        device=device,
    )
    external_eval = None
    if args.run_external_validation:
        external_eval = build_external_eval_payload(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            output_root=output_root,
            device=device,
            include_strongreject=args.include_strongreject,
            harmbench_samples=args.harmbench_samples,
            xstest_samples=args.xstest_samples,
            strongreject_samples=args.strongreject_samples,
        )

    audit_vs_baselines = {
        "condition": {"model": model_name, "task": task_name, "method": method},
        "predictors": {
            "audit_score": audit_bundle["audit_scores"]["audit_score"],
            **audit_bundle["baselines"],
        },
        "targets": safety_regression,
        "external_targets": external_eval["regression"] if external_eval else {},
        "base_safety_metrics": base_safety_metrics,
        "updated_safety_metrics": safety["metrics"],
    }

    save_eval_json(run_dir / "utility_metrics.json", utility["metrics"])
    save_eval_json(run_dir / "safety_metrics.json", safety["metrics"])
    save_audit_json(run_dir / "audit_scores.json", audit_bundle["audit_scores"])
    save_audit_json(run_dir / "layer_drift.json", audit_bundle["layer_drift"])
    save_audit_json(run_dir / "audit_vs_baselines.json", audit_vs_baselines)
    if external_eval:
        save_benchmark_json(run_dir / "external_benchmarks.json", external_eval)

    manifest = {
        "model": model_name,
        "task": task_name,
        "method": method,
        "run_dir": str(run_dir),
        "training": training_bundle["metadata"],
    }
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(manifest), handle, indent=2)

    _cleanup_model(model)
    return audit_vs_baselines


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_summary_table(output_root: Path) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "cache":
            continue
        utility_path = run_dir / "utility_metrics.json"
        safety_path = run_dir / "safety_metrics.json"
        audit_path = run_dir / "audit_scores.json"
        baseline_path = run_dir / "audit_vs_baselines.json"
        manifest_path = run_dir / "run_manifest.json"
        if not all(path.exists() for path in [utility_path, safety_path, audit_path, baseline_path, manifest_path]):
            continue

        utility = _load_json(utility_path)
        safety = _load_json(safety_path)
        audit = _load_json(audit_path)
        baseline = _load_json(baseline_path)
        manifest = _load_json(manifest_path)

        row = {
            "run_dir_name": run_dir.name,
            "run_dir": str(run_dir),
            "model": manifest["model"],
            "task": manifest["task"],
            "method": manifest["method"],
            **utility,
            **safety,
            "audit_score": audit["audit_score"],
            "canary_refusal_rate": audit["canary_refusal_rate"],
            "refusal_consistency": audit["refusal_consistency"],
            "late_layer_safety_drift": audit["late_layer_safety_drift"],
            "safety_specificity": audit["safety_specificity"],
            **baseline["predictors"],
            **baseline["targets"],
        }
        external_path = run_dir / "external_benchmarks.json"
        if external_path.exists():
            external = _load_json(external_path)
            row.update(external.get("metrics", {}))
            row.update(external.get("regression", {}))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "summary_table.csv", index=False)
    return summary


def _safe_correlation(xs: Sequence[float], ys: Sequence[float], fn):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 2:
        return float("nan"), float("nan"), int(len(xs))
    try:
        stat, pvalue = fn(xs, ys)
        return float(stat), float(pvalue), int(len(xs))
    except Exception:
        return float("nan"), float("nan"), int(len(xs))


def build_correlation_table(output_root: Path) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "cache":
            continue
        baseline_path = run_dir / "audit_vs_baselines.json"
        if not baseline_path.exists():
            continue
        payload = _load_json(baseline_path)
        predictors = payload["predictors"]
        targets = payload["targets"]
        for predictor_name, predictor_value in predictors.items():
            for target_name, target_value in targets.items():
                records.append(
                    {
                        "predictor": predictor_name,
                        "predictor_value": predictor_value,
                        "target": target_name,
                        "target_value": target_value,
                    }
                )

    rows: List[Dict[str, object]] = []
    df = pd.DataFrame(records)
    if not df.empty:
        for predictor in sorted(df["predictor"].unique()):
            for target in sorted(df["target"].unique()):
                subset = df[(df["predictor"] == predictor) & (df["target"] == target)]
                pearson_r, pearson_p, n = _safe_correlation(subset["predictor_value"], subset["target_value"], pearsonr)
                spearman_rho, spearman_p, _ = _safe_correlation(
                    subset["predictor_value"], subset["target_value"], spearmanr
                )
                rows.append(
                    {
                        "predictor": predictor,
                        "target": target,
                        "n": n,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                    }
                )
    correlation_table = pd.DataFrame(rows)
    correlation_table.to_csv(output_root / "audit_correlation.csv", index=False)
    return correlation_table


def main() -> None:
    args = parse_args()
    apply_runtime_overrides(args)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = resolve_conditions(args)

    if not args.skip_finetune:
        for model_name, task_name, method in tqdm(conditions, desc="UpgradeGuard conditions"):
            run_dir = run_dir_for(output_root, model_name, task_name, method)
            try:
                run_condition(args, model_name, task_name, method)
            except Exception as exc:
                run_dir.mkdir(parents=True, exist_ok=True)
                with (run_dir / "error.json").open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "model": model_name,
                            "task": task_name,
                            "method": method,
                            "error": str(exc),
                        },
                        handle,
                        indent=2,
                    )
                _cleanup_model(None)

    if args.backfill_external_validation:
        status_rows = backfill_external_benchmarks_for_saved_runs(
            output_root=output_root,
            device=args.device,
            include_strongreject=args.include_strongreject,
            harmbench_samples=args.harmbench_samples,
            xstest_samples=args.xstest_samples,
            strongreject_samples=args.strongreject_samples,
        )
        pd.DataFrame(status_rows).to_csv(output_root / "external_backfill_status.csv", index=False)

    summary = build_summary_table(output_root)
    build_correlation_table(output_root)
    if not args.skip_posthoc:
        build_posthoc_artifacts(summary, output_root)


if __name__ == "__main__":
    main()
