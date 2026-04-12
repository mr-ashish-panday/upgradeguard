from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["partial_unfreeze", "lora", "qlora", "full_ft"]
METHOD_LABELS = {
    "partial_unfreeze": "Partial unfreeze",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "full_ft": "FullFT",
}
ROOT_PRIORITY_HINTS = (
    "h100_qwen_summarization_seeded",
    "h100_conditioned_method_panels",
    "lightning_conditioned_panels",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and plot seeded Qwen summarization results.")
    parser.add_argument("--seeded-root", required=True, help="Root directory containing seeded Qwen summarization runs.")
    parser.add_argument(
        "--canonical-root",
        action="append",
        default=[],
        help="Optional root(s) containing the canonical seed-42 Qwen summarization panel runs.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for aggregate CSVs and seeded-panel plots.")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of bootstrap samples used for descriptive percentile intervals.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _utility_payload(run_dir: Path) -> Tuple[str, float]:
    payload = _load_json(run_dir / "utility_metrics.json")
    candidates = [
        (key, value)
        for key, value in payload.items()
        if key not in {"task", "num_samples"} and isinstance(value, (int, float))
    ]
    if not candidates:
        raise ValueError(f"No numeric utility metric found in {run_dir / 'utility_metrics.json'}")
    metric_name, metric_value = candidates[0]
    return str(metric_name), float(metric_value)


def _run_priority(path: Path) -> Tuple[int, int]:
    text = str(path).lower()
    for idx, hint in enumerate(ROOT_PRIORITY_HINTS):
        if hint in text:
            return (len(ROOT_PRIORITY_HINTS) - idx, -len(text))
    return (0, -len(text))


def _iter_qwen_summarization_runs(root: Path, source_label: str) -> Iterable[Dict[str, object]]:
    if not root.exists():
        return []

    rows: List[Dict[str, object]] = []
    for manifest_path in root.rglob("run_manifest.json"):
        run_dir = manifest_path.parent
        manifest = _load_json(manifest_path)
        model = str(manifest.get("model", ""))
        task = str(manifest.get("task", ""))
        method = str(manifest.get("method", ""))
        if "Qwen/Qwen2.5-7B-Instruct" not in model or task != "summarization" or method not in METHOD_ORDER:
            continue

        required = [
            run_dir / "utility_metrics.json",
            run_dir / "audit_scores.json",
            run_dir / "external_benchmarks.json",
            run_dir / "training_summary.json",
        ]
        if not all(path.exists() for path in required):
            continue

        utility_metric, utility_value = _utility_payload(run_dir)
        audit_scores = _load_json(run_dir / "audit_scores.json")
        external = _load_json(run_dir / "external_benchmarks.json")
        training_summary = _load_json(run_dir / "training_summary.json")

        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": int(training_summary["seed"]),
                "utility_metric": utility_metric,
                "utility_value": float(utility_value),
                "audit_score": float(audit_scores["audit_score"]),
                "external_regression": float(
                    external["regression"]["external_composite_safety_regression"]
                ),
                "run_dir": str(run_dir),
                "source": source_label,
                "priority_a": _run_priority(run_dir)[0],
                "priority_b": _run_priority(run_dir)[1],
            }
        )
    return rows


def _deduplicate(rows: List[Dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        by=["method", "seed", "priority_a", "priority_b", "source", "run_dir"],
        ascending=[True, True, False, False, True, True],
    )
    frame = frame.drop_duplicates(subset=["method", "seed"], keep="first").copy()
    frame = frame.drop(columns=["priority_a", "priority_b"])
    return frame.sort_values(by=["method", "seed"]).reset_index(drop=True)


def _bootstrap_interval(values: np.ndarray, n_bootstrap: int) -> Tuple[float, float]:
    if len(values) < 2:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(42)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [5, 95])
    return float(lo), float(hi)


def build_summary(frame: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        subset = frame[frame["method"] == method].copy()
        if subset.empty:
            continue
        ext_values = subset["external_regression"].to_numpy(dtype=float)
        audit_values = subset["audit_score"].to_numpy(dtype=float)
        utility_values = subset["utility_value"].to_numpy(dtype=float)
        ext_lo, ext_hi = _bootstrap_interval(ext_values, bootstrap_samples)
        audit_lo, audit_hi = _bootstrap_interval(audit_values, bootstrap_samples)
        util_lo, util_hi = _bootstrap_interval(utility_values, bootstrap_samples)
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n": int(len(subset)),
                "utility_metric": subset["utility_metric"].iloc[0],
                "external_regression_mean": float(ext_values.mean()),
                "external_regression_std": float(ext_values.std(ddof=0)),
                "external_regression_p05_boot": ext_lo,
                "external_regression_p95_boot": ext_hi,
                "audit_score_mean": float(audit_values.mean()),
                "audit_score_std": float(audit_values.std(ddof=0)),
                "audit_score_p05_boot": audit_lo,
                "audit_score_p95_boot": audit_hi,
                "utility_mean": float(utility_values.mean()),
                "utility_std": float(utility_values.std(ddof=0)),
                "utility_p05_boot": util_lo,
                "utility_p95_boot": util_hi,
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_win_rates(frame: pd.DataFrame) -> pd.DataFrame:
    qlora = frame[frame["method"] == "qlora"].copy()
    full_ft = frame[frame["method"] == "full_ft"].copy()
    common_seeds = sorted(set(qlora["seed"]).intersection(set(full_ft["seed"])))
    matched_records = []
    for seed in common_seeds:
        q_val = float(qlora.loc[qlora["seed"] == seed, "external_regression"].iloc[0])
        f_val = float(full_ft.loc[full_ft["seed"] == seed, "external_regression"].iloc[0])
        matched_records.append(
            {
                "seed": seed,
                "qlora_external_regression": q_val,
                "fullft_external_regression": f_val,
                "qlora_gt_fullft": float(q_val > f_val),
                "qlora_ge_fullft": float(q_val >= f_val),
                "difference": q_val - f_val,
            }
        )

    all_pair_records = []
    for _, q_row in qlora.iterrows():
        for _, f_row in full_ft.iterrows():
            all_pair_records.append(
                {
                    "qlora_seed": int(q_row["seed"]),
                    "fullft_seed": int(f_row["seed"]),
                    "qlora_gt_fullft": float(q_row["external_regression"] > f_row["external_regression"]),
                    "qlora_ge_fullft": float(q_row["external_regression"] >= f_row["external_regression"]),
                    "difference": float(q_row["external_regression"] - f_row["external_regression"]),
                }
            )

    matched_frame = pd.DataFrame(matched_records)
    all_pairs_frame = pd.DataFrame(all_pair_records)

    summary_rows = [
        {
            "comparison": "QLoRA vs FullFT (matched seeds)",
            "num_comparisons": int(len(matched_frame)),
            "qlora_gt_fullft_rate": float(matched_frame["qlora_gt_fullft"].mean()) if not matched_frame.empty else math.nan,
            "qlora_ge_fullft_rate": float(matched_frame["qlora_ge_fullft"].mean()) if not matched_frame.empty else math.nan,
            "mean_difference": float(matched_frame["difference"].mean()) if not matched_frame.empty else math.nan,
        },
        {
            "comparison": "QLoRA vs FullFT (all seed pairs)",
            "num_comparisons": int(len(all_pairs_frame)),
            "qlora_gt_fullft_rate": float(all_pairs_frame["qlora_gt_fullft"].mean()) if not all_pairs_frame.empty else math.nan,
            "qlora_ge_fullft_rate": float(all_pairs_frame["qlora_ge_fullft"].mean()) if not all_pairs_frame.empty else math.nan,
            "mean_difference": float(all_pairs_frame["difference"].mean()) if not all_pairs_frame.empty else math.nan,
        },
    ]
    return pd.DataFrame(summary_rows), matched_frame, all_pairs_frame


def build_worst_method_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    common_seeds = sorted(
        set.intersection(*[set(frame.loc[frame["method"] == method, "seed"]) for method in METHOD_ORDER if not frame.loc[frame["method"] == method].empty])
    )
    rows: List[Dict[str, object]] = []
    counts = {method: 0.0 for method in METHOD_ORDER}
    for seed in common_seeds:
        subset = frame[frame["seed"] == seed].copy()
        max_value = subset["external_regression"].max()
        winners = subset[np.isclose(subset["external_regression"], max_value)]["method"].tolist()
        share = 1.0 / len(winners)
        for winner in winners:
            counts[winner] += share
        rows.append(
            {
                "seed": seed,
                "worst_method": ", ".join(METHOD_LABELS[w] for w in winners),
                "max_external_regression": float(max_value),
            }
        )

    frequency_rows = [
        {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "frequency": counts[method] / len(common_seeds) if common_seeds else math.nan,
            "num_common_seeds": len(common_seeds),
        }
        for method in METHOD_ORDER
    ]
    return pd.DataFrame(rows), pd.DataFrame(frequency_rows)


def plot_seed_points(frame: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    metric_specs = [
        ("external_regression", "External safety regression", "external_regression_mean"),
        ("audit_score", "Audit score", "audit_score_mean"),
        ("utility_value", f"Utility ({summary['utility_metric'].iloc[0]})", "utility_mean"),
    ]

    for axis, (column, title, summary_col) in zip(axes, metric_specs):
        for idx, method in enumerate(METHOD_ORDER):
            subset = frame[frame["method"] == method].copy()
            if subset.empty:
                continue
            offsets = np.linspace(-0.12, 0.12, num=len(subset))
            x_vals = np.full(len(subset), idx, dtype=float) + offsets
            axis.scatter(
                x_vals,
                subset[column],
                alpha=0.85,
                s=44,
                edgecolors="black",
                linewidths=0.4,
                label=METHOD_LABELS[method] if axis is axes[0] else None,
            )
            mean_value = float(summary.loc[summary["method"] == method, summary_col].iloc[0])
            axis.hlines(mean_value, idx - 0.22, idx + 0.22, colors="black", linewidth=2.2)
        axis.set_title(title)
        axis.set_xticks(range(len(METHOD_ORDER)))
        axis.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
        axis.grid(True, axis="y", alpha=0.25)

    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Seeded Qwen summarization follow-up", fontsize=14)
    fig.savefig(output_dir / "figure_qwen_summarization_seeded.pdf")
    fig.savefig(output_dir / "figure_qwen_summarization_seeded.png", dpi=220)
    plt.close(fig)


def write_markdown_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    worst_frequency: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Seeded Qwen Summarization Report")
    lines.append("")
    lines.append("This report aggregates the seeded Qwen summarization reruns requested for the workshop/arXiv upgrade.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    for method in METHOD_ORDER:
        subset = frame[frame["method"] == method]
        if subset.empty:
            continue
        lines.append(f"- `{METHOD_LABELS[method]}`: `{len(subset)}` total seeds")
    lines.append("")
    lines.append("## Method Summary")
    lines.append("")
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['method_label']}`: ext reg mean `{row['external_regression_mean']:.4f}`, "
            f"audit mean `{row['audit_score_mean']:.4f}`, "
            f"{row['utility_metric']} mean `{row['utility_mean']:.4f}`"
        )
    lines.append("")
    lines.append("## Disputed Pair")
    lines.append("")
    for _, row in pairwise_summary.iterrows():
        lines.append(
            f"- `{row['comparison']}`: QLoRA > FullFT rate `{row['qlora_gt_fullft_rate']:.3f}`, "
            f"QLoRA >= FullFT rate `{row['qlora_ge_fullft_rate']:.3f}`, "
            f"mean ext-reg difference `{row['mean_difference']:.4f}` over `{int(row['num_comparisons'])}` comparisons"
        )
    lines.append("")
    lines.append("## Worst-Method Frequency")
    lines.append("")
    for _, row in worst_frequency.iterrows():
        lines.append(
            f"- `{row['method_label']}`: worst-method frequency `{row['frequency']:.3f}` "
            f"across `{int(row['num_common_seeds'])}` common seeds"
        )
    lines.append("")
    (output_dir / "qwen_summarization_seeded_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    rows.extend(_iter_qwen_summarization_runs(Path(args.seeded_root), "seeded"))
    for root in args.canonical_root:
        rows.extend(_iter_qwen_summarization_runs(Path(root), "canonical"))

    frame = _deduplicate(rows)
    if frame.empty:
        raise SystemExit("No Qwen summarization runs found for aggregation.")

    summary = build_summary(frame, args.bootstrap_samples)
    pairwise_summary, pairwise_matched, pairwise_all = build_pairwise_win_rates(frame)
    worst_by_seed, worst_frequency = build_worst_method_frequency(frame)

    frame.to_csv(output_dir / "qwen_summarization_seed_points.csv", index=False)
    summary.to_csv(output_dir / "qwen_summarization_seed_summary.csv", index=False)
    pairwise_summary.to_csv(output_dir / "qwen_summarization_pairwise_win_rates.csv", index=False)
    pairwise_matched.to_csv(output_dir / "qwen_summarization_pairwise_matched.csv", index=False)
    pairwise_all.to_csv(output_dir / "qwen_summarization_pairwise_all_pairs.csv", index=False)
    worst_by_seed.to_csv(output_dir / "qwen_summarization_worst_by_seed.csv", index=False)
    worst_frequency.to_csv(output_dir / "qwen_summarization_worst_frequency.csv", index=False)

    plot_seed_points(frame, summary, output_dir)
    write_markdown_report(frame, summary, pairwise_summary, worst_frequency, output_dir)


if __name__ == "__main__":
    main()
