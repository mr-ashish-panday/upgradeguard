from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from upgradeguard.posthoc import build_posthoc_artifacts


REQUIRED_FILES = (
    "utility_metrics.json",
    "safety_metrics.json",
    "audit_scores.json",
    "audit_vs_baselines.json",
    "run_manifest.json",
)


def _training_signature(training: Dict[str, object]) -> str:
    return "|".join(
        [
            f"bs={training.get('batch_size', 'na')}",
            f"lr={training.get('learning_rate', 'na')}",
            f"ep={training.get('epochs', 'na')}",
            f"seed={training.get('seed', 'na')}",
        ]
    )


def _infer_run_variant(method: str, run_dir: Path) -> str:
    run_dir_text = str(run_dir).lower()
    if "optimized" in run_dir_text:
        return "optimized"
    if method == "full_ft":
        return "standard"
    return "standard"


def discover_enriched_summaries(roots: Sequence[str]) -> List[Path]:
    candidates: List[Path] = []
    seen: set[str] = set()
    search_roots = [REPO_ROOT / "results", *(Path(root) for root in roots)]
    for root in search_roots:
        if not root.exists():
            continue
        for summary_path in root.rglob("summary_table_enriched.csv"):
            key = str(summary_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(summary_path)
    return sorted(candidates)


def attach_existing_task_similarity(summary: pd.DataFrame, roots: Sequence[str]) -> pd.DataFrame:
    if summary.empty:
        return summary

    enriched_sources: List[pd.DataFrame] = []
    required = {"task_similarity_risk", "model", "task"}
    for summary_path in discover_enriched_summaries(roots):
        try:
            source = pd.read_csv(summary_path)
        except Exception:
            continue
        if not required.issubset(source.columns):
            continue
        keep_columns = [
            column
            for column in (
                "run_dir",
                "run_dir_name",
                "model",
                "task",
                "method",
                "run_variant",
                "training_signature",
                "task_similarity_risk",
            )
            if column in source.columns
        ]
        if "task_similarity_risk" not in keep_columns:
            continue
        trimmed = source[keep_columns].dropna(subset=["task_similarity_risk"]).copy()
        if trimmed.empty:
            continue
        enriched_sources.append(trimmed)

    if not enriched_sources:
        return summary

    enriched = summary.copy()
    if "task_similarity_risk" not in enriched.columns:
        enriched["task_similarity_risk"] = pd.NA

    lookup = pd.concat(enriched_sources, ignore_index=True)

    def _merge_fill(target: pd.DataFrame, keys: List[str], fill_column: str) -> pd.DataFrame:
        usable = [key for key in keys if key in target.columns and key in lookup.columns]
        if not usable:
            return target
        candidates = (
            lookup.dropna(subset=usable + ["task_similarity_risk"])
            .drop_duplicates(subset=usable, keep="first")[usable + ["task_similarity_risk"]]
            .rename(columns={"task_similarity_risk": fill_column})
        )
        merged = target.merge(candidates, on=usable, how="left")
        merged["task_similarity_risk"] = merged["task_similarity_risk"].fillna(merged[fill_column])
        return merged.drop(columns=[fill_column])

    for keys in (
        ["run_dir"],
        ["run_dir_name"],
        ["model", "task", "method", "run_variant", "training_signature"],
        ["model", "task", "method", "run_variant"],
        ["model", "task", "method"],
        ["model", "task"],
        ["task"],
    ):
        fill_column = "__task_similarity_fill__"
        enriched = _merge_fill(enriched, keys, fill_column)
        if enriched["task_similarity_risk"].notna().all():
            break

    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a master UpgradeGuard results sheet from multiple roots.")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="One or more directories to scan recursively for completed run directories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the consolidated summary and post-hoc outputs should be written.",
    )
    parser.add_argument(
        "--exclude-substring",
        action="append",
        default=["remote_log_snapshots", "__pycache__", "cache"],
        help="Exclude run directories whose path contains this substring. May be passed multiple times.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_run_dirs(roots: Sequence[str], exclude_substrings: Sequence[str]) -> List[Path]:
    run_dirs: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        for manifest_path in Path(root).rglob("run_manifest.json"):
            run_dir = manifest_path.parent
            run_dir_str = str(run_dir)
            if any(fragment in run_dir_str for fragment in exclude_substrings):
                continue
            if not all((run_dir / filename).exists() for filename in REQUIRED_FILES):
                continue
            key = str(run_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def build_summary_from_run_dirs(run_dirs: Iterable[Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        utility = _load_json(run_dir / "utility_metrics.json")
        safety = _load_json(run_dir / "safety_metrics.json")
        audit = _load_json(run_dir / "audit_scores.json")
        baseline = _load_json(run_dir / "audit_vs_baselines.json")
        manifest = _load_json(run_dir / "run_manifest.json")
        stronger_path = run_dir / "stronger_baselines.json"
        stronger = _load_json(stronger_path) if stronger_path.exists() else {}
        predictors = dict(baseline.get("predictors", {}))
        predictors.update(stronger)
        training = dict(manifest.get("training", {}))
        run_variant = _infer_run_variant(str(manifest["method"]), run_dir)
        update_label = (
            f"{manifest['method']}_{run_variant}"
            if run_variant not in {"", "standard"}
            else str(manifest["method"])
        )

        row = {
            "run_dir_name": run_dir.name,
            "run_dir": str(run_dir.resolve()),
            "model": manifest["model"],
            "task": manifest["task"],
            "method": manifest["method"],
            "run_variant": run_variant,
            "update_label": update_label,
            "training_signature": _training_signature(training),
            "train_batch_size": training.get("batch_size"),
            "train_learning_rate": training.get("learning_rate"),
            "train_epochs": training.get("epochs"),
            "train_seed": training.get("seed"),
            **utility,
            **safety,
            **{key: value for key, value in audit.items() if key != "raw_canary_outputs"},
            **predictors,
            **baseline.get("targets", {}),
        }

        external_path = run_dir / "external_benchmarks.json"
        if external_path.exists():
            external = _load_json(external_path)
            row.update(external.get("metrics", {}))
            row.update(external.get("regression", {}))

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(args.roots, args.exclude_substring)
    inventory = pd.DataFrame({"run_dir": [str(path.resolve()) for path in run_dirs]})
    inventory.to_csv(output_dir / "run_inventory.csv", index=False)

    summary = build_summary_from_run_dirs(run_dirs)
    summary = attach_existing_task_similarity(summary, args.roots)
    summary.to_csv(output_dir / "summary_table_raw.csv", index=False)
    build_posthoc_artifacts(summary, output_dir)


if __name__ == "__main__":
    main()
