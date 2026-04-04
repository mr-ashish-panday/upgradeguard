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

        row = {
            "run_dir_name": run_dir.name,
            "run_dir": str(run_dir.resolve()),
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
            **baseline.get("predictors", {}),
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
    summary.to_csv(output_dir / "summary_table.csv", index=False)
    build_posthoc_artifacts(summary, output_dir)


if __name__ == "__main__":
    main()
