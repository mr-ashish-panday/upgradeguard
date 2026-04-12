from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate token-form drift on a saved run.")
    parser.add_argument("--run-dir", required=True, help="Saved run directory with model_artifacts and run_manifest.json.")
    parser.add_argument(
        "--prompt-bank",
        choices=["canaries", "safety_probes"],
        default="canaries",
        help="Prompt source to perturb.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for token-form drift outputs.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _records_for_bank(prompt_bank: str) -> List[Dict[str, object]]:
    from upgradeguard import canaries, probes

    if prompt_bank == "canaries":
        return [dict(item) for item in canaries.get_canary_prompts()]
    return [dict(item) for item in probes.SAFETY_PROBES]


def main() -> None:
    args = parse_args()
    from upgradeguard.audit import generate_responses
    from upgradeguard.benchmarks import load_saved_run_model
    from upgradeguard.finetune import get_torch_device
    from upgradeguard.metrics import classify_refusal
    from upgradeguard.robustness import build_token_form_drift_records

    device = get_torch_device(args.device)
    model, tokenizer, manifest = load_saved_run_model(args.run_dir, device=device)
    if model is None or tokenizer is None:
        raise RuntimeError(f"Unable to load saved run from {args.run_dir}.")

    records = build_token_form_drift_records(_records_for_bank(args.prompt_bank))
    responses = generate_responses(
        model,
        tokenizer,
        [str(record["prompt"]) for record in records],
        device,
        desc=f"Token-form drift ({args.prompt_bank})",
    )

    output_rows: List[Dict[str, object]] = []
    for record, response in zip(records, responses):
        output_rows.append(
            {
                "model": manifest.get("model"),
                "task": manifest.get("task"),
                "method": manifest.get("method"),
                "prompt_bank": args.prompt_bank,
                "id": record.get("id"),
                "family": record.get("category", record.get("label", "unknown")),
                "variant": record.get("variant", "canonical"),
                "source_prompt": record.get("source_prompt", record.get("prompt")),
                "prompt": record.get("prompt"),
                "response": response,
                "refused": bool(classify_refusal(response)),
            }
        )

    dataframe = pd.DataFrame(output_rows)
    summary = (
        dataframe.groupby(["prompt_bank", "family", "variant"], dropna=False)["refused"]
        .mean()
        .reset_index()
        .rename(columns={"refused": "refusal_rate"})
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_dir / "token_form_drift_outputs.csv", index=False)
    summary.to_csv(output_dir / "token_form_drift_summary.csv", index=False)
    with (output_dir / "token_form_drift_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_dir": str(Path(args.run_dir).resolve()),
                "model": manifest.get("model"),
                "task": manifest.get("task"),
                "method": manifest.get("method"),
                "prompt_bank": args.prompt_bank,
                "rows": len(output_rows),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
