from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    from upgradeguard.robustness import decoding_regimes

    parser = argparse.ArgumentParser(description="Run external benchmarks for a saved run under a specific decode regime.")
    parser.add_argument("--run-dir", required=True, help="Saved run directory with model_artifacts and run_manifest.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for decode-regime external outputs.")
    parser.add_argument(
        "--cache-root",
        help="Optional shared cache/output root for benchmark CSVs and base-model reference metrics.",
    )
    parser.add_argument("--decode-regime", choices=sorted(decoding_regimes()), default="greedy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-strongreject", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from upgradeguard.benchmarks import build_external_eval_payload, load_saved_run_model, save_json
    from upgradeguard.finetune import get_torch_device
    from upgradeguard.robustness import decoding_regimes

    device = get_torch_device(args.device)
    model, tokenizer, manifest = load_saved_run_model(args.run_dir, device=device)
    if model is None or tokenizer is None or manifest is None:
        raise RuntimeError(f"Unable to load saved run from {args.run_dir}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root) if args.cache_root else output_dir
    payload = build_external_eval_payload(
        model_name=str(manifest["model"]),
        model=model,
        tokenizer=tokenizer,
        output_root=cache_root,
        device=device,
        include_strongreject=args.include_strongreject,
        generation_kwargs=decoding_regimes()[args.decode_regime],
    )
    save_json(output_dir / f"external_benchmarks_{args.decode_regime}.json", payload)


if __name__ == "__main__":
    main()
