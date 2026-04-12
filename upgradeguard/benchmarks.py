from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence
from urllib.request import urlretrieve

import pandas as pd
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM

from upgradeguard import config
from upgradeguard.audit import generate_responses
from upgradeguard.finetune import get_torch_device, get_torch_dtype, load_tokenizer
from upgradeguard.metrics import (
    classify_refusal,
    compute_external_safety_regression,
    compute_refusal_rate,
    to_serializable,
)

HARM_BENCH_TEST_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_test.csv"
)
HARM_BENCH_VAL_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_val.csv"
)
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
STRONGREJECT_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_small_dataset.csv"
)


def _benchmark_cache_dir(output_root: str | Path) -> Path:
    cache_dir = config.cache_root(output_root) / "benchmarks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_csv(output_root: str | Path, name: str, url: str) -> Path:
    path = _benchmark_cache_dir(output_root) / f"{name}.csv"
    if not path.exists():
        urlretrieve(url, path)
    return path


def _external_cache_path(
    output_root: str | Path,
    model_name: str,
    include_strongreject: bool,
    harmbench_samples: int | None,
    xstest_samples: int | None,
    strongreject_samples: int | None,
) -> Path:
    filename = (
        f"base_external_metrics_hb{harmbench_samples or 'all'}"
        f"_xs{xstest_samples or 'all'}"
        f"_sr{strongreject_samples or 0 if include_strongreject else 0}.json"
    )
    path = config.cache_root(output_root) / config.slugify_model_name(model_name) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_cached_csv(output_root: str | Path, name: str, url: str) -> pd.DataFrame:
    return pd.read_csv(_cache_csv(output_root, name, url))


def _limit_records(records: List[Dict[str, object]], limit: int | None) -> List[Dict[str, object]]:
    if limit is None or limit <= 0 or limit >= len(records):
        return records
    return records[:limit]


def _balanced_limit(records: List[Dict[str, object]], limit: int | None) -> List[Dict[str, object]]:
    if limit is None or limit <= 0 or limit >= len(records):
        return records
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["label"]), []).append(record)
    labels = sorted(grouped)
    target_per_label = max(1, limit // max(1, len(labels)))
    selected: List[Dict[str, object]] = []
    for label in labels:
        selected.extend(grouped[label][:target_per_label])
    if len(selected) < limit:
        leftovers: List[Dict[str, object]] = []
        for label in labels:
            leftovers.extend(grouped[label][target_per_label:])
        selected.extend(leftovers[: limit - len(selected)])
    return selected[:limit]


def load_harmbench_records(output_root: str | Path, split: str = "test", limit: int | None = None) -> List[Dict[str, object]]:
    if split not in {"test", "val"}:
        raise ValueError("HarmBench split must be 'test' or 'val'.")
    url = HARM_BENCH_TEST_URL if split == "test" else HARM_BENCH_VAL_URL
    dataframe = _load_cached_csv(output_root, f"harmbench_{split}", url)
    records: List[Dict[str, object]] = []
    for row in dataframe.to_dict(orient="records"):
        context = str(row.get("ContextString") or "").strip()
        behavior = str(row.get("Behavior") or "").strip()
        prompt = behavior if not context else f"{context}\n\nRequest:\n{behavior}"
        records.append(
            {
                "id": str(row.get("BehaviorID") or len(records)),
                "benchmark": "harmbench",
                "label": "unsafe",
                "family": str(row.get("SemanticCategory") or "unspecified"),
                "prompt": prompt,
            }
        )
    return _limit_records(records, limit)


def load_xstest_records(output_root: str | Path, limit: int | None = None) -> List[Dict[str, object]]:
    dataframe = _load_cached_csv(output_root, "xstest", XSTEST_URL)
    records: List[Dict[str, object]] = []
    for row in dataframe.to_dict(orient="records"):
        records.append(
            {
                "id": str(row.get("id") or len(records)),
                "benchmark": "xstest",
                "label": str(row.get("label") or "safe"),
                "family": str(row.get("type") or "unspecified"),
                "focus": str(row.get("focus") or ""),
                "prompt": str(row.get("prompt") or "").strip(),
            }
        )
    return _balanced_limit(records, limit)


def load_strongreject_records(output_root: str | Path, limit: int | None = None) -> List[Dict[str, object]]:
    dataframe = _load_cached_csv(output_root, "strongreject_small", STRONGREJECT_URL)
    records: List[Dict[str, object]] = []
    for idx, row in enumerate(dataframe.to_dict(orient="records")):
        records.append(
            {
                "id": f"strongreject_{idx:03d}",
                "benchmark": "strongreject",
                "label": "unsafe",
                "family": str(row.get("category") or "unspecified"),
                "prompt": str(row.get("forbidden_prompt") or "").strip(),
            }
        )
    return _limit_records(records, limit)


def _load_base_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_torch_dtype(device),
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def _evaluate_prompt_bank(
    model,
    tokenizer,
    records: Sequence[Mapping[str, object]],
    device: str,
    desc: str,
    generation_kwargs: Mapping[str, object] | None = None,
) -> List[Dict[str, object]]:
    outputs = generate_responses(
        model,
        tokenizer,
        [str(record["prompt"]) for record in records],
        device,
        desc=desc,
        generation_kwargs=generation_kwargs,
    )
    return [
        {
            **dict(record),
            "response": output,
            "refused": classify_refusal(output),
        }
        for record, output in zip(records, outputs)
    ]


def evaluate_external_benchmarks(
    model,
    tokenizer,
    output_root: str | Path,
    device: str,
    include_strongreject: bool = False,
    harmbench_samples: int | None = None,
    xstest_samples: int | None = None,
    strongreject_samples: int | None = None,
    generation_kwargs: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    harmbench_records = load_harmbench_records(output_root, split="test", limit=harmbench_samples)
    xstest_records = load_xstest_records(output_root, limit=xstest_samples)
    strongreject_records = (
        load_strongreject_records(output_root, limit=strongreject_samples)
        if include_strongreject
        else []
    )

    harmbench_outputs = _evaluate_prompt_bank(
        model,
        tokenizer,
        harmbench_records,
        device,
        desc="External eval HarmBench",
        generation_kwargs=generation_kwargs,
    )
    xstest_outputs = _evaluate_prompt_bank(
        model,
        tokenizer,
        xstest_records,
        device,
        desc="External eval XSTest",
        generation_kwargs=generation_kwargs,
    )
    strongreject_outputs = (
        _evaluate_prompt_bank(
            model,
            tokenizer,
            strongreject_records,
            device,
            desc="External eval StrongREJECT",
            generation_kwargs=generation_kwargs,
        )
        if strongreject_records
        else []
    )

    xstest_safe = [row for row in xstest_outputs if row["label"] == "safe"]
    xstest_unsafe = [row for row in xstest_outputs if row["label"] == "unsafe"]
    metrics = {
        "harmbench_harmful_refusal_rate": compute_refusal_rate([row["response"] for row in harmbench_outputs]),
        "harmbench_attack_success_rate": 1.0 - compute_refusal_rate([row["response"] for row in harmbench_outputs]),
        "harmbench_num_samples": len(harmbench_outputs),
        "xstest_safe_overrefusal_rate": compute_refusal_rate([row["response"] for row in xstest_safe]),
        "xstest_unsafe_refusal_rate": compute_refusal_rate([row["response"] for row in xstest_unsafe]),
        "xstest_balanced_accuracy": float(
            (
                compute_refusal_rate([row["response"] for row in xstest_unsafe])
                + (1.0 - compute_refusal_rate([row["response"] for row in xstest_safe]))
            )
            / 2.0
        ),
        "xstest_num_samples": len(xstest_outputs),
    }
    if strongreject_outputs:
        strongreject_refusal_rate = compute_refusal_rate([row["response"] for row in strongreject_outputs])
        metrics.update(
            {
                "strongreject_refusal_rate": strongreject_refusal_rate,
                "strongreject_attack_success_rate": 1.0 - strongreject_refusal_rate,
                "strongreject_num_samples": len(strongreject_outputs),
            }
        )

    return {
        "metrics": metrics,
        "preview": {
            "harmbench_outputs": harmbench_outputs[:10],
            "xstest_outputs": xstest_outputs[:10],
            "strongreject_outputs": strongreject_outputs[:10],
        },
    }


def ensure_base_external_benchmark_metrics(
    model_name: str,
    output_root: str | Path,
    device: str | None = None,
    include_strongreject: bool = False,
    harmbench_samples: int | None = None,
    xstest_samples: int | None = None,
    strongreject_samples: int | None = None,
) -> Dict[str, float]:
    cache_path = _external_cache_path(
        output_root=output_root,
        model_name=model_name,
        include_strongreject=include_strongreject,
        harmbench_samples=harmbench_samples,
        xstest_samples=xstest_samples,
        strongreject_samples=strongreject_samples,
    )
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    resolved_device = get_torch_device(device)
    tokenizer = load_tokenizer(model_name)
    model = _load_base_model(model_name, resolved_device)
    try:
        metrics = evaluate_external_benchmarks(
            model=model,
            tokenizer=tokenizer,
            output_root=output_root,
            device=resolved_device,
            include_strongreject=include_strongreject,
            harmbench_samples=harmbench_samples,
            xstest_samples=xstest_samples,
            strongreject_samples=strongreject_samples,
        )["metrics"]
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(to_serializable(metrics), handle, indent=2)
        return metrics
    finally:
        del model


def materialize_base_external_benchmark_metrics(
    model_name: str,
    output_root: str | Path,
    device: str | None = None,
    include_strongreject: bool = False,
    harmbench_samples: int | None = None,
    xstest_samples: int | None = None,
    strongreject_samples: int | None = None,
) -> Dict[str, object]:
    metrics = ensure_base_external_benchmark_metrics(
        model_name=model_name,
        output_root=output_root,
        device=device,
        include_strongreject=include_strongreject,
        harmbench_samples=harmbench_samples,
        xstest_samples=xstest_samples,
        strongreject_samples=strongreject_samples,
    )
    payload = {
        "model": model_name,
        "metrics": metrics,
        "protocol": {
            "include_strongreject": include_strongreject,
            "harmbench_samples": harmbench_samples,
            "xstest_samples": xstest_samples,
            "strongreject_samples": strongreject_samples,
        },
    }
    save_json(
        Path(output_root) / f"base_external_{config.slugify_model_name(model_name)}.json",
        payload,
    )
    return payload


def build_external_eval_payload(
    model_name: str,
    model,
    tokenizer,
    output_root: str | Path,
    device: str,
    include_strongreject: bool = False,
    harmbench_samples: int | None = None,
    xstest_samples: int | None = None,
    strongreject_samples: int | None = None,
    generation_kwargs: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    updated = evaluate_external_benchmarks(
        model=model,
        tokenizer=tokenizer,
        output_root=output_root,
        device=device,
        include_strongreject=include_strongreject,
        harmbench_samples=harmbench_samples,
        xstest_samples=xstest_samples,
        strongreject_samples=strongreject_samples,
        generation_kwargs=generation_kwargs,
    )
    base_metrics = ensure_base_external_benchmark_metrics(
        model_name=model_name,
        output_root=output_root,
        device=device,
        include_strongreject=include_strongreject,
        harmbench_samples=harmbench_samples,
        xstest_samples=xstest_samples,
        strongreject_samples=strongreject_samples,
    )
    regression = compute_external_safety_regression(base_metrics, updated["metrics"])
    return {
        "metrics": updated["metrics"],
        "preview": updated["preview"],
        "base_metrics": base_metrics,
        "regression": regression,
    }


def load_saved_run_model(run_dir: str | Path, device: str | None = None):
    run_path = Path(run_dir)
    manifest_path = run_path / "run_manifest.json"
    model_dir = run_path / "model_artifacts"
    if not manifest_path.exists() or not model_dir.exists():
        return None, None, None

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    resolved_device = get_torch_device(device)
    tokenizer = None
    tokenizer_candidates = []
    if (model_dir / "tokenizer_config.json").exists():
        tokenizer_candidates.append(str(model_dir))
    tokenizer_candidates.append(str(manifest["model"]))
    tokenizer_error: Exception | None = None
    for tokenizer_source in tokenizer_candidates:
        try:
            tokenizer = load_tokenizer(tokenizer_source)
            break
        except Exception as exc:
            tokenizer_error = exc
    if tokenizer is None:
        raise RuntimeError(
            f"Unable to load tokenizer for saved run {run_path} from adapter dir or base model."
        ) from tokenizer_error

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=get_torch_dtype(resolved_device),
            low_cpu_mem_usage=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=get_torch_dtype(resolved_device),
            low_cpu_mem_usage=True,
        )
    if hasattr(model, "to"):
        try:
            model.to(resolved_device)
        except Exception:
            pass
    model.eval()
    return model, tokenizer, manifest


def backfill_external_benchmarks_for_saved_runs(
    output_root: str | Path,
    device: str | None = None,
    include_strongreject: bool = False,
    harmbench_samples: int | None = None,
    xstest_samples: int | None = None,
    strongreject_samples: int | None = None,
    run_dirs: Iterable[str | Path] | None = None,
) -> List[Dict[str, object]]:
    output_root = Path(output_root)
    statuses: List[Dict[str, object]] = []
    if run_dirs is None:
        candidate_run_dirs = sorted(output_root.iterdir())
    else:
        candidate_run_dirs = [Path(run_dir) for run_dir in run_dirs]
    for run_dir in candidate_run_dirs:
        if not run_dir.is_dir() or run_dir.name == "cache":
            continue
        external_path = run_dir / "external_benchmarks.json"
        if external_path.exists():
            statuses.append({"run_dir": str(run_dir), "status": "already_exists"})
            continue

        model, tokenizer, manifest = load_saved_run_model(run_dir, device=device)
        if model is None:
            statuses.append({"run_dir": str(run_dir), "status": "missing_saved_model"})
            continue
        try:
            payload = build_external_eval_payload(
                model_name=manifest["model"],
                model=model,
                tokenizer=tokenizer,
                output_root=output_root,
                device=get_torch_device(device),
                include_strongreject=include_strongreject,
                harmbench_samples=harmbench_samples,
                xstest_samples=xstest_samples,
                strongreject_samples=strongreject_samples,
            )
            save_json(external_path, payload)
            statuses.append({"run_dir": str(run_dir), "status": "evaluated"})
        except Exception as exc:
            statuses.append({"run_dir": str(run_dir), "status": "error", "error": str(exc)})
        finally:
            try:
                del model
            except Exception:
                pass
    return statuses


def save_json(path: str | Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(dict(payload)), handle, indent=2)
