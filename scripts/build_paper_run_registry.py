from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "paper_ready_final_refresh_20260407"
PULLS_DIR = ROOT / "pulls"
PANEL_REGISTRY = RESULTS_DIR / "paper_panel_registry.csv"
OUTPUT_CSV = RESULTS_DIR / "paper_run_registry_curated.csv"
OUTPUT_MD = RESULTS_DIR / "paper_run_registry_curated.md"
OUTPUT_TEX = RESULTS_DIR / "paper_run_registry_table.tex"


def nonempty(value: str | None) -> bool:
    return value is not None and value != ""


def parse_float(value: str | None) -> float | None:
    if not nonempty(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def relpath_str(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def short_prefix(model: str, task: str) -> str:
    model_lower = model.lower()
    if "qwen" in model_lower and task == "translation":
        return "QT"
    if "qwen" in model_lower and task == "summarization":
        return "QS"
    if "llama" in model_lower and task == "summarization":
        return "LS"
    if "qwen" in model_lower and task == "code_gen":
        return "QC"
    return task[:2].upper() if task else "RUN"


def short_method(method: str, run_variant: str) -> str:
    if method == "partial_unfreeze":
        return "PartUnf"
    if method == "full_ft" and "optimized" in run_variant:
        return "FullFT (Opt.)"
    mapping = {
        "lora": "LoRA",
        "qlora": "QLoRA",
        "full_ft": "FullFT",
    }
    return mapping.get(method, method)


def short_run_label(row: dict[str, str]) -> str:
    label = f"{short_prefix(row['model'], row['task'])} {short_method(row['method'], row['run_variant'])}"
    if row["run_class"] == "robustness_followup":
        return label + " robust"
    if row["run_class"] == "closure_rerun":
        return label + " closure"
    if row["run_class"] == "appendix_stress_test":
        return label + " stress"
    return label


def normalize_artifact_path(path: Path) -> Path:
    original = str(path)
    replacements = [
        (
            r"C:\Users\Ashish\all\Downloads\results (2)\h100_conditioned_method_panels",
            PULLS_DIR / "downloads_consolidated_20260410" / "results_2" / "h100_conditioned_method_panels",
        ),
        (
            r"C:\Users\Ashish\all\Downloads\results (2)\h100_qwen_translation_dense_optimized",
            PULLS_DIR / "downloads_consolidated_20260410" / "results_2" / "h100_qwen_translation_dense_optimized",
        ),
        (
            r"C:\Users\Ashish\all\Downloads\results (2)\h100_robustness_followup",
            PULLS_DIR / "downloads_consolidated_20260410" / "results_2" / "h100_robustness_followup",
        ),
        (
            r"C:\Users\Ashish\all\Downloads\results (1)\h100_conditioned_method_panels",
            PULLS_DIR / "downloads_consolidated_20260410" / "results_1" / "h100_conditioned_method_panels",
        ),
        (
            r"C:\Users\Ashish\all\Downloads\results (1)\h100_partial_unfreeze_repairs",
            PULLS_DIR / "downloads_consolidated_20260410" / "results_1" / "h100_partial_unfreeze_repairs",
        ),
        (
            r"C:\Users\Ashish\all\Downloads\results\h100_priority_external",
            PULLS_DIR / "downloads_consolidated_20260410" / "results" / "h100_priority_external",
        ),
    ]
    for old, new_base in replacements:
        if original.startswith(old):
            suffix = original[len(old):].lstrip("\\/")
            return new_base / Path(suffix)
    return path


def score_panel_row(row: dict[str, str]) -> tuple[int, int, int, int, int]:
    run_dir = row.get("run_dir", "")
    return (
        1 if nonempty(row.get("external_composite_safety_regression")) else 0,
        1 if nonempty(row.get("utility_score")) or nonempty(row.get("bleu")) or nonempty(row.get("rougeL")) else 0,
        1 if nonempty(row.get("audit_score")) else 0,
        1 if "Safety_paper_1\\pulls" in run_dir or "Safety_paper_1/pulls" in run_dir else 0,
        1 if nonempty(row.get("weight_spectral_score")) else 0,
    )


def clean_lr(value: str | None) -> str:
    if not nonempty(value):
        return ""
    try:
        return f"{float(value):.0e}"
    except ValueError:
        return str(value)


def load_canonical_rows() -> list[dict[str, str]]:
    with PANEL_REGISTRY.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        paper_role = row.get("paper_role", "")
        if paper_role not in {"main_panel", "supporting_panel", "transfer_panel", "appendix_stress_test"}:
            continue
        key = (
            row.get("model", ""),
            row.get("task", ""),
            row.get("method", ""),
            row.get("run_variant", "") or "standard",
            paper_role,
        )
        grouped.setdefault(key, []).append(row)

    canonical: list[dict[str, str]] = []
    for candidates in grouped.values():
        canonical.append(max(candidates, key=score_panel_row))

    return canonical


def canonical_entry(row: dict[str, str]) -> dict[str, str]:
    run_dir = normalize_artifact_path(Path(row["run_dir"]))
    return {
        "run_label": row["run_dir_name"],
        "model": row["model"],
        "task": row["task"],
        "method": row["method"],
        "run_variant": row.get("run_variant", "") or "standard",
        "paper_role": row.get("paper_role", ""),
        "run_class": "canonical_panel" if row.get("paper_role") != "appendix_stress_test" else "appendix_stress_test",
        "seed": row.get("train_seed", ""),
        "batch_size": row.get("train_batch_size", ""),
        "learning_rate": clean_lr(row.get("train_learning_rate", "")),
        "epochs": row.get("train_epochs", ""),
        "artifact_path": str(run_dir),
        "artifact_path_rel": relpath_str(run_dir),
        "utility_score": row.get("utility_score", "") or row.get("bleu", "") or row.get("rougeL", ""),
        "audit_score": row.get("audit_score", ""),
        "external_composite_safety_regression": row.get("external_composite_safety_regression", ""),
        "note": row.get("paper_role_rationale", ""),
    }


def make_lookup(entries: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for entry in entries:
        key = (
            entry["model"],
            entry["task"],
            entry["method"],
            entry["run_variant"],
        )
        out[key] = entry
    return out


def robustness_entries(canonical_lookup: dict[tuple[str, str, str, str], dict[str, str]]) -> list[dict[str, str]]:
    specs = [
        (
            "Qwen translation LoRA robustness followup",
            "Qwen/Qwen2.5-7B-Instruct",
            "translation",
            "lora",
            PULLS_DIR / "downloads_consolidated_20260410" / "results_2" / "h100_robustness_followup" / "qwen_translation_lora",
            "Greedy-vs-sampling external eval plus token-form drift followup on the clean reference panel run.",
        ),
        (
            "Qwen translation QLoRA robustness followup",
            "Qwen/Qwen2.5-7B-Instruct",
            "translation",
            "qlora",
            PULLS_DIR / "downloads_consolidated_20260410" / "results_2" / "h100_robustness_followup" / "qwen_translation_qlora",
            "Greedy-vs-sampling external eval plus token-form drift followup on the clean reference panel run.",
        ),
        (
            "Llama summarization QLoRA robustness followup",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "summarization",
            "qlora",
            PULLS_DIR / "downloads_consolidated_20260410" / "results_2" / "h100_robustness_followup" / "llama_summarization_qlora",
            "Robustness followup used to measure sampling fragility and token-form drift on the transfer panel.",
        ),
        (
            "Qwen summarization LoRA robustness followup",
            "Qwen/Qwen2.5-7B-Instruct",
            "summarization",
            "lora",
            PULLS_DIR / "h100_followup_20260406" / "h100_parallel_followup_remaining" / "qwen_summarization_lora",
            "Sampling external eval and token-form drift followup on the hard-panel LoRA run.",
        ),
        (
            "Qwen summarization QLoRA robustness followup",
            "Qwen/Qwen2.5-7B-Instruct",
            "summarization",
            "qlora",
            PULLS_DIR / "h100_followup_20260406" / "h100_parallel_followup_remaining" / "qwen_summarization_qlora",
            "Sampling external eval and token-form drift followup on the hard-panel QLoRA run.",
        ),
        (
            "Llama summarization LoRA robustness followup",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "summarization",
            "lora",
            PULLS_DIR / "h100_followup_20260406" / "h100_llama_lora_followup",
            "Late followup adding sampling external eval, token-form drift, and stronger baselines for the Llama LoRA transfer run.",
        ),
    ]

    rows: list[dict[str, str]] = []
    for label, model, task, method, artifact_path, note in specs:
        base = canonical_lookup[(model, task, method, "standard")]
        rows.append(
            {
                "run_label": label,
                "model": model,
                "task": task,
                "method": method,
                "run_variant": "standard",
                "paper_role": base["paper_role"],
                "run_class": "robustness_followup",
                "seed": base["seed"],
                "batch_size": base["batch_size"],
                "learning_rate": base["learning_rate"],
                "epochs": base["epochs"],
                "artifact_path": str(artifact_path),
                "artifact_path_rel": relpath_str(artifact_path),
                "utility_score": "",
                "audit_score": "",
                "external_composite_safety_regression": "",
                "note": note,
            }
        )
    return rows


def closure_entry(folder: Path, paper_role: str, note: str) -> dict[str, str]:
    training = json.loads((folder / "training_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((folder / "audit_scores.json").read_text(encoding="utf-8"))
    external = json.loads((folder / "external_benchmarks.json").read_text(encoding="utf-8"))
    utility = json.loads((folder / "utility_metrics.json").read_text(encoding="utf-8"))

    model = training["model_name"]
    task = training["task_name"]
    method = training["method"]
    utility_value = utility.get("bleu")
    if utility_value is None:
        utility_value = utility.get("rougeL")

    label = f"{model.split('/')[-1]} {task} {method} closure rerun"
    return {
        "run_label": label,
        "model": model,
        "task": task,
        "method": method,
        "run_variant": "closure_rerun",
        "paper_role": paper_role,
        "run_class": "closure_rerun",
        "seed": str(training["seed"]),
        "batch_size": str(training["batch_size"]),
        "learning_rate": clean_lr(str(training["learning_rate"])),
        "epochs": str(training["epochs"]),
        "artifact_path": str(folder),
        "artifact_path_rel": relpath_str(folder),
        "utility_score": f"{utility_value:.4f}" if utility_value is not None else "",
        "audit_score": f"{audit['audit_score']:.4f}" if nonempty(str(audit.get("audit_score"))) else "",
        "external_composite_safety_regression": f"{external['regression']['external_composite_safety_regression']:.4f}",
        "note": note,
    }


def closure_entries() -> list[dict[str, str]]:
    base = PULLS_DIR / "downloads_consolidated_20260410" / "dense_qwen_closure_jsons"
    return [
        closure_entry(
            base / "qwen_translation_full_ft",
            "main_panel",
            "Artifact-recovery rerun used to recover stronger baselines and dense checkpoints for the default Qwen translation FullFT anchor.",
        ),
        closure_entry(
            base / "qwen_summarization_full_ft",
            "supporting_panel",
            "Artifact-recovery rerun used to recover stronger baselines and dense checkpoints for the Qwen summarization FullFT anchor; executed with lower batch size after an OOM retry.",
        ),
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "run_label",
        "model",
        "task",
        "method",
        "run_variant",
        "paper_role",
        "run_class",
        "seed",
        "batch_size",
        "learning_rate",
        "epochs",
        "utility_score",
        "audit_score",
        "external_composite_safety_regression",
        "artifact_path",
        "artifact_path_rel",
        "note",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]]) -> None:
    class_counts: dict[str, int] = {}
    for row in rows:
        class_counts[row["run_class"]] = class_counts.get(row["run_class"], 0) + 1

    lines = [
        "# Paper Run Registry",
        "",
        "This file summarizes the paper-facing run registry used by the manuscript revision.",
        "",
        "## Class Counts",
        "",
    ]
    for key in sorted(class_counts):
        lines.append(f"- `{key}`: {class_counts[key]}")

    lines += [
        "",
        "## Registry Columns",
        "",
        "- `run_label`: human-readable name for the run artifact",
        "- `paper_role`: manuscript role (`main_panel`, `supporting_panel`, `transfer_panel`, `appendix_stress_test`)",
        "- `run_class`: provenance class (`canonical_panel`, `robustness_followup`, `closure_rerun`, `appendix_stress_test`)",
        "- `artifact_path_rel`: repo-relative path to the corresponding artifact folder",
        "",
        "## Notes",
        "",
        "- Canonical panel rows are deduplicated from `paper_panel_registry.csv` by preferring the most complete artifact record.",
        "- Robustness followups inherit training metadata from their linked canonical run.",
        "- Closure reruns are listed separately because they were used to recover missing dense checkpoints and stronger-baseline artifacts rather than to overwrite the canonical panel tables.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_tex(rows: list[dict[str, str]]) -> None:
    lines = [
        r"\tiny",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.08\linewidth}>{\raggedright\arraybackslash}p{0.07\linewidth}>{\centering\arraybackslash}p{0.04\linewidth}>{\centering\arraybackslash}p{0.04\linewidth}>{\centering\arraybackslash}p{0.06\linewidth}>{\centering\arraybackslash}p{0.04\linewidth}>{\raggedright\arraybackslash}p{0.33\linewidth}}",
        r"\toprule",
        r"Run & Class & Role & Seed & BS & LR & Ep & Artifact path \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Run & Class & Role & Seed & BS & LR & Ep & Artifact path \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]

    role_short = {
        "main_panel": "main",
        "supporting_panel": "supp.",
        "transfer_panel": "xfer",
        "appendix_stress_test": "appx.",
    }
    class_short = {
        "canonical_panel": "canon.",
        "robustness_followup": "robust.",
        "closure_rerun": "closure",
        "appendix_stress_test": "appx.",
    }

    order = {
        "canonical_panel": 0,
        "appendix_stress_test": 1,
        "robustness_followup": 2,
        "closure_rerun": 3,
    }
    rows_sorted = sorted(rows, key=lambda r: (order.get(r["run_class"], 99), r["paper_role"], r["task"], r["method"], r["run_label"]))

    for row in rows_sorted:
        lines.append(
            " & ".join(
                [
                    latex_escape(short_run_label(row)),
                    latex_escape(class_short.get(row["run_class"], row["run_class"])),
                    latex_escape(role_short.get(row["paper_role"], row["paper_role"])),
                    latex_escape(row["seed"]),
                    latex_escape(row["batch_size"]),
                    latex_escape(row["learning_rate"]),
                    latex_escape(row["epochs"]),
                    r"\path{" + row["artifact_path_rel"] + "}",
                ]
            )
            + r" \\"
        )

    lines.append(r"\end{longtable}")
    OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    canonical_rows = [canonical_entry(row) for row in load_canonical_rows()]
    canonical_lookup = make_lookup(canonical_rows)
    rows = canonical_rows + robustness_entries(canonical_lookup) + closure_entries()
    write_csv(rows)
    write_markdown(rows)
    write_tex(rows)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
