from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter


METHOD_COLORS = {
    "partial_unfreeze": "#1b9e77",
    "lora": "#377eb8",
    "qlora": "#ff7f00",
    "full_ft": "#d62728",
    "full_ft_optimized": "#7f0000",
}

METHOD_LABELS = {
    "partial_unfreeze": "Partial Unfreeze",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "full_ft": "FullFT",
    "full_ft_optimized": "FullFT (Opt.)",
}

METHOD_MARKERS = {
    "partial_unfreeze": "^",
    "lora": "o",
    "qlora": "D",
    "full_ft": "s",
    "full_ft_optimized": "X",
}

PREDICTOR_LABELS = {
    "audit_score": "UpgradeGuard",
    "random_text_activation_drift": "Random-text drift",
    "smoke_test_failure_rate": "Smoke-test failure",
    "weight_spectral_score": "Weight spectral",
    "parameter_distance_l2": "Parameter distance",
}

PREDICTOR_COLORS = {
    "audit_score": "#c0392b",
    "random_text_activation_drift": "#1b9e77",
    "smoke_test_failure_rate": "#377eb8",
    "weight_spectral_score": "#6a3d9a",
    "parameter_distance_l2": "#7f7f7f",
}

PANEL_COLORS = {
    ("Qwen/Qwen2.5-7B-Instruct", "translation"): "#4c78a8",
    ("Qwen/Qwen2.5-7B-Instruct", "summarization"): "#f58518",
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "summarization"): "#54a24b",
}

PANEL_LABELS = {
    ("Qwen/Qwen2.5-7B-Instruct", "translation"): "Qwen trans.",
    ("Qwen/Qwen2.5-7B-Instruct", "summarization"): "Qwen summ.",
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "summarization"): "Llama summ.",
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.frameon": False,
        "font.size": 10,
    }
)

PANEL_LABEL_OFFSETS = {
    "figure_02_qwen_translation_panel": {
        "partial_unfreeze": (8, 8),
        "lora": (8, -10),
        "qlora": (8, 8),
        "full_ft": (8, -8),
        "full_ft_optimized": (8, 8),
    },
    "figure_03_qwen_summarization_panel": {
        "partial_unfreeze": (8, 8),
        "lora": (8, -10),
        "qlora": (8, 8),
        "full_ft": (8, -8),
    },
    "figure_04_llama_transfer_panel": {
        "partial_unfreeze": (8, 8),
        "lora": (8, 8),
        "qlora": (8, -10),
        "full_ft": (8, 8),
    },
}

PANEL_AXIS_PADDING = {
    "figure_02_qwen_translation_panel": {"x_low": 0.08, "x_high": 0.22, "y_low": 0.02, "y_high": 0.12},
    "figure_03_qwen_summarization_panel": {"x_low": 0.08, "x_high": 0.28, "y_low": 0.03, "y_high": 0.12},
    "figure_04_llama_transfer_panel": {"x_low": 0.08, "x_high": 0.24, "y_low": 0.03, "y_high": 0.12},
}

METHOD_LEGEND_LAYOUTS = {
    "figure_02_qwen_translation_panel": {"loc": "upper left", "bbox_to_anchor": (0.01, 0.99), "ncol": 2},
    "figure_03_qwen_summarization_panel": {"loc": "upper right", "bbox_to_anchor": (0.98, 0.99), "ncol": 2},
    "figure_04_llama_transfer_panel": {"loc": "upper left", "bbox_to_anchor": (0.01, 0.99), "ncol": 2},
}

SCATTER_LABEL_OFFSETS = {
    "QT-PU": (8, 10),
    "QT-LoRA": (-18, 8),
    "QT-QLoRA": (8, 10),
    "QT-FullFT": (-40, 8),
    "QT-FTopt": (8, 8),
    "QS-PU": (-34, 10),
    "QS-LoRA": (8, 8),
    "QS-QLoRA": (8, 10),
    "QS-FullFT": (8, 8),
    "LS-PU": (8, 8),
    "LS-LoRA": (8, 8),
    "LS-QLoRA": (-44, -12),
    "LS-FullFT": (8, 8),
}


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def legend_handle_for_method(method: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker=METHOD_MARKERS.get(method, "o"),
        linestyle="None",
        markerfacecolor=METHOD_COLORS.get(method, "#333333"),
        markeredgecolor="black",
        markeredgewidth=0.9,
        markersize=9,
        label=METHOD_LABELS.get(method, method),
    )


def legend_handle_for_panel(model: str, task: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=PANEL_COLORS[(model, task)],
        markeredgecolor="black",
        markeredgewidth=0.9,
        markersize=9,
        label=PANEL_LABELS[(model, task)],
    )


def short_panel_label(model: str, task: str, update_label: str) -> str:
    if "Qwen" in model and task == "translation":
        prefix = "QT"
    elif "Qwen" in model and task == "summarization":
        prefix = "QS"
    elif "llama" in model.lower() and task == "summarization":
        prefix = "LS"
    else:
        prefix = "RUN"
    return f"{prefix}-{METHOD_LABELS.get(update_label, update_label)}"


def short_scatter_label(model: str, task: str, update_label: str) -> str:
    if "Qwen" in model and task == "translation":
        prefix = "QT"
    elif "Qwen" in model and task == "summarization":
        prefix = "QS"
    elif "llama" in model.lower() and task == "summarization":
        prefix = "LS"
    else:
        prefix = "RUN"

    suffix = {
        "partial_unfreeze": "PU",
        "lora": "L",
        "qlora": "QL",
        "full_ft": "FT",
        "full_ft_optimized": "FTopt",
    }.get(update_label, update_label)
    return f"{prefix}-{suffix}"


def load_summary(refresh_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(refresh_dir / "summary_table_enriched.csv")
    return df[df["external_composite_safety_regression"].notna()].copy()


def make_pipeline_diagram(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")

    boxes = [
        (0.02, 0.25, 0.16, 0.5, "Aligned Base\nModel"),
        (0.23, 0.25, 0.18, 0.5, "Benign Update\nMethod Choice"),
        (0.47, 0.25, 0.20, 0.5, "UpgradeGuard\nFixed-Budget Audit"),
        (0.73, 0.25, 0.12, 0.5, "Risk\nSignal"),
        (0.89, 0.25, 0.09, 0.5, "Ship /\nEscalate"),
    ]

    fills = ["#e8f1ff", "#eef7ee", "#fff5e6", "#fdecea", "#f4f0ff"]
    for (x, y, w, h, text), fill in zip(boxes, fills):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#333333",
            facecolor=fill,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            transform=ax.transAxes,
        )

    arrow_pairs = [
        (0.18, 0.49, 0.23, 0.49),
        (0.41, 0.49, 0.47, 0.49),
        (0.67, 0.49, 0.73, 0.49),
        (0.85, 0.49, 0.89, 0.49),
    ]
    for x1, y1, x2, y2 in arrow_pairs:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.8,
                color="#333333",
                transform=ax.transAxes,
            )
        )

    ax.text(
        0.32,
        0.15,
        "FullFT / LoRA / QLoRA / Partial Unfreeze",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.text(
        0.57,
        0.15,
        "Behavioral canaries + safety drift + specificity controls",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.text(
        0.935,
        0.15,
        "full external evaluation",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    fig.suptitle("UpgradeGuard Post-Update Assurance Workflow", y=0.98, fontsize=15, weight="bold")
    save_figure(fig, out_dir, "figure_01_pipeline")


def panel_subset(df: pd.DataFrame, model: str, task: str) -> pd.DataFrame:
    subset = df[(df["model"] == model) & (df["task"] == task)].copy()
    subset["sort_key"] = subset["update_label"].map(
        {
            "partial_unfreeze": 0,
            "lora": 1,
            "qlora": 2,
            "full_ft": 3,
            "full_ft_optimized": 4,
        }
    )
    return subset.sort_values(["sort_key", "train_learning_rate"], kind="stable")


def annotate_callout(
    ax: plt.Axes,
    label: str,
    x: float,
    y: float,
    dx: int,
    dy: int,
    color: str,
    fontsize: float = 9.0,
) -> None:
    ha = "left" if dx >= 0 else "right"
    va = "bottom" if dy >= 0 else "top"
    ax.annotate(
        label,
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=fontsize,
        color=color,
        weight="bold",
        ha=ha,
        va=va,
        annotation_clip=False,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
            "boxstyle": "round,pad=0.18",
        },
    )


def set_padded_limits(ax: plt.Axes, x_vals: pd.Series, y_vals: pd.Series, stem: str) -> None:
    padding = PANEL_AXIS_PADDING[stem]
    x_span = max(float(x_vals.max() - x_vals.min()), 1e-6)
    y_span = max(float(y_vals.max() - y_vals.min()), 1e-6)
    ax.set_xlim(
        float(x_vals.min()) - padding["x_low"] * x_span,
        float(x_vals.max()) + padding["x_high"] * x_span,
    )
    ax.set_ylim(
        max(0.0, float(y_vals.min()) - padding["y_low"] * y_span),
        float(y_vals.max()) + padding["y_high"] * y_span + 0.002,
    )


def annotate_points(ax: plt.Axes, data: pd.DataFrame, x_col: str, y_col: str, stem: str) -> None:
    offsets = PANEL_LABEL_OFFSETS[stem]
    for _, row in data.iterrows():
        label = METHOD_LABELS.get(row["update_label"], row["update_label"])
        dx, dy = offsets.get(row["update_label"], (6, 6))
        annotate_callout(
            ax,
            label,
            row[x_col],
            row[y_col],
            dx,
            dy,
            METHOD_COLORS.get(row["update_label"], "#333333"),
            fontsize=8.8,
        )


def make_method_panel(
    df: pd.DataFrame,
    out_dir: Path,
    model: str,
    task: str,
    utility_col: str,
    utility_label: str,
    stem: str,
    title: str,
) -> None:
    data = panel_subset(df, model, task)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for _, row in data.iterrows():
        method = row["update_label"]
        ax.scatter(
            row[utility_col],
            row["external_composite_safety_regression"],
            s=185,
            color=METHOD_COLORS.get(method, "#333333"),
            marker=METHOD_MARKERS.get(method, "o"),
            edgecolor="white",
            linewidth=1.7,
            zorder=3,
            alpha=0.96,
        )

    method_order = [m for m in ["partial_unfreeze", "lora", "qlora", "full_ft", "full_ft_optimized"] if m in set(data["update_label"])]
    legend_layout = METHOD_LEGEND_LAYOUTS.get(stem, {"loc": "upper left", "bbox_to_anchor": (0.01, 0.99), "ncol": 2})
    legend = ax.legend(
        handles=[legend_handle_for_method(method) for method in method_order],
        loc=legend_layout["loc"],
        bbox_to_anchor=legend_layout["bbox_to_anchor"],
        ncol=legend_layout["ncol"],
        title="Update method",
        fontsize=9,
        title_fontsize=9.5,
        frameon=True,
        fancybox=False,
        borderpad=0.5,
        handletextpad=0.55,
        columnspacing=1.2,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d6d6d6")
    legend.get_frame().set_alpha(0.95)

    ax.text(
        0.985,
        0.04,
        "Preferred region",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4a4a4a",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "boxstyle": "round,pad=0.18"},
    )
    ax.annotate(
        "",
        xy=(0.96, 0.06),
        xytext=(0.88, 0.13),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "color": "#4a4a4a", "lw": 1.2},
    )

    ax.set_title(title, pad=10)
    ax.set_xlabel(utility_label)
    ax.set_ylabel("External safety regression (lower is safer)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.18)

    x_vals = data[utility_col].dropna()
    y_vals = data["external_composite_safety_regression"]
    set_padded_limits(ax, x_vals, y_vals, stem)

    save_figure(fig, out_dir, stem)


def make_audit_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.7))
    fig.subplots_adjust(right=0.62)
    for _, row in df.iterrows():
        panel_key = (row["model"], row["task"])
        ax.scatter(
            row["audit_score"],
            row["external_composite_safety_regression"],
            s=140,
            color=PANEL_COLORS.get(panel_key, "#333333"),
            marker=METHOD_MARKERS.get(row["update_label"], "o"),
            edgecolor="black",
            linewidth=0.9,
            alpha=0.95,
            zorder=3,
        )

    highlighted = {
        ("Qwen/Qwen2.5-7B-Instruct", "translation", "full_ft_optimized"): ("QT FullFT (Opt.)", (10, 10)),
        ("Qwen/Qwen2.5-7B-Instruct", "translation", "full_ft"): ("QT FullFT", (8, 10)),
        ("Qwen/Qwen2.5-7B-Instruct", "summarization", "qlora"): ("QS QLoRA", (-10, 12)),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "summarization", "full_ft"): ("LS FullFT", (10, 8)),
    }
    for _, row in df.iterrows():
        key = (row["model"], row["task"], row["update_label"])
        if key not in highlighted:
            continue
        label, (dx, dy) = highlighted[key]
        annotate_callout(
            ax,
            label,
            row["audit_score"],
            row["external_composite_safety_regression"],
            dx,
            dy,
            "#222222",
            fontsize=8.4,
        )

    x = df["audit_score"].to_numpy()
    y = df["external_composite_safety_regression"].to_numpy()
    coeffs = np.polyfit(x, y, 1)
    xp = np.linspace(x.min(), x.max(), 200)
    yp = coeffs[0] * xp + coeffs[1]
    ax.plot(xp, yp, color="#444444", linestyle="--", linewidth=1.6, zorder=2)

    ax.set_ylabel("External safety regression")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.20)
    x_margin = max(float(x.max() - x.min()), 1e-6)
    y_margin = max(float(y.max() - y.min()), 1e-6)
    ax.set_xlim(float(x.min()) - 0.08 * x_margin, float(x.max()) + 0.10 * x_margin)
    ax.set_ylim(max(0.0, float(y.min()) - 0.04 * y_margin), float(y.max()) + 0.06 * y_margin)
    ax.text(
        0.02,
        0.97,
        r"Global Spearman $\rho = 0.619$",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
    )

    panel_handles = [
        legend_handle_for_panel("Qwen/Qwen2.5-7B-Instruct", "translation"),
        legend_handle_for_panel("Qwen/Qwen2.5-7B-Instruct", "summarization"),
        legend_handle_for_panel("meta-llama/Meta-Llama-3.1-8B-Instruct", "summarization"),
    ]
    method_handles = [
        legend_handle_for_method(method)
        for method in ["partial_unfreeze", "lora", "qlora", "full_ft", "full_ft_optimized"]
        if method in set(df["update_label"])
    ]

    legend_panels = ax.legend(
        handles=panel_handles,
        title="Panel",
        loc="upper left",
        bbox_to_anchor=(1.00, 0.54),
        ncol=1,
        fontsize=8.8,
        title_fontsize=9.1,
        frameon=True,
        fancybox=False,
    )
    legend_panels.get_frame().set_facecolor("white")
    legend_panels.get_frame().set_edgecolor("#d6d6d6")
    legend_panels.get_frame().set_alpha(0.95)
    ax.add_artist(legend_panels)

    legend_methods = ax.legend(
        handles=method_handles,
        title="Method",
        loc="upper left",
        bbox_to_anchor=(1.00, 0.98),
        ncol=1,
        fontsize=8.8,
        title_fontsize=9.1,
        frameon=True,
        fancybox=False,
    )
    legend_methods.get_frame().set_facecolor("white")
    legend_methods.get_frame().set_edgecolor("#d6d6d6")
    legend_methods.get_frame().set_alpha(0.95)
    ax.set_xlabel("UpgradeGuard audit score")

    save_figure(fig, out_dir, "figure_05_audit_vs_external_scatter")


def make_global_vs_conditioned(refresh_dir: Path, out_dir: Path) -> None:
    global_df = pd.read_csv(refresh_dir / "predictor_comparison.csv")
    global_df = global_df[global_df["target"] == "external_composite_safety_regression"].copy()
    cond_df = pd.read_csv(refresh_dir / "conditioned_predictor_summary.csv")
    cond_df = cond_df[cond_df["conditioning"] == "within_model_task"].copy()

    predictors = [
        "audit_score",
        "random_text_activation_drift",
        "smoke_test_failure_rate",
        "weight_spectral_score",
        "parameter_distance_l2",
    ]

    merged = global_df[global_df["predictor"].isin(predictors)][
        ["predictor", "spearman_rho", "pairwise_ordering_accuracy"]
    ].merge(
        cond_df[cond_df["predictor"].isin(predictors)][
            ["predictor", "mean_spearman_rho", "weighted_pairwise_ordering_accuracy"]
        ],
        on="predictor",
        how="inner",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=False, sharey=False)
    panels = [
        ("spearman_rho", "mean_spearman_rho", "Global Spearman", "Within-panel mean Spearman"),
        (
            "pairwise_ordering_accuracy",
            "weighted_pairwise_ordering_accuracy",
            "Global pairwise ordering",
            "Within-panel weighted pairwise ordering",
        ),
    ]

    # Per-predictor label offsets to prevent overlap in both panels
    label_offsets_left = {
        "audit_score":                  (6,   5),
        "random_text_activation_drift": (6,   5),
        "smoke_test_failure_rate":       (6,  -12),
        "weight_spectral_score":         (6,   5),
        "parameter_distance_l2":         (6,  -12),
    }
    label_offsets_right = {
        "audit_score":                  (-72,  5),
        "random_text_activation_drift": (6,    5),
        "smoke_test_failure_rate":       (6,   -12),
        "weight_spectral_score":         (6,    5),
        "parameter_distance_l2":         (6,   -12),
    }
    offset_maps = [label_offsets_left, label_offsets_right]

    for ax, (xcol, ycol, xlabel, ylabel), offsets in zip(axes, panels, offset_maps):
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.0)
        for _, row in merged.iterrows():
            predictor = row["predictor"]
            ax.scatter(
                row[xcol],
                row[ycol],
                s=140,
                color=PREDICTOR_COLORS[predictor],
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
            dx, dy = offsets.get(predictor, (6, 5))
            ax.annotate(
                PREDICTOR_LABELS[predictor],
                (row[xcol], row[ycol]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8.5,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Global Mixed-Set Ranking vs. Within-Panel Update Ranking", y=1.02, fontsize=15, weight="bold")
    save_figure(fig, out_dir, "figure_06_global_vs_conditioned")


def make_budget_ablation(refresh_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(refresh_dir / "budget_ablation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    axis_order = ["canaries", "layers", "probes"]
    axis_titles = {
        "canaries": "Canary budget",
        "layers": "Observed layers",
        "probes": "Safety probe budget",
    }
    default_points = {"canaries": "10", "layers": "last2", "probes": "25"}

    for ax, axis_name in zip(axes, axis_order):
        g = df[df["axis"] == axis_name].copy()
        if axis_name == "layers":
            order = ["last1", "last2", "last4", "all"]
            g["plot_x"] = g["budget"].map({k: i for i, k in enumerate(order)})
            x = g.sort_values("plot_x")["plot_x"]
            labels = g.sort_values("plot_x")["budget"].tolist()
        else:
            g["plot_x"] = g["budget"].astype(int)
            x = g.sort_values("plot_x")["plot_x"]
            labels = [str(v) for v in g.sort_values("plot_x")["budget"].tolist()]

        g = g.sort_values("plot_x")
        ax.plot(x, g["spearman_rho"], marker="o", color="#c0392b", linewidth=2.2, label="Spearman")
        ax.plot(x, g["pearson_r"], marker="s", color="#377eb8", linewidth=1.8, alpha=0.85, label="Pearson")

        default_budget = default_points[axis_name]
        default_row = g[g["budget"].astype(str) == default_budget]
        if not default_row.empty:
            ax.scatter(
                default_row["plot_x"],
                default_row["spearman_rho"],
                s=220,
                color="#f1c40f",
                marker="*",
                edgecolor="black",
                linewidth=0.8,
                zorder=4,
            )

        ax.set_title(axis_titles[axis_name])
        ax.set_ylim(0, 0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Correlation with external regression")
    axes[1].legend(loc="lower right")
    fig.suptitle("Budget Scaling Shows Early Saturation", y=1.03, fontsize=15, weight="bold")
    save_figure(fig, out_dir, "figure_07_budget_scaling")


def make_escalation_curve(refresh_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(refresh_dir / "escalation_curve.csv")
    predictors = [
        "audit_score",
        "random_text_activation_drift",
        "smoke_test_failure_rate",
        "weight_spectral_score",
        "parameter_distance_l2",
    ]
    df = df[df["predictor"].isin(predictors) & (df["target"] == "external_composite_safety_regression")].copy()

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for predictor in predictors:
        g = df[df["predictor"] == predictor].sort_values("escalation_budget")
        if g.empty:
            continue
        ax.plot(
            g["escalation_budget"],
            g["captured_risk_mass"],
            marker="o",
            linewidth=2.3 if predictor == "audit_score" else 1.8,
            color=PREDICTOR_COLORS[predictor],
            label=PREDICTOR_LABELS[predictor],
        )

    ax.set_title("Escalation Budget vs. Captured External Risk")
    ax.set_xlabel("Fraction of candidate updates escalated")
    ax.set_ylabel("Captured risk mass")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    save_figure(fig, out_dir, "figure_08_escalation_curve")


def make_audit_ablation(refresh_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(refresh_dir / "conditioned_predictor_summary.csv")
    df = df[
        (df["conditioning"] == "within_model_task")
        & (
            df["predictor"].isin(
                [
                    "audit_behavioral_component",
                    "audit_representation_component",
                    "audit_score",
                    "canary_failure_rate",
                    "late_layer_safety_drift",
                ]
            )
        )
    ].copy()

    label_map = {
        "audit_score": "Full audit",
        "audit_behavioral_component": "Behavioral bundle",
        "audit_representation_component": "Representation bundle",
        "canary_failure_rate": "Canary only",
        "late_layer_safety_drift": "Drift only",
    }
    df["label"] = df["predictor"].map(label_map)
    order = ["Full audit", "Representation bundle", "Drift only", "Behavioral bundle", "Canary only"]
    df["sort_key"] = df["label"].map({k: i for i, k in enumerate(order)})
    df = df.sort_values("sort_key")

    colors = {
        "Full audit": "#c0392b",
        "Representation bundle": "#6a3d9a",
        "Drift only": "#9467bd",
        "Behavioral bundle": "#377eb8",
        "Canary only": "#74add1",
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(
        df["label"],
        df["weighted_pairwise_ordering_accuracy"],
        color=[colors[v] for v in df["label"]],
        edgecolor="black",
        linewidth=0.8,
    )
    for i, value in enumerate(df["weighted_pairwise_ordering_accuracy"]):
        ax.text(i, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9, weight="bold")

    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Within panel weighted ordering accuracy")
    ax.set_title("Representation Signal in Within Panel Ordering")
    ax.grid(True, axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save_figure(fig, out_dir, "figure_09_audit_ablation")


def make_gating_frontier(refresh_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(refresh_dir / "gating_simulation.csv")
    df = df[df["target"] == "external_composite_safety_regression"].copy()
    predictors = [
        "audit_score",
        "random_text_activation_drift",
        "smoke_test_failure_rate",
        "weight_spectral_score",
        "parameter_distance_l2",
    ]
    df = df[df["predictor"].isin(predictors)]

    # Per-predictor label offsets to avoid overlap
    gating_label_offsets = {
        "audit_score":                  (7,   -14),
        "random_text_activation_drift": (7,     8),
        "smoke_test_failure_rate":       (7,     5),
        "weight_spectral_score":         (-82,   5),
        "parameter_distance_l2":         (7,    -14),
    }

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for _, row in df.iterrows():
        predictor = row["predictor"]
        ax.scatter(
            row["full_eval_cost_saved_mean"],
            row["risky_updates_caught_rate_mean"],
            s=170,
            color=PREDICTOR_COLORS[predictor],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        dx, dy = gating_label_offsets.get(predictor, (7, 5))
        ax.annotate(
            PREDICTOR_LABELS[predictor],
            (row["full_eval_cost_saved_mean"], row["risky_updates_caught_rate_mean"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
        )

    ax.set_title("Gating Frontier: Risk Caught vs. Full-Eval Cost Saved")
    ax.set_xlabel("Full external evaluation cost saved")
    ax.set_ylabel("Risky updates caught rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0.18, 0.90)
    ax.set_ylim(0.28, 0.92)
    ax.grid(True, alpha=0.25)
    save_figure(fig, out_dir, "figure_10_gating_frontier")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from the final merged evidence package.")
    parser.add_argument(
        "--refresh-dir",
        type=Path,
        default=Path(r"C:\Users\Ashish\Safety_paper_1\results\paper_ready_final_refresh_20260407"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    refresh_dir = args.refresh_dir
    out_dir = args.out_dir or (refresh_dir / "figures")

    df = load_summary(refresh_dir)

    make_pipeline_diagram(out_dir)
    make_method_panel(
        df,
        out_dir,
        model="Qwen/Qwen2.5-7B-Instruct",
        task="translation",
        utility_col="bleu",
        utility_label="Utility (BLEU)",
        stem="figure_02_qwen_translation_panel",
        title="Qwen Translation: Reference Panel",
    )
    make_method_panel(
        df,
        out_dir,
        model="Qwen/Qwen2.5-7B-Instruct",
        task="summarization",
        utility_col="rougeL",
        utility_label="Utility (ROUGE-L)",
        stem="figure_03_qwen_summarization_panel",
        title="Qwen Summarization: Boundary Case Panel",
    )
    make_method_panel(
        df,
        out_dir,
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        task="summarization",
        utility_col="rougeL",
        utility_label="Utility (ROUGE-L)",
        stem="figure_04_llama_transfer_panel",
        title="Llama Summarization: Transfer Panel",
    )
    make_audit_scatter(df, out_dir)
    make_global_vs_conditioned(refresh_dir, out_dir)
    make_budget_ablation(refresh_dir, out_dir)
    make_escalation_curve(refresh_dir, out_dir)
    make_audit_ablation(refresh_dir, out_dir)
    make_gating_frontier(refresh_dir, out_dir)


if __name__ == "__main__":
    main()
