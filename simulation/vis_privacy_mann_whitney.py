import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = BASE_DIR / "experiment_results"
DEFAULT_OUTPUT = (
    DEFAULT_RESULTS_ROOT / "detailed_metric_plots" / "privacy_attacks_mann_whitney_synth_vs_anon_synth.png"
)

VARIANTS = ("synth", "anon_synth")
LABELS = {
    "synth": "Synthetic",
    "anon_synth": "Anonymized + Synthetic",
}
METRICS = {
    "inference_average_risk": ("Attribute-inference Risk", "priv_risk"),
    "linkage_health_risk": ("Linkage Risk", "risk_value"),
    "singling_out_risk": ("Singling-out Risk", None),
}


def finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def pretty(name: str) -> str:
    return re.sub(r"[_-]+", " ", name).strip().title()


def load_jsons(directory: Path) -> List[Dict[str, Any]]:
    documents = []
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not read {path}: {error}") from error
        if isinstance(payload, dict):
            documents.append(payload)
    return documents


def collect_risk_documents(results_root: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    cinnamon_root = results_root / "cinnamon_anonym_anonsynth"
    synth_root = results_root / "synthesis_only"
    datasets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for dataset_dir in sorted(path for path in cinnamon_root.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        variants = {
            "anon_synth": load_jsons(dataset_dir / "anon_synth" / "risk_eval"),
            "synth": load_jsons(synth_root / dataset / "synth" / "risk_eval"),
        }
        if all(variants.values()):
            datasets[dataset] = variants
    if not datasets:
        raise RuntimeError(f"No complete risk results found under {results_root}")
    return datasets


def risk_value(document: Mapping[str, Any], metric: str, value_key: Optional[str]) -> Optional[float]:
    if metric == "singling_out_risk":
        univariate = risk_value(document, "univariate_singling_out_risk", "risk_value")
        multivariate = risk_value(document, "multivariate_singling_out_risk", "risk_value")
        return max(univariate, multivariate) if None not in (univariate, multivariate) else None
    value = document.get(metric)
    return finite_number(value.get(value_key)) if isinstance(value, dict) and value_key else None


def calculate_results(
    datasets: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for dataset in sorted(datasets):
        for metric, (_, value_key) in METRICS.items():
            values = {
                variant: np.asarray(
                    [
                        value
                        for document in datasets[dataset][variant]
                        if (value := risk_value(document, metric, value_key)) is not None
                    ],
                    dtype=float,
                )
                for variant in VARIANTS
            }
            synth = values["synth"]
            anon_synth = values["anon_synth"]
            test = mannwhitneyu(anon_synth, synth, alternative="two-sided", method="auto")
            probability_lower = 1.0 - float(test.statistic) / (len(anon_synth) * len(synth))
            row: Dict[str, Any] = {
                "dataset": dataset,
                "metric": metric,
                "u_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "probability_anon_synth_lower": probability_lower,
            }
            for variant in VARIANTS:
                row[f"n_{variant}"] = len(values[variant])
                row[f"mean_{variant}"] = float(values[variant].mean())
                row[f"std_{variant}"] = float(values[variant].std(ddof=1)) if len(values[variant]) > 1 else 0.0
            results.append(row)

    for result in results:
        probability_lower = result["probability_anon_synth_lower"]
        result["direction"] = (
            "anon_synth_lower" if probability_lower > 0.5 else "anon_synth_higher" if probability_lower < 0.5 else "equal"
        )
    return results


def write_csv(results: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def plot_heatmap(results: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    datasets = sorted({str(result["dataset"]) for result in results})
    result_by_key = {(result["dataset"], result["metric"]): result for result in results}
    gap_width = 0.5
    column_specs: List[Dict[str, Any]] = []
    for metric_index, (metric, (metric_title, _)) in enumerate(METRICS.items()):
        for variant in VARIANTS:
            column_specs.append({"variant": variant, "metric": metric, "title": metric_title, "is_gap": False})
        if metric_index < len(METRICS) - 1:
            column_specs.append({"is_gap": True})

    data: List[List[float]] = []
    for dataset in datasets:
        row = []
        for spec in column_specs:
            if spec["is_gap"]:
                row.append(float("nan"))
            else:
                result = result_by_key[(dataset, spec["metric"])]
                row.append(result["probability_anon_synth_lower"])
        data.append(row)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.facecolor": "#fbfbfb",
            "figure.facecolor": "white",
            "grid.color": "#c9c9c9",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.6,
        }
    )
    base = plt.get_cmap("YlGnBu")
    cmap = ListedColormap(base(np.linspace(0.82, 0, 256)))
    cmap.set_bad("#ffffff")
    norm = Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(12.0, max(2.5, 0.7 * len(datasets) + 1.5)))
    column_widths = [gap_width if spec["is_gap"] else 1.0 for spec in column_specs]
    x_edges = [0.0]
    for width in column_widths:
        x_edges.append(x_edges[-1] + width)
    y_edges = list(range(len(datasets) + 1))
    image = ax.pcolormesh(x_edges, y_edges, data, cmap=cmap, norm=norm, shading="flat")
    ax.invert_yaxis()

    centers = [start + width / 2 for start, width in zip(x_edges[:-1], column_widths)]
    tick_indices = [index for index, spec in enumerate(column_specs) if not spec["is_gap"]]
    ax.set_xticks([centers[index] for index in tick_indices])
    ax.set_xticklabels([LABELS[column_specs[index]["variant"]] for index in tick_indices])
    ax.set_yticks([row + 0.5 for row in range(len(datasets))])
    ax.set_yticklabels([pretty(dataset) for dataset in datasets])
    ax.tick_params(axis="x", labelsize=9, pad=6)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xticks(x_edges, minor=True)
    ax.set_yticks(y_edges, minor=True)
    ax.grid(which="minor", color="#e1e1e1", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for metric, (metric_title, _) in METRICS.items():
        indices = [
            index
            for index, spec in enumerate(column_specs)
            if not spec["is_gap"] and spec["metric"] == metric
        ]
        center = (x_edges[indices[0]] + x_edges[indices[-1] + 1]) / 2
        ax.text(
            center,
            1.03,
            metric_title,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="normal",
            color="#222222",
        )

    for row_index, dataset in enumerate(datasets):
        for column_index, spec in enumerate(column_specs):
            if spec["is_gap"]:
                continue
            result = result_by_key[(dataset, spec["metric"])]
            variant = spec["variant"]
            ax.text(
                centers[column_index],
                row_index + 0.32,
                f"{result[f'mean_{variant}']:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="normal",
                color="#111111",
            )
            ax.text(
                centers[column_index],
                row_index + 0.57,
                f"±{result[f'std_{variant}']:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="normal",
                color="#333333",
            )
            if variant == "anon_synth":
                p_value = result["p_value"]
                p_label = "p<.001" if p_value < 0.001 else f"p={p_value:.3f}"
                ax.text(
                    centers[column_index],
                    row_index + 0.79,
                    p_label,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold" if p_value < 0.05 else "normal",
                    color="#333333",
                )

    for row_index, dataset in enumerate(datasets):
        for metric in METRICS:
            result = result_by_key[(dataset, metric)]
            indices = [
                index
                for index, spec in enumerate(column_specs)
                if not spec["is_gap"] and spec["metric"] == metric
            ]
            if result["p_value"] >= 0.05:
                for index in indices:
                    ax.add_patch(
                        Rectangle(
                            (x_edges[index], row_index),
                            x_edges[index + 1] - x_edges[index],
                            1,
                            facecolor="#eeeeee",
                            edgecolor="none",
                            alpha=0.62,
                            zorder=2,
                        )
                    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, ticks=(0.0, 0.5, 1.0))
    colorbar.ax.set_yticklabels(("Worse", "Similar", "Better"))
    colorbar.ax.tick_params(labelsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def self_check() -> None:
    assert risk_value(
        {
            "univariate_singling_out_risk": {"risk_value": 0.2},
            "multivariate_singling_out_risk": {"risk_value": 0.4},
        },
        "singling_out_risk",
        None,
    ) == 0.4


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot privacy attacks for all variants with Mann–Whitney U tests.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    self_check()
    results = calculate_results(collect_risk_documents(args.results_root))
    write_csv(results, args.output.with_suffix(".csv"))
    plot_heatmap(results, args.output)
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
