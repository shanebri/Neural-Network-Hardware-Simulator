from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.flopCounter import count_mlp_macs, macs_to_flops
from analysis.hardware_estimates import (
    compute_flop_rate_cpu,
    compute_flop_rate_gpu,
    estimate_runtime,
)
from models.hardware_specs import cpu_specs, gpu_specs, m4_pro_specs


def build_runtime_table(
    model,
    batch_sizes: Sequence[int],
    gpu_precisions: Iterable[str] = ("fp16", "fp32", "fp64"),
    include_gpu: bool = True,
    include_cpu: bool = True,
    include_m4: bool = True,
) -> pd.DataFrame:
    """
    Create a tidy table of runtime estimates across batch sizes and hardware/precisions.
    """
    rows: List[dict] = []
    for batch in batch_sizes:
        macs = count_mlp_macs(model, batch_size=batch)
        flops = macs_to_flops(macs)

        if include_cpu:
            flop_rate = compute_flop_rate_cpu(cpu_specs)
            rows.append(
                {
                    "hardware": cpu_specs["name"],
                    "precision": "fp32",
                    "batch_size": batch,
                    "flops": flops,
                    "flop_rate": flop_rate,
                    "est_runtime": estimate_runtime(flops, flop_rate),
                }
            )

        if include_m4:
            flop_rate = compute_flop_rate_cpu(m4_pro_specs)
            rows.append(
                {
                    "hardware": m4_pro_specs["name"],
                    "precision": "fp32",
                    "batch_size": batch,
                    "flops": flops,
                    "flop_rate": flop_rate,
                    "est_runtime": estimate_runtime(flops, flop_rate),
                }
            )

        if include_gpu:
            for precision in gpu_precisions:
                flop_rate = compute_flop_rate_gpu(gpu_specs, precision=precision)
                rows.append(
                    {
                        "hardware": gpu_specs["name"],
                        "precision": precision,
                        "batch_size": batch,
                        "flops": flops,
                        "flop_rate": flop_rate,
                        "est_runtime": estimate_runtime(flops, flop_rate),
                    }
                )

    df = pd.DataFrame(rows)
    df["config"] = df["hardware"] + " (" + df["precision"] + ")"
    return df


def plot_runtime_vs_batch(
    df: pd.DataFrame,
    output_path: Path,
    yscale: str = "log",
    title: str = "Estimated Runtime vs Batch Size",
) -> Path:
    """
    Plot runtime vs. batch size for each hardware/precision config.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="batch_size",
        y="est_runtime",
        hue="config",
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Estimated runtime (s)")
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    ax.legend(title="Hardware (precision)")
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_runtime_by_precision(
    df: pd.DataFrame,
    batch_size: int,
    output_path: Path,
    title: str | None = None,
) -> Path:
    """
    Plot runtime bars across precisions for a fixed batch size.
    """
    sns.set_theme(style="whitegrid")
    subset = df[df["batch_size"] == batch_size]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=subset,
        x="config",
        y="est_runtime",
        ax=ax,
    )
    ax.set_ylabel("Estimated runtime (s)")
    ax.set_xlabel("Hardware (precision)")
    ax.set_title(title or f"Estimated Runtime at Batch {batch_size}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
