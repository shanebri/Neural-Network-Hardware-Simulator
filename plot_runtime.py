import argparse
from pathlib import Path

import torch

from analysis.visualization import (
    build_runtime_table,
    plot_runtime_by_precision,
    plot_runtime_vs_batch,
)
from models.mlp import SimpleMLP


# small cli for generating runtime plots
def _parse_int_list(raw: str) -> list[int]:
    # allow "1,8,32" style inputs
    return [int(v) for v in raw.split(",") if v.strip()]


def _parse_str_list(raw: str) -> list[str]:
    # allow "fp16,fp32" style inputs
    return [v.strip() for v in raw.split(",") if v.strip()]


def main():
    # set up cli arguments
    parser = argparse.ArgumentParser(
        description="Generate runtime estimate plots for the SimpleMLP."
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,8,32,128",
        help="Comma-separated batch sizes to evaluate (default: 1,8,32,128).",
    )
    parser.add_argument(
        "--precisions",
        default="fp16,fp32,fp64",
        help="Comma-separated GPU precisions to plot (default: fp16,fp32,fp64).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to store plots (default: outputs).",
    )
    parser.add_argument(
        "--bar-batch-size",
        type=int,
        default=None,
        help="Batch size to use for the bar plot (default: largest batch).",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write the estimates to CSV alongside the plots.",
    )
    args = parser.parse_args()

    # parse comma-separated args
    batch_sizes = _parse_int_list(args.batch_sizes)
    gpu_precisions = _parse_str_list(args.precisions)
    bar_batch_size = args.bar_batch_size or max(batch_sizes)

    output_dir = args.output_dir
    # ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # build the model and estimate table
    model = SimpleMLP()
    df = build_runtime_table(
        model=model,
        batch_sizes=batch_sizes,
        gpu_precisions=gpu_precisions,
    )

    # output plot filenames
    line_path = output_dir / "runtime_vs_batch.png"
    bar_path = output_dir / f"runtime_by_precision_b{bar_batch_size}.png"

    # generate plots
    plot_runtime_vs_batch(df, output_path=line_path)
    plot_runtime_by_precision(df, batch_size=bar_batch_size, output_path=bar_path)

    if args.write_csv:
        # optionally dump the table for spreadsheet use
        csv_path = output_dir / "runtime_estimates.csv"
        df.to_csv(csv_path, index=False)
        print(f"Wrote estimates: {csv_path}")

    # simple terminal feedback
    print(f"Saved line plot: {line_path}")
    print(f"Saved bar plot: {bar_path}")


if __name__ == "__main__":
    main()
