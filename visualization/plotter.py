"""
Plot generation utilities for experiment results.

Handles creation of accuracy degradation curves and other visualizations.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import logging
import os

import matplotlib

# Force the non-interactive Agg backend so this module is safe to import in
# child processes spawned by ProcessPoolExecutor on any OS.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (import after backend select)
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def _drop_at_75_score(method_df: pd.DataFrame, x_col: str, y_col: str) -> float:
    """Single ranking rule for legends/colors: higher DROP@75 first."""
    if method_df.empty:
        return float("-inf")

    base_level = method_df[x_col].min()
    base_acc = method_df.loc[method_df[x_col] == base_level, y_col].mean()

    if (method_df[x_col] == 75).any():
        acc_at_75 = method_df.loc[method_df[x_col] == 75, y_col].mean()
    else:
        nearest_idx = (method_df[x_col] - 75).abs().idxmin()
        acc_at_75 = float(method_df.loc[nearest_idx, y_col])

    return float(base_acc - acc_at_75)


def _render_curve_group(
    group_df: pd.DataFrame,
    output_path: Path,
    gen_model: str,
    judge_model: str,
    fill_strat: str,
    x_col: str,
    y_col: str,
    hue_col: str,
) -> str:
    """Render a single accuracy-degradation figure to ``output_path``.

    Top-level so it pickles cleanly for ProcessPoolExecutor on Windows
    (spawn-based child processes).
    """
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    rank_series = group_df.groupby(hue_col).apply(
        lambda d: _drop_at_75_score(d, x_col=x_col, y_col=y_col),
        include_groups=False
    )
    hue_order = rank_series.sort_values(ascending=False).index.tolist()
    palette = sns.color_palette("viridis", n_colors=len(hue_order))

    sns.lineplot(
        data=group_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        hue_order=hue_order,
        palette=palette,
        marker="o",
        linewidth=2.5,
    )

    if "mean_accuracy_top5" in group_df.columns:
        sns.lineplot(
            data=group_df,
            x=x_col,
            y="mean_accuracy_top5",
            hue=hue_col,
            hue_order=hue_order,
            palette=palette,
            marker="s",
            linewidth=2.0,
            linestyle="--",
            alpha=0.7,
            legend=False
        )

    x_min = group_df[x_col].min()
    x_max = group_df[x_col].max()
    x_range = x_max - x_min
    x_padding = max(2, x_range * 0.02)

    plt.title(
        f"Accuracy Degradation (Solid: Top-1, Dashed: Top-5)\nGenerator: {gen_model} | Judge: {judge_model} | Fill: {fill_strat}",
        fontsize=16,
    )
    plt.xlabel("Percentage of Pixels Removed (%)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.legend(title=hue_col.replace("_", " ").title())

    plt.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close()
    return str(output_path)


def _resolve_plot_workers() -> int:
    """Pick a sensible default for the plot ProcessPool worker count."""
    env = os.environ.get("PHASE4_PLOT_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 1
    return max(1, min(cpu - 1, 8))


def plot_accuracy_degradation_curves(
    results_df: pd.DataFrame,
    output_dir: Path,
    x_col: str = "occlusion_level",
    y_col: str = "mean_accuracy",
    hue_col: str = "attribution_method",
    max_workers: int | None = None,
):
    """Generate and save accuracy-degradation plots in parallel.

    One figure per ``(generating_model, judging_model, fill_strategy)`` group is
    rendered concurrently using :class:`concurrent.futures.ProcessPoolExecutor`
    with the matplotlib ``Agg`` backend (process-safe). Set
    ``PHASE4_PLOT_WORKERS`` in the environment to override the default worker
    count; pass ``max_workers=1`` to fall back to sequential rendering for
    debugging.
    """
    group_cols = ["generating_model", "judging_model", "fill_strategy"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect (group_df, output_path, *names) tuples first so we can either
    # parallelise or run inline depending on the requested worker count.
    jobs = []
    for name, group_df in results_df.groupby(group_cols):
        gen_model, judge_model, fill_strat = name
        filename = f"{gen_model}_{judge_model}_{fill_strat}.png"
        output_path = output_dir / filename
        jobs.append(
            (group_df.copy(), output_path, gen_model, judge_model, fill_strat)
        )

    if not jobs:
        return

    if max_workers is None:
        max_workers = _resolve_plot_workers()
    max_workers = max(1, min(max_workers, len(jobs)))

    if max_workers == 1:
        for group_df, output_path, gen_model, judge_model, fill_strat in jobs:
            _render_curve_group(
                group_df, output_path, gen_model, judge_model, fill_strat,
                x_col, y_col, hue_col,
            )
            logging.info(f"Saved plot: {output_path}")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _render_curve_group,
                group_df, output_path, gen_model, judge_model, fill_strat,
                x_col, y_col, hue_col,
            ): output_path
            for group_df, output_path, gen_model, judge_model, fill_strat in jobs
        }
        for fut in as_completed(futures):
            output_path = futures[fut]
            try:
                fut.result()
                logging.info(f"Saved plot: {output_path}")
            except Exception as e:
                logging.error(f"Plot rendering failed for {output_path}: {e}")


def plot_fill_strategy_comparison(
    results_df: pd.DataFrame,
    output_dir: Path,
    x_col: str = "occlusion_level",
    y_col: str = "mean_accuracy"
):
    """
    Generate comparison plot of fill strategies averaged across all models and methods.
    
    Shows the overall impact of different occlusion fill strategies (gray, blur, etc.)
    by averaging over all generating models, judging models, and attribution methods.
    
    Args:
        results_df: DataFrame containing aggregated results
        output_dir: Directory to save plot
        x_col: Column name for x-axis (occlusion level)
        y_col: Column name for y-axis (mean accuracy)
    """
    # Average across all models and methods, grouping only by strategy and level
    agg_dict = {y_col: 'mean'}
    if 'mean_accuracy_top5' in results_df.columns:
        agg_dict['mean_accuracy_top5'] = 'mean'
    
    strategy_df = results_df.groupby(['fill_strategy', x_col]).agg(agg_dict).reset_index()
    
    # Calculate dynamic x-axis range from ACTUAL data (before adding boundary points)
    x_min = strategy_df[x_col].min()
    x_max = strategy_df[x_col].max()
    x_range = x_max - x_min
    x_padding = max(2, x_range * 0.02)  # At least 2% padding, or 2 units
    
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    plot = sns.lineplot(
        data=strategy_df,
        x=x_col,
        y=y_col,
        hue='fill_strategy',
        marker='o',
        linewidth=3,
        markersize=8
    )
    
    if 'mean_accuracy_top5' in strategy_df.columns:
        sns.lineplot(
            data=strategy_df,
            x=x_col,
            y='mean_accuracy_top5',
            hue='fill_strategy',
            marker='s',
            linewidth=2,
            linestyle='--',
            markersize=6,
            alpha=0.7,
            legend=False
        )
    
    plt.title(
        "Fill Strategy Comparison (Solid: Top-1, Dashed: Top-5)\nAveraged Across All Models and Attribution Methods",
        fontsize=16,
        fontweight='bold'
    )
    plt.xlabel("Percentage of Pixels Removed (%)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.legend(title='Fill Strategy', fontsize=12, title_fontsize=13)
    plt.grid(True, alpha=0.3)
    
    filename = "fill_strategy_comparison.png"
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved fill strategy comparison plot: {output_path}")
