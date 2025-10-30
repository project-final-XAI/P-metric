"""
Plot generation utilities for experiment results.

Handles creation of accuracy degradation curves and other visualizations.
"""

from pathlib import Path
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_accuracy_degradation_curves(
        results_df: pd.DataFrame,
        output_dir: Path,
        x_col: str = "occlusion_level",
        y_col: str = "mean_accuracy",
        hue_col: str = "attribution_method"
):
    """
    Generate and save plots of accuracy degradation curves.
    
    Args:
        results_df: DataFrame containing aggregated results
        output_dir: Directory to save plots
        x_col: Column name for x-axis (occlusion level)
        y_col: Column name for y-axis (mean accuracy)
        hue_col: Column name to differentiate curves by color
    """
    # Create unique plot for each combination
    group_cols = ["generating_model", "judging_model", "fill_strategy"]

    for name, group_df in results_df.groupby(group_cols):
        gen_model, judge_model, fill_strat = name

        # Add boundary points (0%, 1) and (100%, 0) for each method
        extended_group_df = group_df.copy()
        # for hue_value in group_df[hue_col].unique():
        #     extended_group_df = pd.concat([
        #         extended_group_df,
        #         pd.DataFrame({
        #             x_col: [0, 100],
        #             y_col: [1, 0],
        #             hue_col: [hue_value, hue_value]
        #         })
        #     ])

        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid")

        plot = sns.lineplot(
            data=extended_group_df,
            x=x_col,
            y=y_col,
            hue=hue_col,
            marker='o',
            linewidth=2.5
        )

        plt.title(
            f"Accuracy Degradation\nGenerator: {gen_model} | Judge: {judge_model} | Fill: {fill_strat}",
            fontsize=16
        )
        plt.xlabel("Percentage of Pixels Removed (%)", fontsize=12)
        plt.ylabel("Top-1 Accuracy", fontsize=12)
        plt.ylim(-0.05, 1.05)
        plt.xlim(-2, 102)
        plt.legend(title=hue_col.replace('_', ' ').title())

        filename = f"{gen_model}_{judge_model}_{fill_strat}.png"
        output_path = output_dir / filename
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
        logging.info(f"Saved plot: {output_path}")


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
    strategy_df = results_df.groupby(['fill_strategy', x_col])[y_col].mean().reset_index()
    
    # Add boundary points (0%, 1) and (100%, 0) for each strategy
    extended_df = strategy_df.copy()
    for strategy in strategy_df['fill_strategy'].unique():
        extended_df = pd.concat([
            extended_df,
            pd.DataFrame({
                'fill_strategy': [strategy, strategy],
                x_col: [0, 100],
                y_col: [1, 0]
            })
        ])
    
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    plot = sns.lineplot(
        data=extended_df,
        x=x_col,
        y=y_col,
        hue='fill_strategy',
        marker='o',
        linewidth=3,
        markersize=8
    )
    
    plt.title(
        "Fill Strategy Comparison\nAveraged Across All Models and Attribution Methods",
        fontsize=16,
        fontweight='bold'
    )
    plt.xlabel("Percentage of Pixels Removed (%)", fontsize=14)
    plt.ylabel("Top-1 Accuracy", fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.xlim(-2, 102)
    plt.legend(title='Fill Strategy', fontsize=12, title_fontsize=13)
    plt.grid(True, alpha=0.3)
    
    filename = "fill_strategy_comparison.png"
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved fill strategy comparison plot: {output_path}")
