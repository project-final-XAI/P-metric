# run_analysis.py
"""
Final script for Phase 3: Analysis and Visualization.

This script reads the raw evaluation data, calculates summary statistics and
faithfulness metrics (AUC, DROP), saves the aggregated results, and generates
plots of the accuracy degradation curves.

Usage:
    python run_analysis.py
"""
import pandas as pd

try:
    import config
    from modules.metrics_calculator import calculate_auc, calculate_drop
    from plotting import plot_accuracy_degradation_curves
except ImportError as e:
    print(f"Error importing modules: {e}")
    exit()


def analyze_results():
    """Main function to run the analysis phase."""
    print("--- Starting Phase 3: Analysis & Visualization ---")

    # 1. Load raw results
    results_csv_path = config.RESULTS_DIR / "evaluation_results.csv"
    if not results_csv_path.exists():
        print(f"Error: Results file not found at '{results_csv_path}'. Please run Phase 2 first.")
        return

    print(f"Loading raw data from {results_csv_path}...")
    df = pd.read_csv(results_csv_path)

    # 2. Calculate mean accuracy for each curve
    group_cols = [
        "generating_model", "attribution_method", "judging_model",
        "fill_strategy", "occlusion_level"
    ]

    # Calculate the mean accuracy across all images for each point on the curve
    agg_df = df.groupby(group_cols)['is_correct'].mean().reset_index()
    agg_df.rename(columns={'is_correct': 'mean_accuracy'}, inplace=True)

    # 3. Calculate P-Metrics (AUC, DROP) for each complete curve
    metrics_list = []
    curve_group_cols = ["generating_model", "attribution_method", "judging_model", "fill_strategy"]

    for name, curve_df in agg_df.groupby(curve_group_cols):
        gen_model, method, judge_model, fill_strat = name

        # Get accuracy at 0% occlusion (baseline)
        # We assume 1.0 if not present, but a full run should include P=0
        baseline_acc = curve_df[curve_df['occlusion_level'] == 0]['mean_accuracy'].iloc[0] if 0 in curve_df[
            'occlusion_level'].values else 1.0

        accuracies = curve_df['mean_accuracy'].tolist()
        levels = curve_df['occlusion_level'].tolist()

        # Calculate metrics
        auc = calculate_auc(accuracies, levels)
        drop75 = calculate_drop(accuracies, levels, initial_accuracy=baseline_acc, drop_level=75)

        metrics_list.append({
            "generating_model": gen_model,
            "attribution_method": method,
            "judging_model": judge_model,
            "fill_strategy": fill_strat,
            "auc": auc,
            "drop_at_75": drop75
        })

    config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_list)

    # 4. Save aggregated data and metrics
    agg_output_path = config.ANALYSIS_DIR / "aggregated_accuracy_curves.csv"
    metrics_output_path = config.ANALYSIS_DIR / "faithfulness_metrics.csv"

    agg_df.to_csv(agg_output_path, index=False)
    metrics_df.to_csv(metrics_output_path, index=False)

    print(f"Aggregated curve data saved to {agg_output_path}")
    print(f"Faithfulness metrics (AUC, DROP) saved to {metrics_output_path}")

    # 5. Generate plots
    print("\nGenerating plots...")
    config.ANALYSIS_DIR.mkdir(exist_ok=True)
    plot_accuracy_degradation_curves(agg_df, output_dir=config.ANALYSIS_DIR)

    print("\n--- Phase 3: Analysis Complete! ---")


if __name__ == "__main__":
    analyze_results()
