"""
Statistical analysis and visualization for SSMS vs P-Metric comparison.

Computes correlations, statistical tests, and creates visualizations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict
import logging

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(results_dir: Path) -> pd.DataFrame:
    """
    Load and merge SSMS and P-Metric results.
    
    Args:
        results_dir: Directory containing result CSV files
        
    Returns:
        Merged DataFrame with both SSMS and P-Metric metrics
    """
    ssms_path = results_dir / "results_ssms.csv"
    pmetric_path = results_dir / "results_pmetric.csv"
    
    if not ssms_path.exists():
        raise FileNotFoundError(f"SSMS results not found: {ssms_path}")
    if not pmetric_path.exists():
        raise FileNotFoundError(f"P-Metric results not found: {pmetric_path}")
    
    # Load CSVs
    ssms_df = pd.read_csv(ssms_path)
    pmetric_df = pd.read_csv(pmetric_path)
    
    if len(ssms_df) == 0:
        raise ValueError("SSMS results CSV is empty")
    if len(pmetric_df) == 0:
        raise ValueError("P-Metric results CSV is empty")
    
    # Merge on (image_id, explainer, judge_model)
    merged = pd.merge(
        ssms_df,
        pmetric_df,
        on=['image_id', 'explainer', 'judge_model'],
        how='inner'
    )
    
    if len(merged) == 0:
        raise ValueError("No matching records found between SSMS and P-Metric results")
    
    logging.info(f"Loaded {len(merged)} merged results")
    return merged


def compute_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute statistical correlations and tests between SSMS and P-Metric.
    
    Args:
        df: Merged DataFrame with SSMS and P-Metric metrics
        
    Returns:
        Dictionary with correlation statistics
    """
    results = {}
    
    # Spearman rank correlation: AUC vs SSMS_score
    spearman_corr, spearman_p = spearmanr(df['AUC'], df['SSMS_score'])
    results['spearman_rho'] = spearman_corr
    results['spearman_pvalue'] = spearman_p
    
    # Pearson correlation (linear relationship)
    pearson_corr, pearson_p = stats.pearsonr(df['AUC'], df['SSMS_score'])
    results['pearson_r'] = pearson_corr
    results['pearson_pvalue'] = pearson_p
    
    # Paired t-test: differences between normalized AUC and SSMS
    # Normalize both to [0, 1] for fair comparison
    auc_norm = (df['AUC'] - df['AUC'].min()) / (df['AUC'].max() - df['AUC'].min() + 1e-8)
    ssms_norm = df['SSMS_score']  # Already in [0, 1]
    
    t_stat, t_p = ttest_rel(auc_norm, ssms_norm)
    results['ttest_statistic'] = t_stat
    results['ttest_pvalue'] = t_p
    
    # Wilcoxon signed-rank test (non-parametric)
    try:
        wilcoxon_stat, wilcoxon_p = wilcoxon(auc_norm, ssms_norm)
        results['wilcoxon_statistic'] = wilcoxon_stat
        results['wilcoxon_pvalue'] = wilcoxon_p
    except ValueError:
        # All differences are zero
        results['wilcoxon_statistic'] = 0.0
        results['wilcoxon_pvalue'] = 1.0
    
    # Cohen's d effect size
    differences = auc_norm - ssms_norm
    std_diff = np.std(differences, ddof=1)
    if std_diff > 1e-8:
        cohens_d = np.mean(differences) / std_diff
    else:
        cohens_d = 0.0  # No variation
    results['cohens_d'] = cohens_d
    
    # Agreement thresholds
    if abs(spearman_corr) > 0.8:
        agreement_level = "strong"
    elif abs(spearman_corr) > 0.6:
        agreement_level = "moderate"
    else:
        agreement_level = "weak"
    results['agreement_level'] = agreement_level
    
    return results


def create_visualizations(df: pd.DataFrame, plots_dir: Path):
    """
    Create visualization plots.
    
    Args:
        df: Merged DataFrame with results
        plots_dir: Directory to save plots
    """
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Scatter plot: SSMS vs AUC with regression line
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df['AUC'], df['SSMS_score'], alpha=0.5, s=50)
    
    # Add regression line
    X = df['AUC'].values.reshape(-1, 1)
    y = df['SSMS_score'].values
    reg = LinearRegression().fit(X, y)
    x_line = np.linspace(df['AUC'].min(), df['AUC'].max(), 100)
    y_line = reg.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'Regression (R²={reg.score(X, y):.3f})')
    
    ax.set_xlabel('P-Metric AUC', fontsize=12)
    ax.set_ylabel('SSMS Score', fontsize=12)
    ax.set_title('SSMS vs P-Metric AUC Correlation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "scatter_ssms_vs_auc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap (if multiple explainers/judges)
    if len(df['explainer'].unique()) > 1 or len(df['judge_model'].unique()) > 1:
        # Group by explainer and judge_model, compute mean correlation
        grouped = df.groupby(['explainer', 'judge_model']).agg({
            'AUC': 'mean',
            'SSMS_score': 'mean'
        }).reset_index()
        
        # Create pivot table for heatmap
        pivot_auc = grouped.pivot(index='explainer', columns='judge_model', values='AUC')
        pivot_ssms = grouped.pivot(index='explainer', columns='judge_model', values='SSMS_score')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='viridis', ax=axes[0])
        axes[0].set_title('Mean AUC by Explainer × Judge', fontweight='bold')
        
        sns.heatmap(pivot_ssms, annot=True, fmt='.3f', cmap='viridis', ax=axes[1])
        axes[1].set_title('Mean SSMS Score by Explainer × Judge', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Boxplots: AUC and SSMS distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].boxplot(df['AUC'], vert=True)
    axes[0].set_ylabel('AUC', fontsize=12)
    axes[0].set_title('P-Metric AUC Distribution', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(df['SSMS_score'], vert=True)
    axes[1].set_ylabel('SSMS Score', fontsize=12)
    axes[1].set_title('SSMS Score Distribution', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "boxplots_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Acc(P) curves: sample curves for different explainers
    # This would require storing the full accuracy curves, which we don't have in CSV
    # For now, skip this or create a simplified version
    
    logging.info(f"Saved visualizations to {plots_dir}")


def print_summary(stats_dict: Dict[str, float], df: pd.DataFrame):
    """
    Print summary statistics.
    
    Args:
        stats_dict: Dictionary with correlation statistics
        df: Merged DataFrame with results
    """
    print("\n" + "="*70)
    print(" " * 20 + "SSMS vs P-Metric Comparison Summary")
    print("="*70)
    
    print(f"\nDataset: {len(df)} image-explainer-judge combinations")
    print(f"Explainers: {df['explainer'].nunique()} ({', '.join(df['explainer'].unique())})")
    print(f"Judge Models: {df['judge_model'].nunique()} ({', '.join(df['judge_model'].unique())})")
    
    print("\n" + "-"*70)
    print("CORRELATION ANALYSIS")
    print("-"*70)
    print(f"Spearman Rank Correlation (ρ): {stats_dict['spearman_rho']:.4f}")
    print(f"  p-value: {stats_dict['spearman_pvalue']:.4e}")
    print(f"Pearson Correlation (r): {stats_dict['pearson_r']:.4f}")
    print(f"  p-value: {stats_dict['pearson_pvalue']:.4e}")
    
    print("\n" + "-"*70)
    print("STATISTICAL TESTS")
    print("-"*70)
    print(f"Paired t-test statistic: {stats_dict['ttest_statistic']:.4f}")
    print(f"  p-value: {stats_dict['ttest_pvalue']:.4e}")
    print(f"Wilcoxon signed-rank test: {stats_dict['wilcoxon_statistic']:.4f}")
    print(f"  p-value: {stats_dict['wilcoxon_pvalue']:.4e}")
    print(f"Cohen's d (effect size): {stats_dict['cohens_d']:.4f}")
    
    print("\n" + "-"*70)
    print("AGREEMENT ASSESSMENT")
    print("-"*70)
    agreement = stats_dict['agreement_level']
    rho = stats_dict['spearman_rho']
    print(f"Agreement Level: {agreement.upper()} (ρ = {rho:.4f})")
    
    if abs(rho) > 0.8:
        print("✓ STRONG AGREEMENT: SSMS shows strong correlation with P-Metric.")
        print("  → SSMS can potentially replace expensive P-Metric evaluation (~19x speedup)")
    elif abs(rho) > 0.6:
        print("⚠ MODERATE AGREEMENT: SSMS shows moderate correlation with P-Metric.")
        print("  → SSMS may be useful but requires further validation")
    else:
        print("✗ WEAK AGREEMENT: SSMS shows weak correlation with P-Metric.")
        print("  → SSMS may not be a suitable replacement for P-Metric")
    
    print("\n" + "-"*70)
    print("METRIC STATISTICS")
    print("-"*70)
    print(f"P-Metric AUC:")
    print(f"  Mean: {df['AUC'].mean():.4f}, Std: {df['AUC'].std():.4f}")
    print(f"  Min: {df['AUC'].min():.4f}, Max: {df['AUC'].max():.4f}")
    print(f"\nSSMS Score:")
    print(f"  Mean: {df['SSMS_score'].mean():.4f}, Std: {df['SSMS_score'].std():.4f}")
    print(f"  Min: {df['SSMS_score'].min():.4f}, Max: {df['SSMS_score'].max():.4f}")
    
    print("\n" + "="*70 + "\n")

