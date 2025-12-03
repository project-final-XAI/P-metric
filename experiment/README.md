# SSMS vs P-Metric Comparison Experiment

This experiment compares **Single-Step Masking Score (SSMS)** with **P-Metric** to determine if SSMS can replace the expensive multi-step P-Metric evaluation.

## Overview

**P-Metric** evaluates faithfulness by progressively occluding pixels at multiple levels (0%, 5%, 10%, ..., 75%) and measuring accuracy degradation. This requires **19+ evaluations per image**.

**SSMS** evaluates faithfulness with a **single adaptive mask** based on the heatmap, requiring only **1 evaluation per image**.

If SSMS correlates strongly with P-Metric, it can provide a **~19x speedup** for faithfulness evaluation.

## Quick Start

### Quick Test (20 images, ~5-10 minutes)
```bash
python experiment/main.py --quick
```

### Full Experiment (300 images, ~1-2 hours)
```bash
python experiment/main.py
```

### Custom Number of Images
```bash
python experiment/main.py --num-images 100
```

## Output Files

### Results CSVs
- `experiment/results_ssms.csv` - SSMS scores and metadata
- `experiment/results_pmetric.csv` - P-Metric metrics (AUC, DROP, InflectionPoints)

### Visualizations
- `experiment/plots/scatter_ssms_vs_auc.png` - Correlation scatter plot
- `experiment/plots/correlation_heatmap.png` - Heatmap by explainer/judge (if multiple)
- `experiment/plots/boxplots_metrics.png` - Distribution comparisons

## Configuration

Edit `experiment/config.py` to customize:
- Number of images
- Models (generating and judging)
- Attribution methods (explainers)
- Occlusion parameters
- SSMS parameters (alpha_max, eps)

## Interpretation

### Correlation Results

The experiment computes **Spearman rank correlation** between SSMS and P-Metric AUC:

- **ρ > 0.8**: **STRONG AGREEMENT** - SSMS can replace P-Metric (~19x speedup)
- **0.6 < ρ ≤ 0.8**: **MODERATE AGREEMENT** - SSMS may be useful but needs validation
- **ρ ≤ 0.6**: **WEAK AGREEMENT** - SSMS may not be suitable replacement

### Statistical Tests

- **Spearman correlation**: Non-parametric rank correlation
- **Paired t-test**: Tests if normalized AUC and SSMS differ significantly
- **Wilcoxon test**: Non-parametric alternative to t-test
- **Cohen's d**: Effect size measure

## How It Works

### SSMS Algorithm
1. Clip heatmap H ≥ 0, normalize to [0, 1]
2. Compute S = sum(H), N = H.size
3. Calculate alpha = min((N - S + eps) / (S + eps), alpha_max)
4. Create mask M = clip(alpha * H, 0, 1)
5. Apply mask: I* = I * M (on normalized image)
6. Evaluate judge model on masked image
7. SSMS_score = 1 if correct, 0 if wrong

### P-Metric Algorithm
1. Sort pixels by heatmap importance (ascending)
2. For each occlusion level P (0%, 5%, 10%, 25%, 50%, 75%):
   - Occlude bottom P% of pixels
   - Evaluate judge model
   - Record accuracy Acc(P)
3. Compute AUC, DROP, InflectionPoints from Acc(P) curve

## Code Structure

```
experiment/
├── main.py              # Entry point
├── config.py            # Configuration
├── metrics_ssms.py      # SSMS implementation
├── metrics_pmetric.py   # P-Metric implementation
├── evaluator.py         # Evaluation pipeline
├── analysis.py          # Statistical analysis
├── requirements.txt     # Additional dependencies
└── README.md            # This file
```

## Dependencies

Most dependencies are in root `requirements.txt`. Additional dependencies:
- `scipy>=1.7.0` (statistical tests)
- `scikit-learn>=1.0.0` (regression for visualization)

Install with:
```bash
pip install -r experiment/requirements.txt
```

## Notes

- The experiment reuses existing infrastructure from the main project:
  - Data loaders (`data/loader.py`)
  - Model loaders (`models/loader.py`)
  - Attribution methods (`attribution/registry.py`)
  - Occlusion utilities (`evaluation/occlusion.py`)
  - Metrics (`evaluation/metrics.py`)

- Heatmaps are generated on-the-fly if not found in `results/heatmaps/`

- Only PyTorch judge models are used (LLM judges excluded for speed)

- Results are saved incrementally, so you can stop and resume if needed

## Troubleshooting

**Out of memory**: Reduce `num_images` or use `--quick` mode

**Slow execution**: Check GPU availability (`config.DEVICE`), reduce number of explainers/judges

**Missing heatmaps**: Heatmaps are generated automatically, but this takes time. Consider running Phase 1 of main pipeline first.


