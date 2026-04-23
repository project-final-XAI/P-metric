# CROSS-XAI: Cross-Model Explainable AI Evaluation Framework

A framework for evaluating attribution methods using cross-model, occlusion-based faithfulness testing.

This README is intentionally detailed and aligned with the **current** codebase.

---

## 1) Why This Project Exists

XAI methods produce heatmaps that claim to show which pixels mattered for a model decision.  
The hard part is not generating heatmaps, it is **validating** them.

Human inspection is useful but subjective. CROSS-XAI turns this into a measurable process:

1. Rank pixels by method importance.
2. Progressively remove top-ranked pixels.
3. Ask independent judging models how performance degrades.
4. Convert degradation curves into objective metrics.

If removing "important" pixels rapidly hurts judging performance, the heatmap is more faithful.

---

## 1.1) Intuition With a Concrete Example

Suppose an image contains a fish, and an attribution method says the fish body is most important.

- At 0% occlusion, judges should classify correctly.
- At 20% occlusion, if we remove top-ranked fish pixels, accuracy should start dropping.
- At 60% occlusion, a faithful method should usually make judges struggle.
- At 90-100% occlusion, performance should be very low.

If the curve stays high while "important" pixels are removed, that usually means the method ranking was not truly aligned with model evidence.

---

## 2) High-Level Pipeline (4 Phases)

The pipeline is orchestrated by `run_all.py` and runs:

- **Phase 1** (`core/phase1_runner.py`): generate and save heatmaps.
- **Phase 2** (`core/phase2_runner.py`): generate occluded images using sorted heatmap pixels.
- **Phase 3** (`core/phase3_runner.py`): score occluded images with judging models.
- **Phase 4** (`core/phase4_runner.py`): aggregate curves, compute metrics, and create plots/reports.

Why this separation matters:

- Phase 1 is expensive; caching avoids recomputing heatmaps for every judge/strategy.
- Phases 2/3 can be iterated repeatedly using the same cached heatmaps.
- Phase 4 can be rerun alone after experiments complete.

---

## 2.1) Core Methodology Loop (Code View)

```python
# CROSS-XAI conceptual loop (pseudocode)

for generating_model in GENERATING_MODELS:
    for xai_method in ATTRIBUTION_METHODS:
        # ---------------------------------------------------------
        # PHASE 1: CREATE HEATMAPS
        # ---------------------------------------------------------
        heatmaps = generate_heatmaps(
            model=generating_model,
            method=xai_method,
            dataset=DATASET_NAME,
        )

        for judging_model in JUDGING_MODELS:
            for fill_strategy in FILL_STRATEGIES:
                for image, heatmap in zip(dataset_images, heatmaps):
                    for occlusion_level in OCCLUSION_LEVELS:
                        # -------------------------------------------------
                        # PHASE 2: OCCLUDE USING HEATMAP RANKING
                        # -------------------------------------------------
                        occluded_image = occlude_pixels(
                            image=image,
                            heatmap=heatmap,
                            occlusion_level=occlusion_level,
                            fill_strategy=fill_strategy,
                        )

                        # -------------------------------------------------
                        # PHASE 3: JUDGE PERFORMANCE
                        # -------------------------------------------------
                        score = evaluate_with_judge(
                            judge_model=judging_model,
                            image=occluded_image,
                        )

                        save_result(
                            dataset=DATASET_NAME,
                            gen_model=generating_model,
                            judge_model=judging_model,
                            method=xai_method,
                            strategy=fill_strategy,
                            level=occlusion_level,
                            score=score,
                        )

# ---------------------------------------------------------
# PHASE 4: AGGREGATE + METRICS + PLOTS
# ---------------------------------------------------------
aggregate_curves_and_metrics()
export_analysis_tables_and_plots()
```

This pseudocode mirrors the actual runner split in the codebase:

- `core/phase1_runner.py`
- `core/phase2_runner.py`
- `core/phase3_runner.py`
- `core/phase4_runner.py`

---

## 3) Current Architecture (Code-Accurate)

```text
P-metric/
├── run_all.py                             # Main entry point (phases 1→4)
├── config.py                              # Experiment configuration
├── core/
│   ├── phase1_runner.py
│   ├── phase2_runner.py
│   ├── phase3_runner.py
│   ├── phase4_runner.py
│   ├── file_manager.py                    # Path contracts for all phases
│   ├── gpu_manager.py                     # Batch sizing / thermal checks
│   └── gpu_utils.py
├── attribution/
│   ├── base.py
│   ├── registry.py
│   ├── model_dependent/
│   ├── model_independent/
│   │   ├── dinov2_methods.py
│   │   ├── unet_based.py
│   │   └── unet_dino.py
│   └── continuous/
├── evaluation/
│   ├── occlusion.py
│   ├── metrics.py
│   └── judging/
├── scripts/
│   ├── clear_method_cache.py
│   ├── create_excel_reports.py
│   └── create_imgenet_data.py
└── test/
    ├── DINOv2/
    └── U2Net/
```

---

## 4) End-to-End Methodology Details

### Phase 1: Heatmap Generation

For each `(generating_model, attribution_method, image)`:

- method is loaded from `attribution.registry`
- heatmap is computed
- two outputs are saved:
  - visual PNG (`regular`)
  - sorted pixel indices (`sorted`) for occlusion order

This sorted representation is key: later phases do not need to recompute method internals.

### Phase 2: Progressive Occlusion

Using sorted pixel indices from Phase 1:

- apply occlusion levels from `config.OCCLUSION_LEVELS`
- apply multiple fill strategies from `config.FILL_STRATEGIES`
- save occluded images per strategy and level

### Phase 3: Judging

Each occluded image is scored by each judging model.  
Results are written to CSVs per `(gen_model, judge_model, method, strategy)`.

### Phase 4: Aggregation + Metrics

Phase 4 loads all CSVs and computes:

- aggregated accuracy-vs-occlusion curves
- faithfulness metrics such as AUC and DROP
- dataset-specific plots
- optional Excel reports (via `scripts/create_excel_reports.py`)

Why AUC and DROP are useful together:

- **AUC** summarizes the full degradation curve (global behavior).
- **DROP** focuses on a specific high-occlusion point (local stress test).
- Using both avoids over-trusting methods that perform well only in one region of the curve.

---

## 5) Attribution Method Families

Methods are configured in `config.py` and instantiated via `attribution/registry.py`.

- **Model-dependent** methods (gradients/activations):
  - `saliency`, `inputxgradient`, `guided_backprop`, `integrated_gradients`, `gradientshap`, `occlusion`, `xrai`, `grad_cam`, `guided_gradcam`, `random_baseline`, `c3f`
- **Model-independent** methods:
  - DINO variants (`dinov2_attention`, `dinov2_PC1`, `dinov2_PC_EV`, `dinov2_PC_L2`, `dinov2_COMBO_FIXED`, `dinov2_ENT`, `dinov2_COMBO_ENT_SMOOTH`)
  - U2Net methods (`U2Net-Saliency`, `u2net_dino_fusion`)
- **Wrappers**:
  - continuous smoothing wrappers
  - U2Net-underlay wrappers for model-dependent methods

---

## 6) DINOv2 and U2Net Implementation Notes

### DINO attention backend

In current code, DINO methods require attention outputs:

```python
DINO_ATTN_IMPLEMENTATION = "eager"
```

Using SDPA can result in missing attention tensors (`attentions=None`) for methods that depend on `output_attentions=True`.

### U2Net behavior

U2Net is salient-object style and can produce strong foreground masks.  
Visual appearance depends heavily on normalization and colormap (for example `hot` can look very white/red on saturated maps).

Practical consequence:

- U2Net maps can look "too hot" visually but still be useful for ranking foreground vs background.
- DINO maps often look more textured and gradient-like.
- Fusion (`u2net_dino_fusion`) is intended to combine boundary strength (U2Net) with semantic texture (DINO).

---

## 7) Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure experiment

Edit in `config.py`:

- dataset (`DATASET_NAME`, `DATASET_CONFIG`)
- generating models (`GENERATING_MODELS`)
- judging models (`JUDGING_MODELS`)
- attribution methods (`ATTRIBUTION_METHODS`)
- occlusion settings (`OCCLUSION_LEVELS`, `FILL_STRATEGIES`)

### Run full pipeline

```bash
python run_all.py
```

### Run only Phase 4

```bash
python core/phase4_runner.py
```

Use this when Phase 3 outputs already exist and you only want refreshed analysis.

---

## 8) Output Structure (Current)

```text
results/
├── heatmaps/                              # Phase 1
│   └── {dataset}/{gen_model}/{method}/
│       ├── regular/*.png
│       └── sorted/*.npy
├── occluded/                              # Phase 2
│   └── {dataset}/{gen_model}/{strategy}/{method}/...
├── evaluation/                            # Phase 3
│   └── {dataset}/{gen_model}/{judge_model}/{method}/{strategy}.csv
└── analysis/                              # Phase 4
    ├── aggregated_accuracy_curves.csv
    ├── faithfulness_metrics.csv
    └── {dataset}/*.png
```

---

## 8.1) How to Read Results Correctly

When comparing methods, avoid judging by PNG appearance alone.

Recommended reading order:

1. Open `results/analysis/faithfulness_metrics.csv`.
2. Filter by one dataset + one generating model + one judging model.
3. Compare methods under the same fill strategy first.
4. Validate with the per-dataset plots in `results/analysis/{dataset}/`.

Interpretation tips:

- A method can produce visually "pretty" heatmaps but weak faithfulness scores.
- A mask-like method can look saturated yet perform strongly in occlusion metrics.
- Fill strategy changes absolute numbers; compare methods under equal strategy conditions.

---

## 9) Clear Cached Data for Specific Methods

`scripts/clear_method_cache.py` is currently parameter-baked (no CLI args).

Edit constants:

- `DATASET`
- `METHODS_TO_CLEAR`
- `INCLUDE_EVALUATION`
- `DRY_RUN`

Run:

```bash
python scripts/clear_method_cache.py
```

Behavior:

- searches/deletes under:
  - `results/heatmaps/{dataset}`
  - `results/occluded/{dataset}`
- optionally also:
  - `results/evaluation/{dataset}` (if `INCLUDE_EVALUATION=True`)
- if `DRY_RUN=True`, no deletion is performed; only targets are listed

---

## 9.1) Reproducibility and Re-Runs

If you update a method implementation, clear its cached artifacts before re-running comparisons.

Typical safe workflow:

1. Set `DRY_RUN=True` in `scripts/clear_method_cache.py` and verify targets.
2. Set `DRY_RUN=False` and delete.
3. Re-run `python run_all.py` (or just required phases).
4. Rebuild Phase 4 outputs for fresh metrics.

This prevents stale heatmaps/occlusions from contaminating new method evaluations.

---

## 10) Troubleshooting

- **`ModuleNotFoundError: No module named 'core'` when running Phase 4**
  - Run from repo root, or use current `core/phase4_runner.py` bootstrap.
- **DINO errors related to attentions / `NoneType`**
  - Ensure `DINO_ATTN_IMPLEMENTATION = "eager"` in `config.py`.
- **U2Net device mismatch (`cpu` vs `cuda`)**
  - Current `unet_based.py` aligns inference tensors to model device; restart run after pulling latest changes.
- **Heatmaps look too hot/white**
  - This can be expected for mask-like methods (U2Net) with aggressive colormap scaling.
  - Try changing `HEATMAP_COLORMAP` for visualization-only differences.

---

## 11) Project Evolution Timeline

The framework evolved in multiple waves, each driven by bottlenecks discovered during large experiment runs.

### Stage 1: Initial CROSS-XAI Pipeline

- Built the first end-to-end idea: generate heatmaps, occlude by importance, and judge degradation.
- Main insight: visual heatmaps are not enough; metric-based faithfulness is required.

### Stage 2: Separation into Explicit Phases

- Split workflow into distinct runners to avoid recomputing expensive heatmaps for every downstream variant.
- This enabled reusing Phase 1 artifacts while iterating quickly on judging and analysis settings.

### Stage 3: GPU + Throughput Optimization

- Added GPU management utilities and dynamic batching behavior.
- Improved stability under long jobs with varied method costs and model memory profiles.

### Stage 4: Method Expansion (DINO/U2Net/Wrappers)

- Added model-independent DINOv2 and U2Net families.
- Added fusion and wrapper methods to compare semantic, boundary-focused, and hybrid attribution behavior.

### Stage 5: Reliability and Cache-Centric Iteration

- Strengthened cache-based workflow (`heatmaps`, `occluded`, `evaluation`, `analysis`) so phases can be rerun selectively.
- Added targeted method cleanup (`scripts/clear_method_cache.py`) for reproducible re-runs after method/config changes.

### Stage 6: Reporting and Interpretability Workflow

- Consolidated metric aggregation and visualization in Phase 4.
- Added tabular + plot outputs to support method ranking, cross-model comparison, and dataset-specific interpretation.

---

## 12) Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA recommended for throughput
- See `requirements.txt` for full dependency list

## 13) License

See `LICENSE`.