# CROSS-XAI: Cross-Model Explainable AI Evaluation Framework

A comprehensive framework for evaluating attribution methods using cross-model occlusion-based evaluation.

## Overview

This framework implements the CROSS-XAI methodology to objectively evaluate the faithfulness of XAI attribution methods by measuring their impact on independent "judging" models through progressive pixel occlusion.

## Architecture

```
P-metric/
├── core/                          # Core orchestration
│   ├── experiment_runner.py      # Main experiment coordinator
│   └── gpu_manager.py            # GPU resource management
├── attribution/                   # XAI methods (11 total)
│   ├── base.py                   # Base class and adapter pattern
│   ├── gradient_based.py         # Saliency, Input×Gradient, SmoothGrad
│   ├── integration_based.py      # Integrated Gradients, GradientSHAP
│   ├── cam_based.py              # GradCAM, Guided GradCAM
│   ├── perturbation_based.py     # Occlusion, XRAI
│   ├── other.py                  # Guided Backprop, Random Baseline
│   └── registry.py               # Method registry
├── models/                        # Model utilities
│   ├── loader.py                 # Model loading
│   └── architectures.py          # Layer selection
├── evaluation/                    # Evaluation utilities
│   ├── occlusion.py              # Occlusion strategies
│   └── metrics.py                # AUC, DROP calculations
├── data/                          # Data utilities
│   └── loader.py                 # Dataset loading
├── visualization/                 # Plotting utilities
│   └── plotter.py                # Accuracy degradation curves
├── scripts/                       # Entry points
│   ├── run_phase1.py             # Heatmap generation
│   ├── run_phase2.py             # Occlusion evaluation
│   ├── run_phase3.py             # Analysis and visualization
│   └── run_full.py               # Complete experiment
└── config.py                      # Configuration
```

## Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your ImageNet subset in `data/imagenet/` with class folders.

### 3. Run Experiment

**Option A: Run all phases**
```bash
python scripts/run_full.py --dataset imagenet
```

**Option B: Run individual phases**
```bash
# Phase 1: Generate heatmaps
python scripts/run_phase1.py --dataset imagenet

# Phase 2: Evaluate with occlusion
python scripts/run_phase2.py --dataset imagenet

# Phase 3: Analysis and visualization
python scripts/run_phase3.py
```

## Features

### ✅ All 11 Attribution Methods
- **Gradient-based**: Saliency, Input×Gradient, SmoothGrad
- **Integration-based**: Integrated Gradients, GradientSHAP
- **CAM-based**: GradCAM, Guided GradCAM
- **Perturbation-based**: Occlusion, XRAI
- **Other**: Guided Backprop, Random Baseline

### ✅ Multi-Model Support
- **CNN**: ResNet50, MobileNetV2, VGG16
- **Transformer**: ViT-B/16, Swin-T
- **Automatic layer selection** for CAM methods

### ✅ Intelligent Batching
- **Batch processing** where supported
- **Micro-batching** for memory-intensive methods
- **Single-image fallback** for incompatible methods
- **GPU-aware optimization** with automatic batch size adjustment

### ✅ Resume Capability
- **Automatic detection** of existing heatmaps
- **Skip completed work** and continue from where stopped
- **No data loss** during interruptions

## Configuration

Edit `config.py` to customize:

```python
# Models
GENERATING_MODELS = ["resnet50", "mobilenet_v2", "vit_b_16", "swin_t"]
JUDGING_MODELS = ["resnet50", "vit_b_16", "swin_t"]

# Attribution methods
ATTRIBUTION_METHODS = [
    "saliency", "integrated_gradients", "grad_cam", 
    "occlusion", "xrai", "random_baseline"
]

# Occlusion settings
OCCLUSION_LEVELS = list(range(5, 100, 5))  # 5%, 10%, ..., 95%
FILL_STRATEGIES = ["gray", "blur", "random_noise"]
```

## Output Structure

```
results/
├── heatmaps/                      # Phase 1: Attribution maps
│   ├── resnet50-saliency-image_00000.npy
│   └── ...
├── evaluation/                     # Phase 2: Evaluation results
│   └── evaluation_results.csv
└── analysis/                       # Phase 3: Final results
    ├── aggregated_accuracy_curves.csv
    ├── faithfulness_metrics.csv
    └── *.png                      # Accuracy degradation plots
```

## Performance

- **CPU/GPU compatible**: Works on both CPU and GPU (auto-detected)
- **GPU-optimized**: Automatic batch size adjustment based on GPU memory
- **Batch processing**: Phase 2 now processes multiple images simultaneously (up to 10x faster)
- **Model caching**: Models loaded once and reused across methods
- **Memory efficient**: Micro-batching for memory-intensive methods
- **Resume-friendly**: Continue from any interruption point
- **Progress tracking**: Detailed progress bars and logging for all operations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended)
- See `requirements.txt` for full list

## License

See LICENSE file for details.