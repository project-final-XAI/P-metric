# CROSS-XAI: Cross-Model Explainable AI Evaluation Framework

A comprehensive framework for evaluating attribution methods using cross-model occlusion-based evaluation. 

Whether you are an absolute beginner to Artificial Intelligence or a seasoned researcher, this document will guide you through everything you need to know: what this project is, why it was built, how it works under the hood, and how you can run it yourself.

---

## 1. Overview and Core Concept

### What is XAI (Explainable AI)?
When complex AI models (like deep neural networks) look at an image and say, *"This is a dog,"* or *"This is a cancerous cell,"* they don't naturally explain *why*. **Explainable AI (XAI)** methods attempt to solve this by generating a "heatmap." A heatmap highlights the specific pixels in the image that the AI looked at to make its decision. 

### The Problem: Who Watches the Watchmen?
There are dozens of different XAI methods to generate these heatmaps. But how do we know if a heatmap is actually correct? What if the XAI method highlights the background grass instead of the dog, but claims the grass is what the AI cared about? Human evaluation is slow, subjective, and prone to error.

### The Solution: CROSS-XAI
This framework evaluates the XAI heatmaps objectively using independent "judging" models. 
Imagine we have a heatmap that claims the dog's snout is the most important part of the image. If we take a black marker and cross out (occlude) the dog's snout, and then show the image to a *second, independent AI judge*, that judge should suddenly become very confused. 
If we hide the "important" pixels and the judge's confidence drops rapidly, the heatmap was right! If we hide the pixels and the judge still easily recognizes the dog, the heatmap was wrong.

---

## 2. Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your datasets in `data/` with class folders (ImageFolder format).

Supported datasets:
- **ImageNet**: `data/imagenet/`
- **SIPaKMeD**: `data/SIPaKMeD/` (medical cell images)
- **Custom**: Add to `config.py` DATASET_CONFIG

### 3. Run Experiment

**Option A: Run all phases**
```bash
python scripts/run_full.py --dataset imagenet
python scripts/run_full.py --dataset SIPaKMeD
```

**Option B: Run individual phases**
```bash
# Phase 1: Generate heatmaps
python scripts/run_phase1.py --dataset imagenet

# Phase 2: Evaluate with occlusion (resumable!)
python scripts/run_phase2.py --dataset imagenet

# Phase 3: Analysis and visualization
python scripts/run_phase3.py                    # All datasets
python scripts/run_phase3.py --dataset imagenet # Specific dataset
```

**🎯 Pro Tip**: Phase 2 can be interrupted and resumed instantly (<0.1s)!

---

## 3. The Core Methodology (The 4-Phase Pipeline)

To make the evaluation scientifically rigorous and highly scalable, the entire architecture is driven by a 4-phase pipeline. Below is the exact methodology loop demonstrating the framework's core operations.

```python
# The Core Methodology Loop of CROSS-XAI

for generating_model in GENERATING_MODELS:
    for xai_method in ATTRIBUTION_METHODS:
        
        # ---------------------------------------------------------
        # PHASE 1: CREATE (Generate Heatmaps)
        # Reason: Isolate heatmap generation to avoid re-computing 
        # expensive gradients for every single judge and occlusion level.
        # ---------------------------------------------------------
        heatmaps = generate_heatmaps(generating_model, xai_method, dataset)
        
        for judging_model in JUDGING_MODELS:
            for fill_strategy in FILL_STRATEGIES: # e.g., gray, blur, random noise
                
                # ---------------------------------------------------------
                # PHASE 2 & 3: OCCLUDE and JUDGE
                # Reason: Progressively hide the "most important" pixels 
                # and ask an independent judge to evaluate the degraded image.
                # ---------------------------------------------------------
                for image, heatmap in zip(dataset, heatmaps):
                    for occlusion_level in OCCLUSION_LEVELS: # e.g., 5%, 10%, 15%...
                        
                        # PHASE 2: Occlude
                        occluded_image = occlude_pixels(
                            image, 
                            heatmap, 
                            occlusion_level, 
                            fill_strategy
                        )
                        
                        # PHASE 3: Judge
                        score = evaluate_with_judge(judging_model, occluded_image)
                        
                        # Save progress incrementally
                        save_result(
                            generating_model, judging_model, xai_method, 
                            fill_strategy, occlusion_level, score
                        )

# ---------------------------------------------------------
# PHASE 4: AGGREGATE AND ANALYZE
# Reason: Calculate the AUC (Area Under the Curve) of the performance 
# degradation curves to objectively rank all the XAI methods.
# ---------------------------------------------------------
aggregate_all_results_and_plot()
```

---

## 4. Architecture

```
P-metric/
├── core/                          # Core orchestration
│   ├── experiment_runner.py      # Main experiment coordinator
│   ├── gpu_manager.py            # GPU resource management
│   ├── file_manager.py           # 🆕 Centralized file management
│   └── progress_tracker.py       # 🆕 Fast resume capability
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

---

## 5. Dataset Integration & Rationale

We designed the framework to evaluate XAI across drastically different visual domains to prove if an XAI method is universally good, or only good at certain tasks.

### ImageNet (Natural Domain)
ImageNet is the gold standard for computer vision benchmarks. It features 1,000 diverse classes ranging from dogs to airplanes. We use it to evaluate how well XAI methods handle complex, naturalistic, multi-object scenes with varying backgrounds, lighting, and occlusions.

### SIPaKMeD (Medical Domain)
SIPaKMeD is a specialized dataset of Pap smear images, consisting of 5 classes of isolated cervical cells (ranging from normal to severely dysplastic). 
* **The Challenge:** Medical images lack distinct outlines (like a dog has) and rely heavily on localized microscopic textures (like nucleus size).
* **Cropped vs Uncropped:** We integrated both the full SIPaKMeD images and `SIPaKMeD_cropped` versions. Cropped images forced the models to look *only* at the cell, mitigating background artifacts. 
* **Custom Training:** Because pre-trained ImageNet models fail to judge medical data accurately, we trained and fine-tuned custom models (`sipakmed_cropped_ResNet50`, `sipakmed_cropped_efficientnet`) specifically for this domain, serving as specialized "judges" in our pipeline.

---

## 6. Comprehensive XAI Methods Explained

The framework supports a massive suite of 11+ XAI techniques. Here is a breakdown of how they work:

### A. Model-Dependent Methods (Using Gradients/Math)
* **Saliency:** Looks at the AI's math and asks, "If I change this pixel slightly, does the AI's output change?" Fast, but often noisy.
* **Input x Gradient:** Multiplies the saliency map by the actual image, sharpening the result.
* **SmoothGrad:** Creates 50 noisy copies of the image, generates heatmaps for all of them, and averages them together to smooth out the noise.
* **Guided Backprop:** Modifies the math to *only* highlight pixels that positively contributed to the answer, ignoring negative pixels.
* **Integrated Gradients (IG):** Solves mathematical flaws in Saliency by drawing a straight line from a black image to the real image, calculating the importance at every step along the way.
* **GradCAM:** Instead of looking at individual pixels, it looks at the final "filters" inside the AI to highlight broad, general regions (like drawing a big circle over the dog).
* **XRAI:** Combines Integrated Gradients with image segmentation, highlighting entire physical regions rather than scattering isolated pixels.

### B. Self-Supervised & Model-Independent Methods (Advanced)
* **DINOv2 (Attention & PCA):** DINOv2 is a revolutionary model from Meta. Unlike normal AI that is trained with labels, DINOv2 learns by looking at billions of images. Because of this, it naturally learns to perfectly separate foreground objects from backgrounds.
* **U2Net (Salient Object Detection):** An architecture specifically built to cut out the foreground object (like Photoshop's magic wand).
* **U2Net + DINO Fusion:** A powerful novel technique blending the crisp, exact boundaries of U2Net with the deep semantic understanding of DINOv2.

---

## 7. Features & Advanced Hardware Optimizations

Generating millions of heatmaps and running iterative occlusions requires supercomputing power. We built sophisticated optimizations to maximize modern hardware (like the RTX 5090).

### ✅ Vision-Language Models (LLM Judges)
Instead of using standard image classifiers as "judges", we integrated Large Language Models (LLMs) via APIs. Standard classifiers just spit out probabilities. LLMs can provide binary judgments (Yes/No) or semantic similarity scores (Cosine Similarity), offering a much more human-like, nuanced understanding of how occlusion affects the image.

### ✅ GPU Manager & Dynamic VRAM Tiering
The system actively probes your Graphics Card (GPU) memory. It categorizes it into tiers: High (>= 22GB), Mid (>= 16GB), and Low (< 8GB). Batch sizes dynamically scale up or down depending on the tier. If an Out-Of-Memory (OOM) error is imminent, the batch size is slashed gracefully instead of crashing.

### ✅ Ultimate Speed Optimizations
* **TensorFloat-32 (TF32):** Explicitly enabled for Ampere/Ada GPUs, yielding up to 3x faster matrix math.
* **Mixed Precision (FP16):** Inference across models defaults to `torch.amp.autocast`, slashing memory footprints by half.
* **Expandable Segments:** Prevents PyTorch from suffering "memory fragmentation" when constantly swapping heavy AI models in and out of VRAM.
* **Intelligent Batching:** Methods like XRAI are detected automatically and processed in micro-batches to ensure stability, while lightweight methods run in massive parallel batches.
* **Lightning-Fast Resume:** The framework saves state instantly to `.progress.json`. If your power goes out, restarting the script will resume exactly where it left off in less than 0.1 seconds.

---

## 8. Configuration

Edit `config.py` to customize your experiment:

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

---

## 9. Output Structure (🆕 Redesigned!)

All results are saved neatly without overwriting previous data.

```
results/
├── heatmaps/                      # Phase 1: Attribution maps
│   ├── imagenet/                  # 🆕 Per-dataset organization
│   │   └── resnet50-saliency-image_00000_sorted.npy
│   └── SIPaKMeD/
│       └── ...
├── evaluation/                     # Phase 2: Evaluation results
│   ├── imagenet/
│   │   ├── .progress.json        # 🆕 Fast resume tracking
│   │   └── {gen_model}/          # 🆕 Hierarchical structure
│   │       └── {judge_model}/
│   │           └── {method}/
│   │               └── {strategy}.csv
│   └── SIPaKMeD/
│       └── ...
└── analysis/                       # Phase 3: Final results
    ├── aggregated_accuracy_curves.csv
    ├── faithfulness_metrics.csv
    ├── imagenet/                  # 🆕 Per-dataset plots
    │   └── *.png
    └── SIPaKMeD/
        └── *.png
```

**Key Improvements**:
- 🎯 Organized by dataset (no overwrites!)
- 📁 Hierarchical structure for easy navigation
- ⚡ Fast resume with `.progress.json`
- 📊 Separate visualizations per dataset

---

## 10. Performance Benchmarks

### Benchmarks (v2.0)

| Feature | Before | After | Improvement |
|---------|--------|-------|-------------|
| **Resume time** | ~60s | <0.1s | **600x faster** ⚡ |
| **Code readability** | 150-line functions | 20-40 line functions | Much cleaner |
| **Multi-dataset** | ❌ Overwrites | ✅ Isolated | Full support |
| **File organization** | 1 huge CSV | Hierarchical structure | Easy navigation |

---

## 11. Project Evolution Timeline

The development of the P-metric framework evolved over five distinct phases, constantly adapting to new research challenges.

### Phase 1: Core Framework (October 2025)
* **The Goal:** Build the foundational pipeline to run CROSS-XAI.
* **The Reality:** We realized that generating heatmaps over and over for every single judge was impossibly slow. We separated Phase 1 (Create) and Phase 2 (Occlude/Judge) so heatmaps were only generated once and saved to disk.

### Phase 2: Hardware Scaling (November 2025)
* **The Goal:** Maximize the RTX 5090 and introduce complex models.
* **The Reality:** VRAM bottlenecks plagued the system. The dynamic GPU manager was born here, allowing us to evaluate heavy transformer models without crashing.

### Phase 3: The LLM Revolution (December 2025 - January 2026)
* **The Goal:** Improve the quality of the "Judges".
* **The Reality:** Standard judges were too rigid. We integrated LLMs that could look at an occluded image and provide deep semantic similarity scores, completely changing the accuracy of our XAI evaluations.

### Phase 4: DINOv2 Integration (February - March 2026)
* **The Goal:** Incorporate self-supervised models.
* **The Reality:** Comparing supervised heatmaps against DINOv2’s natural semantic segmentation provided a massive leap in understanding how well XAI methods actually capture object boundaries.

### Phase 5: U2Net Wrappers & Perfection (April 2026)
* **The Goal:** Handle "continuous" methods and physical boundaries.
* **The Reality:** XAI methods were cheating by highlighting background noise. We built U2Net wrappers to physically constrain evaluations to the foreground object, ensuring the XAI methods were graded strictly on true object identification.

---

## 12. Requirements & Utilities

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended)
- See `requirements.txt` for full list

### Visualize Heatmaps
```bash
# View random heatmaps from ImageNet
python read_heatmap.py --dataset imagenet --num_samples 5

# View heatmaps from SIPaKMeD
python read_heatmap.py --dataset SIPaKMeD --num_samples 3
```

## Documentation & License

- **📖 Complete Documentation**: See `REDESIGN_NOTES.md` for detailed architecture and design decisions
- **📝 Changelog**: See `CHANGELOG.md` for version history
- **🎨 Design Principles**: DRY, Separation of Concerns, Performance-first

See `LICENSE` file for details.