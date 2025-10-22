"""
Main script to orchestrate the CROSS-XAI experiment, optimized for parallel GPU processing.

This script now uses ThreadPoolExecutor and pre-loads models to minimize
I/O and process overhead, improving GPU utilization and overall speed.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Needed for Intel MKL/OpenMP conflicts
import sys
import importlib
import multiprocessing # Still needed for set_start_method

# Ensure safetensors is loaded correctly
try:
    import safetensors
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors"])
    import safetensors

os.environ["PYTHONPATH"] = os.path.dirname(__file__)
sys.modules["safetensors"] = importlib.import_module("safetensors")


import argparse
import csv
import warnings
# --- השינוי העיקרי: מעבר ל-ThreadPoolExecutor ---
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Dict, Any, Tuple

import numpy as np
import torch
from tqdm import tqdm

import run_analysis

warnings.filterwarnings("once")

# --- Global Dictionaries for Worker Processes/Threads ---
# These will be initialized in the main process and shared/inherited by workers/threads.
worker_data_store: Dict[str, Any] = {}
# Dictionary to hold the pre-loaded models on the GPU.
worker_model_cache: Dict[str, torch.nn.Module] = {}

try:
    # Assuming config, data_loader, model_loader, etc. are correctly set up
    import config
    from utils.data_loader import get_dataloader
    from utils.model_loader import load_model
    from modules.attribution_generator import generate_attribution
    from modules.occlusion_evaluator import sort_pixels, apply_occlusion, evaluate_judging_model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the project's root directory.")
    exit()

# --- Worker Functions for Parallel Processing ---

# --- Phase 1 Worker: Now receives required model from the global cache ---
def process_heatmap_generation_task(args: Tuple[str, str, str]) -> str | None:
    """
    Worker function for Phase 1. Generates one heatmap using a pre-loaded model.
    """
    model_name, method_name, image_id = args
    heatmap_filename = f"{model_name}-{method_name}-{image_id}.npy"
    heatmap_path = config.HEATMAP_DIR / heatmap_filename

    heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    if heatmap_path.exists():
        return None
    try:
        model = worker_model_cache[model_name]
    except KeyError:
        raise RuntimeError(f"Model {model_name} not found in global cache.")

    # Load the specific image on demand
    # We assume worker_data_store['image_label_map'] was populated with (img, lbl)
    try:
        image_batch, label = worker_data_store['image_label_map'][image_id]
    except KeyError:
        raise KeyError(f"Image ID {image_id} not found in dataset map.")


    heatmap = generate_attribution(
        model=model,
        image=image_batch.to(config.DEVICE), # Move image to GPU inside the worker
        target_class=label,
        method_name=method_name
    )

    if heatmap is not None:
        np.save(heatmap_path, heatmap)
    return heatmap_filename


# --- Phase 2 Worker: Logic largely remains the same, leveraging shared GPU models ---
def process_heatmap_evaluation_task(heatmap_path):
    """
    Worker function for Phase 2. Evaluates one heatmap on the GPU using shared models.
    """
    results = []
    parts = heatmap_path.stem.split('-')
    gen_model, method, img_id = parts[0], parts[1], parts[2]

    # Use the shared data (pre-loaded on CPU/RAM)
    original_image_batch, true_label = worker_data_store['image_label_map'][img_id]
    heatmap = np.load(heatmap_path)
    sorted_pixel_indices = sort_pixels(heatmap)

    for judge_name, judge_model in worker_data_store['judging_models'].items():
        if judge_name == gen_model:
            continue

        for strategy in config.FILL_STRATEGIES:
            for p_level in config.OCCLUSION_LEVELS:
                # Occlusion happens on CPU (fast), then move tensor to GPU (masked_image)
                masked_image = apply_occlusion(
                    image=original_image_batch[0],
                    sorted_pixel_indices=sorted_pixel_indices,
                    occlusion_level=p_level,
                    strategy=strategy
                ).unsqueeze(0).to(config.DEVICE)

                is_correct = evaluate_judging_model(
                    judging_model=judge_model,
                    masked_image=masked_image,
                    true_label=true_label
                )
                # Store the result locally (in the thread)
                results.append([gen_model, method, img_id, judge_name, strategy, p_level, is_correct])
    return results


# --- Main Orchestration Functions ---

def run_phase_1_generate_heatmaps(dataset_name: str):
    print("--- Starting Phase 1: Heatmap Generation (GPU Parallel/Threaded) ---")
    print(f"Limiting to {config.MAX_WORKERS} parallel threads.")

    # 1. Load ALL generating models ONCE into the shared cache (and VRAM)
    print("Pre-loading generating models into VRAM...")
    for model_name in config.GENERATING_MODELS:
        if model_name not in worker_model_cache:
            model = load_model(model_name).to(config.DEVICE)
            worker_model_cache[model_name] = model
            print(f"Loaded model: {model_name}")

    # 2. Prepare shared data
    print("Loading dataset into memory...")
    dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
    # Store image data (CPU/RAM) for quick access by workers
    worker_data_store['image_label_map'] = {
        f"image_{i:05d}": (img, lbl.item())
        for i, (img, lbl) in enumerate(dataloader)
    }

    # 3. Define tasks and execute
    all_image_ids = worker_data_store['image_label_map'].keys()
    tasks = list(product(config.GENERATING_MODELS, config.ATTRIBUTION_METHODS, all_image_ids))
    tasks_to_run = [t for t in tasks if not (config.HEATMAP_DIR / f"{t[0]}-{t[1]}-{t[2]}.npy").exists()]

    if not tasks_to_run:
        print("All heatmaps already generated.")
    else:
        print(f"{len(tasks_to_run)} heatmaps to generate.")

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            list(tqdm(
                executor.map(process_heatmap_generation_task, tasks_to_run),
                total=len(tasks_to_run),
                desc="Generating Heatmaps"
            ))

    print("\n--- Phase 1: Heatmap Generation Complete! ---")


def run_phase_2_evaluate(dataset_name: str):
    print("--- Starting Phase 2: Evaluation (GPU Parallel/Threaded) ---")
    print(f"Limiting to {config.MAX_WORKERS} parallel threads.")

    # 1. Prepare shared data for workers, including moving models to GPU
    print("Loading dataset and judging models into memory (and VRAM)...")
    dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
    worker_data_store['image_label_map'] = {
        f"image_{i:05d}": (img, lbl.item())
        for i, (img, lbl) in enumerate(dataloader)
    }

    # Load judging models ONCE and move to GPU for the threads
    # This dictionary is shared among all threads.
    worker_data_store['judging_models'] = {
        name: load_model(name).to(config.DEVICE)
        for name in config.JUDGING_MODELS
    }

    # 2. Define tasks and execute
    heatmap_files = list(config.HEATMAP_DIR.glob("*.npy"))
    if not heatmap_files:
        print("Error: No heatmaps found. Please run Phase 1 first.")
        return
    print(f"Found {len(heatmap_files)} heatmaps to evaluate.")

    all_results = []
    # --- שינוי: שימוש ב-ThreadPoolExecutor ---
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        result_iterator = executor.map(process_heatmap_evaluation_task, heatmap_files)
        for result_chunk in tqdm(result_iterator, total=len(heatmap_files), desc="Evaluating Heatmaps"):
            all_results.extend(result_chunk) # Collect results from all threads

    print(f"Collected {len(all_results)} result rows. Writing to CSV...")
    results_csv_path = config.RESULTS_DIR / "evaluation_results_gpu.csv"
    csv_header = ["generating_model", "attribution_method", "image_id", "judging_model", "fill_strategy",
                  "occlusion_level", "is_correct"]
    with open(results_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(all_results)

    print(f"\n--- Phase 2: Evaluation Complete! Results saved to {results_csv_path} ---")


if __name__ == "__main__":
    # Retain spawn method for robustness with PyTorch/CUDA, even though we use threads
    multiprocessing.set_start_method("spawn", force=True)
    print("Using DEVICE:", config.DEVICE)

    parser = argparse.ArgumentParser(
        description="Run the CROSS-XAI experiment with parallel processing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
            '--phase',
        type=int,
        choices=[1, 2, 3],
        help=(
            "Which part of the experiment to run:\n"
            "1: Generate Heatmaps (Parallel/Threaded)\n"
            "2: Evaluate Heatmaps (Parallel/Threaded)\n"
            "3: Run Phase 1, then Phase 2, then analysis"
        )
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='imagenet',
        help="Name of the dataset to use (default: 'imagenet')."
    )
    args = parser.parse_args()

    if args.phase == 1:
        run_phase_1_generate_heatmaps(dataset_name=args.dataset)
    elif args.phase == 2:
        run_phase_2_evaluate(dataset_name=args.dataset)
    else:
        run_phase_1_generate_heatmaps(dataset_name="imagenet")
        run_phase_2_evaluate(dataset_name="imagenet")
        print("\n--- Starting Analysis ---")
        # Ensure run_analysis is available
        run_analysis.analyze_results()