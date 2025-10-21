"""
Main script to orchestrate the CROSS-XAI experiment, optimized for parallel GPU processing.

This script manages a pool of worker processes, with the number of workers
limited to prevent GPU memory overload. Each worker loads models and data
onto the GPU for efficient computation.
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Dict, Any

import numpy as np
import torch
from tqdm import tqdm

import run_analysis

# --- Global Dictionaries for Worker Processes ---
# These will be initialized in the main process and inherited by workers.
worker_data_store: Dict[str, Any] = {}
worker_model_cache: Dict[str, torch.nn.Module] = {}

try:
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
def process_heatmap_generation_task(args):
    dataset, task_args = args
    model_name, method_name, image_id = task_args
    heatmap_filename = f"{model_name}-{method_name}-{image_id}.npy"
    heatmap_path = config.HEATMAP_DIR / heatmap_filename

    # Ensure the directory exists before saving
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    if heatmap_path.exists():
        return None

    # Load model inside the worker
    model = load_model(model_name).to(config.DEVICE)

    # Load the specific image on demand (not the whole dataset)
    dataloader = get_dataloader(dataset, batch_size=1, shuffle=False)
    for i, (img, lbl) in enumerate(dataloader):
        curr_id = f"image_{i:05d}"
        if curr_id == image_id:
            image_batch, label = img, lbl.item()
            break
    else:
        raise KeyError(f"Image ID {image_id} not found in dataset.")

    heatmap = generate_attribution(
        model=model,
        image=image_batch.to(config.DEVICE),
        target_class=label,
        method_name=method_name
    )

    if heatmap is not None:
        np.save(heatmap_path, heatmap)
    return heatmap_filename


def process_heatmap_evaluation_task(heatmap_path):
    """
    Worker function for Phase 2. Evaluates one heatmap on the GPU.
    """
    results = []
    parts = heatmap_path.stem.split('-')
    gen_model, method, img_id = parts[0], parts[1], parts[2]

    original_image_batch, true_label = worker_data_store['image_label_map'][img_id]
    heatmap = np.load(heatmap_path)
    sorted_pixel_indices = sort_pixels(heatmap)

    for judge_name, judge_model in worker_data_store['judging_models'].items():
        if judge_name == gen_model:
            continue

        for strategy in config.FILL_STRATEGIES:
            for p_level in config.OCCLUSION_LEVELS:
                # Occlusion happens on CPU, then move tensor to GPU
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
                results.append([gen_model, method, img_id, judge_name, strategy, p_level, is_correct])
    return results


# --- Main Orchestration Functions ---

def run_phase_1_generate_heatmaps(dataset_name: str):
    print("--- Starting Phase 1: Heatmap Generation (GPU Parallel) ---")
    print(f"Limiting to {config.MAX_WORKERS} parallel processes to manage VRAM.")

    # ... (rest of the function is the same, but now the executor is limited)
    print("Loading dataset into memory...")
    dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
    worker_data_store['image_label_map'] = {f"image_{i:05d}": (img, lbl.item()) for i, (img, lbl) in
                                            enumerate(dataloader)}
    all_image_ids = worker_data_store['image_label_map'].keys()
    tasks = list(product(config.GENERATING_MODELS, config.ATTRIBUTION_METHODS, all_image_ids))
    tasks_to_run = [t for t in tasks if not (config.HEATMAP_DIR / f"{t[0]}-{t[1]}-{t[2]}.npy").exists()]

    if not tasks_to_run:
        print("All heatmaps already generated.")
    else:
        print(f"{len(tasks_to_run)} heatmaps to generate.")

        tasks_with_dataset = [(dataset_name, t) for t in tasks_to_run]

        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            list(tqdm(
                executor.map(process_heatmap_generation_task, tasks_with_dataset),
                total=len(tasks_with_dataset),
                desc="Generating Heatmaps"
            ))

    print("\n--- Phase 1: Heatmap Generation Complete! ---")


def run_phase_2_evaluate(dataset_name: str):
    print("--- Starting Phase 2: Evaluation (GPU Parallel) ---")
    print(f"Limiting to {config.MAX_WORKERS} parallel processes to manage VRAM.")

    # 1. Prepare shared data for workers, including moving models to GPU
    print("Loading dataset and judging models into memory (and VRAM)...")
    dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
    worker_data_store['image_label_map'] = {f"image_{i:05d}": (img, lbl.item()) for i, (img, lbl) in
                                            enumerate(dataloader)}

    # Load models and move to GPU for the workers
    worker_data_store['judging_models'] = {
        name: load_model(name).to(config.DEVICE) for name in config.JUDGING_MODELS
    }

    heatmap_files = list(config.HEATMAP_DIR.glob("*.npy"))
    if not heatmap_files:
        print("Error: No heatmaps found. Please run Phase 1 first.")
        return
    print(f"Found {len(heatmap_files)} heatmaps to evaluate.")

    all_results = []
    # The key change is here: max_workers is set from config
    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        result_iterator = executor.map(process_heatmap_evaluation_task, heatmap_files)
        for result_chunk in tqdm(result_iterator, total=len(heatmap_files), desc="Evaluating Heatmaps"):
            all_results.extend(result_chunk)

    print(f"Collected {len(all_results)} result rows. Writing to CSV...")
    results_csv_path = config.RESULTS_DIR / "evaluation_results_gpu.csv"
    csv_header = ["generating_model", "attribution_method", "image_id", "judging_model", "fill_strategy",
                  "occlusion_level", "is_correct"]
    with open(results_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(all_results)

    print(f"\n--- Phase 2: Evaluation Complete! Results saved to {results_csv_path} ---")


# (The __main__ block remains the same as your previous version)
if __name__ == "__main__":
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
            "1: Generate Heatmaps (Parallel)\n"
            "2: Evaluate Heatmaps (Parallel)\n"
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
        run_analysis.analyze_results()
