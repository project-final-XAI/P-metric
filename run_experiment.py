"""
Main script to orchestrate the CROSS-XAI experiment, optimized for parallel GPU processing.

This version uses ThreadPoolExecutor, pre-loads models, and implements
full Batch Processing in both Phase 1 and Phase 2 to maximize GPU utilization
and achieve significant performance improvements.
"""
import os
# Ensure this is set for robustness against MKL/OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import importlib
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Tuple, List

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
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

import run_analysis

warnings.filterwarnings("once")

# --- Global Dictionaries for Worker Processes/Threads ---
worker_data_store: Dict[str, Any] = {}
worker_model_cache: Dict[str, torch.nn.Module] = {}

try:
    # Assuming config, data_loader, model_loader, etc. are correctly set up
    import config
    from utils.data_loader import get_dataloader
    from utils.model_loader import load_model
    # Note: generate_attribution must now support batch input!
    from modules.attribution_generator import generate_attribution
    from modules.occlusion_evaluator import sort_pixels, apply_occlusion, evaluate_judging_model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the project's root directory.")
    exit()

# --- Configuration & Utility Functions ---

def batch_tasks(tasks_list, batch_size):
    """Divides a list of tasks into batches."""
    for i in range(0, len(tasks_list), batch_size):
        yield tasks_list[i:i + batch_size]

try:
    HEATMAP_BATCH_SIZE = config.HEATMAP_BATCH_SIZE
except AttributeError:
    HEATMAP_BATCH_SIZE = 4

try:
    EVALUATION_BATCH_SIZE = config.EVALUATION_BATCH_SIZE
except AttributeError:
    EVALUATION_BATCH_SIZE = 64

# --- Worker Functions for Parallel Processing ---

def process_heatmap_generation_task_batch(task_batch: list[Tuple[str, str, str]]):
    """
    Worker function for Phase 1. Generates a batch of heatmaps for the same model/method.
    Leverages pre-loaded models and batch GPU inference.
    """
    if not task_batch:
        return []

    results = []

    model_name, method_name, _ = task_batch[0]

    try:
        model = worker_model_cache[model_name]
    except KeyError:
        raise RuntimeError(f"Model {model_name} not found in global cache.")

    images_to_process = []
    image_ids_to_process = []
    labels_to_process = []

    for single_task in task_batch:
        current_model_name, current_method_name, image_id = single_task

        heatmap_filename = f"{current_model_name}-{current_method_name}-{image_id}.npy"
        heatmap_path = config.HEATMAP_DIR / heatmap_filename

        if heatmap_path.exists():
            results.append(None)
            continue

        try:
            image_batch, label = worker_data_store['image_label_map'][image_id]
        except KeyError:
            print(f"Image ID {image_id} not found.")
            continue

        images_to_process.append(image_batch)
        image_ids_to_process.append(image_id)
        labels_to_process.append(label)

    if not images_to_process:
        return [r for r in results if r is None]

    # Batch Processing on the GPU
    image_batch_tensor = torch.cat(images_to_process, dim=0).to(config.DEVICE)
    target_classes = torch.tensor(labels_to_process).to(config.DEVICE)

    try:
        with torch.no_grad():
            heatmaps_batch = generate_attribution(
                model=model,
                image=image_batch_tensor,
                target_class=target_classes,
                method_name=method_name
            )
    except Exception as e:
        print(f"Error generating attribution for batch: {e}")
        return results

    if heatmaps_batch is not None:
        # Save each Heatmap separately
        for i, heatmap in enumerate(heatmaps_batch):
            current_id = image_ids_to_process[i]
            heatmap_filename = f"{model_name}-{method_name}-{current_id}.npy"
            heatmap_path = config.HEATMAP_DIR / heatmap_filename
            # Use .numpy() on CPU tensor for saving
            np.save(heatmap_path, heatmap.cpu().numpy())
            results.append(heatmap_filename)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def process_heatmap_evaluation_task(heatmap_paths: List[Path]):
    """
    Worker function for Phase 2. Evaluates a batch of heatmaps.
    Optimizes GPU calls by batching all occlusion evaluations for the same judge model.
    """
    all_results = []

    # 1. Prepare all tasks (CPU-bound)
    tasks_to_run = []
    for heatmap_path in heatmap_paths:
        parts = heatmap_path.stem.split('-')
        gen_model, method, img_id = parts[0], parts[1], parts[2]

        # Load necessary data (CPU/RAM)
        original_image_batch, true_label = worker_data_store['image_label_map'][img_id]
        heatmap = np.load(heatmap_path)
        sorted_pixel_indices = sort_pixels(heatmap)

        # Create all combinations (task triplets: judge, strategy, level)
        for judge_name in config.JUDGING_MODELS:
            if judge_name == gen_model:
                continue

            for strategy in config.FILL_STRATEGIES:
                for p_level in config.OCCLUSION_LEVELS:
                    tasks_to_run.append({
                        'gen_model': gen_model,
                        'method': method,
                        'img_id': img_id,
                        'judge_name': judge_name,
                        'judge_model': worker_data_store['judging_models'][judge_name],
                        'strategy': strategy,
                        'p_level': p_level,
                        'true_label': true_label,
                        'original_image': original_image_batch[0], # tensor [C, H, W]
                        'sorted_pixels': sorted_pixel_indices
                    })

    # 2. Sort and Group for GPU Batching
    # Grouping by judge_name ensures sequential calls to the same model, maximizing cache efficiency
    tasks_to_run.sort(key=lambda x: x['judge_name'])

    current_image_batch = []
    current_meta_data = []
    last_judge_name = None

    for task in tasks_to_run:
        # Check if judge model changed OR batch size limit is reached
        if (task['judge_name'] != last_judge_name and current_image_batch) or \
           len(current_image_batch) >= EVALUATION_BATCH_SIZE:

            # --- GPU Inference (Batch) ---
            if current_image_batch:
                judge_model = current_meta_data[0]['judge_model']

                # Stack and move to GPU
                masked_image_batch = torch.stack(current_image_batch).to(config.DEVICE)

                # Perform batch prediction
                with torch.no_grad():
                    outputs = judge_model(masked_image_batch)
                    predictions = outputs.argmax(dim=1).cpu().tolist()

                # Collect results
                for i, metadata in enumerate(current_meta_data):
                    is_correct = (predictions[i] == metadata['true_label'])
                    all_results.append([
                        metadata['gen_model'], metadata['method'], metadata['img_id'],
                        metadata['judge_name'], metadata['strategy'], metadata['p_level'],
                        is_correct
                    ])

            # Reset for the next batch
            current_image_batch = []
            current_meta_data = []

        # Create the occluded image (CPU-bound operation)
        masked_image = apply_occlusion(
            image=task['original_image'],
            sorted_pixel_indices=task['sorted_pixels'],
            occlusion_level=task['p_level'],
            strategy=task['strategy']
        )
        current_image_batch.append(masked_image.unsqueeze(0).squeeze(0)) # Ensure shape [C, H, W]
        current_meta_data.append(task)
        last_judge_name = task['judge_name']

    # --- Handle the last batch ---
    if current_image_batch:
        judge_model = current_meta_data[0]['judge_model']

        masked_image_batch = torch.stack(current_image_batch).to(config.DEVICE)

        with torch.no_grad():
            outputs = judge_model(masked_image_batch)
            predictions = outputs.argmax(dim=1).cpu().tolist()

        for i, metadata in enumerate(current_meta_data):
            is_correct = (predictions[i] == metadata['true_label'])
            all_results.append([
                metadata['gen_model'], metadata['method'], metadata['img_id'],
                metadata['judge_name'], metadata['strategy'], metadata['p_level'],
                is_correct
            ])

    return all_results

# --- Main Orchestration Functions ---

def run_phase_1_generate_heatmaps(dataset_name: str):
    print("--- Starting Phase 1: Heatmap Generation (GPU Parallel/Threaded with Batching) ---")
    print(f"Limiting to {config.MAX_WORKERS} parallel threads with BATCH_SIZE={HEATMAP_BATCH_SIZE}.")

    # 1. Pre-load Models ONCE
    print("Pre-loading generating models into VRAM...")
    for model_name in config.GENERATING_MODELS:
        if model_name not in worker_model_cache:
            model = load_model(model_name).to(config.DEVICE)
            worker_model_cache[model_name] = model
            print(f"Loaded model: {model_name}")

    # 2. Prepare shared data (Dataset Tensors in RAM)
    print("Loading dataset tensors into RAM...")
    dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
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
        # Group tasks by (model_name, method_name) for efficient Batching
        tasks_to_run.sort(key=lambda x: (x[0], x[1]))

        batched_tasks = []
        current_batch = []
        last_key = None

        for t in tasks_to_run:
            current_key = (t[0], t[1])
            if current_key != last_key and current_batch:
                batched_tasks.extend(list(batch_tasks(current_batch, HEATMAP_BATCH_SIZE)))
                current_batch = []

            current_batch.append(t)
            last_key = current_key

        if current_batch:
            batched_tasks.extend(list(batch_tasks(current_batch, HEATMAP_BATCH_SIZE)))

        print(f"{len(tasks_to_run)} heatmaps to generate. Grouped into {len(batched_tasks)} batches.")

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            list(tqdm(
                executor.map(process_heatmap_generation_task_batch, batched_tasks),
                total=len(batched_tasks),
                desc="Generating Heatmaps (Batched)"
            ))

    print("\n--- Phase 1: Heatmap Generation Complete! ---")


def run_phase_2_evaluate(dataset_name: str):
    print("--- Starting Phase 2: Evaluation (GPU Parallel/Threaded with Batching) ---")
    print(f"Limiting to {config.MAX_WORKERS} parallel threads.")

    # 1. Prepare shared data for workers, including moving models to GPU
    print("Loading dataset and judging models into memory (and VRAM)...")
    dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
    worker_data_store['image_label_map'] = {
        f"image_{i:05d}": (img, lbl.item())
        for i, (img, lbl) in enumerate(dataloader)
    }

    # Load judging models ONCE and move to GPU for the threads
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

    # Batch heatmap files for the threads to process
    THREAD_FILE_BATCH_SIZE = 4
    heatmap_files.sort() # Sorting helps slightly with file I/O locality
    batched_heatmap_files = list(batch_tasks(heatmap_files, THREAD_FILE_BATCH_SIZE))

    print(f"Total Batches of files to process by threads: {len(batched_heatmap_files)}")


    all_results = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        result_iterator = executor.map(process_heatmap_evaluation_task, batched_heatmap_files)
        for result_chunk in tqdm(result_iterator, total=len(batched_heatmap_files), desc="Evaluating Heatmaps"):
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


if __name__ == "__main__":
    # Retain spawn method for robustness with PyTorch/CUDA
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
        run_analysis.analyze_results()