# run_experiment.py
"""
Main script to orchestrate the CROSS-XAI experiment.

This script is divided into two phases that can be run independently:
- Phase 1: Generate and save all attribution maps (heatmaps).
- Phase 2: Load heatmaps, perform progressive occlusion, evaluate with judging
           models, and save the results to a CSV file.

Usage:
    python run_experiment.py --phase 1
    python run_experiment.py --phase 2
"""
import argparse
import csv

import numpy as np
from tqdm import tqdm

import run_analysis

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


def run_phase_1_generate_heatmaps(dataset_name: str):
    """
    Generates and saves attribution maps for all combinations of generating
    models, attribution methods, and images in the dataset.
    """
    print("--- Starting Phase 1: Heatmap Generation ---")
    config.HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    try:
        dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: Could not load dataset '{dataset_name}'. {e}")
        print("Please ensure your data is correctly set up.")
        return

    # 2. Loop through models and methods
    for model_name in config.GENERATING_MODELS:
        print(f"\nProcessing generating model: {model_name}")
        model = load_model(model_name)

        # Use tqdm for a progress bar over the dataset
        for i, (image_batch, label_batch) in enumerate(tqdm(dataloader, desc=f"Images for {model_name}")):
            image_id = f"image_{i:05d}"

            for method_name in config.ATTRIBUTION_METHODS:
                # Construct a unique path for the heatmap
                heatmap_filename = f"{model_name}-{method_name}-{image_id}.npy"
                heatmap_path = config.HEATMAP_DIR / heatmap_filename

                # Skip if already generated
                if heatmap_path.exists():
                    continue

                # Generate attribution
                heatmap = generate_attribution(
                    model=model,
                    image=image_batch.to(config.DEVICE),
                    target_class=label_batch.item(),
                    method_name=method_name
                )

                # Save the heatmap if generation was successful
                if heatmap is not None:
                    np.save(heatmap_path, heatmap)

    print("\n--- Phase 1: Heatmap Generation Complete! ---")


def run_phase_2_evaluate(dataset_name: str):
    """
    Loads heatmaps, applies occlusion, evaluates with judging models,
    and records the performance.
    """
    print("--- Starting Phase 2: Evaluation ---")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Prepare an output file
    results_csv_path = config.RESULTS_DIR / "evaluation_results.csv"
    csv_header = [
        "generating_model", "attribution_method", "image_id", "judging_model",
        "fill_strategy", "occlusion_level", "is_correct"
    ]

    # Check if a file exists to decide whether to write a header
    write_header = not results_csv_path.exists()

    with open(results_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(csv_header)

        # 2. Load all judging models once to save time
        print("Loading all judging models into memory...")
        judging_models = {name: load_model(name) for name in config.JUDGING_MODELS}

        # 3. Load dataset to get images and labels
        dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
        image_label_map = {f"image_{i:05d}": (img, lbl.item()) for i, (img, lbl) in enumerate(dataloader)}

        # 4. Get a list of all heatmaps to process
        heatmap_files = list(config.HEATMAP_DIR.glob("*.npy"))
        if not heatmap_files:
            print("Error: No heatmaps found. Please run Phase 1 first.")
            return

        # 5. Main evaluation loop
        for heatmap_path in tqdm(heatmap_files, desc="Evaluating Heatmaps"):
            # Parse info from filename
            parts = heatmap_path.stem.split('-')
            gen_model, method, img_id = parts[0], parts[1], parts[2]

            # Load the corresponding image, label, and heatmap
            original_image_batch, true_label = image_label_map[img_id]
            heatmap = np.load(heatmap_path)

            # Sort pixels once per heatmap
            sorted_pixel_indices = sort_pixels(heatmap)

            # Loop through all evaluation configurations
            for judge_name, judge_model in judging_models.items():
                if judge_name == gen_model:
                    continue  # A model cannot judge its own explanations

                for strategy in config.FILL_STRATEGIES:
                    for p_level in config.OCCLUSION_LEVELS:
                        # Apply occlusion
                        masked_image = apply_occlusion(
                            image=original_image_batch[0],  # Remove batch dim for occlusion func
                            sorted_pixel_indices=sorted_pixel_indices,
                            occlusion_level=p_level,
                            strategy=strategy
                        )

                        # Add batch dim back for the model
                        masked_image_batch = masked_image.unsqueeze(0).to(config.DEVICE)

                        # Evaluate
                        is_correct = evaluate_judging_model(
                            judging_model=judge_model,
                            masked_image=masked_image_batch,
                            true_label=true_label
                        )

                        # Write result
                        row = [gen_model, method, img_id, judge_name, strategy, p_level, is_correct]
                        writer.writerow(row)

    print(f"\n--- Phase 2: Evaluation Complete! Results saved to {results_csv_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CROSS-XAI experiment.")
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2],
        help="Which phase of the experiment to run (1: Generate Heatmaps, 2: Evaluate)."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='imagenet',
        help="Name of the dataset to use (as defined in config.py)."
    )
    args = parser.parse_args()

    if args.phase == 1:
        run_phase_1_generate_heatmaps(dataset_name=args.dataset)
    elif args.phase == 2:
        run_phase_2_evaluate(dataset_name=args.dataset)
    else:
        run_phase_1_generate_heatmaps(dataset_name=args.dataset)
        run_phase_2_evaluate(dataset_name=args.dataset)
        run_analysis.analyze_results()
