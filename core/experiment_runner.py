"""
Main experiment orchestration for CROSS-XAI evaluation.

Coordinates the 3-phase pipeline:
1. Heatmap generation (attribution maps)
2. Occlusion-based evaluation
3. Metrics calculation and visualization
"""

import os
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import logging
from typing import Dict, List, Tuple, Any

from core.gpu_manager import GPUManager
from attribution.registry import get_attribution_method, get_all_methods
from models.loader import load_model
from data.loader import get_dataloader
from evaluation.occlusion import sort_pixels, apply_occlusion, evaluate_judging_model
from evaluation.metrics import calculate_auc, calculate_drop
from visualization.plotter import plot_accuracy_degradation_curves, plot_fill_strategy_comparison

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ExperimentRunner:
    """Main experiment orchestrator."""
    
    def __init__(self, config):
        self.config = config
        self.gpu_manager = GPUManager()
        self.gpu_manager.print_info()
        
        # Create directories
        self.config.HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Model cache to avoid reloading
        self._model_cache: Dict[str, torch.nn.Module] = {}
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not self.config.GENERATING_MODELS:
            raise ValueError("GENERATING_MODELS cannot be empty")
        if not self.config.JUDGING_MODELS:
            raise ValueError("JUDGING_MODELS cannot be empty")
        if not self.config.ATTRIBUTION_METHODS:
            raise ValueError("ATTRIBUTION_METHODS cannot be empty")
        if not self.config.OCCLUSION_LEVELS:
            raise ValueError("OCCLUSION_LEVELS cannot be empty")
    
    def _get_cached_model(self, model_name: str) -> torch.nn.Module:
        """Load model with caching."""
        if model_name not in self._model_cache:
            # logging.info(f"Loading model: {model_name}")
            self._model_cache[model_name] = load_model(model_name)
        return self._model_cache[model_name]
    
    def _load_existing_results(self) -> set:
        """Load existing results from CSV and return set of completed work items."""
        results_csv_path = self.config.RESULTS_DIR / "evaluation_results.csv"
        completed = set()
        
        if not results_csv_path.exists():
            logging.info("No existing results file found - starting fresh")
            return completed
        
        try:
            with open(results_csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 6:
                        # Create tuple of (gen_model, method, img_id, judge_model, strategy, p_level)
                        key = (row[0], row[1], row[2], row[3], row[4], float(row[5]))
                        completed.add(key)
            logging.info(f"Loaded {len(completed)} existing results from previous runs")
        except Exception as e:
            logging.warning(f"Error loading existing results: {e}")
        
        return completed
    
    def _append_results_to_csv(self, results: List[List], write_header: bool = False):
        """Append batch of results to CSV without rewriting entire file."""
        if not results:
            return
        
        results_csv_path = self.config.RESULTS_DIR / "evaluation_results.csv"
        csv_header = [
            "generating_model", "attribution_method", "image_id", 
            "judging_model", "fill_strategy", "occlusion_level", "is_correct"
        ]
        
        mode = 'w' if write_header else 'a'
        with open(results_csv_path, mode, newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(csv_header)
            writer.writerows(results)
        
        logging.info(f"Appended {len(results)} new results to CSV")
        
    def run_phase_1(self, dataset_name: str):
        """Generate heatmaps for all model-method-image combinations."""
        logging.info(f"Starting Phase 1 - Dataset: {dataset_name}")
        logging.info(f"Models: {len(self.config.GENERATING_MODELS)}")
        logging.info(f"Methods: {len(self.config.ATTRIBUTION_METHODS)}")
        
        # Log storage configuration
        if self.config.SAVE_HEATMAPS:
            logging.info("Saving: Full heatmaps + sorted indices")
        else:
            logging.info("Saving: Sorted indices only")
        
        try:
            # Load dataset once
            dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
            image_label_map = {
                f"image_{i:05d}": (img, lbl.item())
                for i, (img, lbl) in enumerate(dataloader)
            }
            logging.info(f"Loaded {len(image_label_map)} images")
            
            # Process each model
            for model_name in tqdm(self.config.GENERATING_MODELS, desc="Models",leave=False, position=0):
                model = self._get_cached_model(model_name)
                tqdm.write(f"{model_name}")
                
                for i, method_name in enumerate(self.config.ATTRIBUTION_METHODS):
                    attribution_method = f"{i+1}/{len(self.config.ATTRIBUTION_METHODS)}  {method_name}"
                    try:
                        self._process_method_batch(model, model_name, method_name, image_label_map, attribution_method)
                    except Exception as e:
                        logging.error(f"Error processing {model_name}-{method_name}: {e}")
                        continue
                    
            logging.info(f"Heatmaps saved to: {self.config.HEATMAP_DIR}")
        except Exception as e:
            logging.error(f"Phase 1 failed: {e}")
            raise
        
    def _process_method_batch(self, model, model_name, method_name, image_label_map, attribution_method=""):
        """Process a batch of images for specific model-method combination."""
        method = get_attribution_method(method_name)
        batch_size = self.gpu_manager.get_batch_size(method_name)
        
        # Collect images to process
        images_to_process = []
        image_ids = []
        labels = []
        
        for img_id, (img, label) in image_label_map.items():
            sorted_path = self.config.HEATMAP_DIR / f"{model_name}-{method_name}-{img_id}_sorted.npy"
            
            # Check what files are required based on config
            files_missing = not sorted_path.exists()  # sorted indices always required
            
            if self.config.SAVE_HEATMAPS:
                heatmap_path = self.config.HEATMAP_DIR / f"{model_name}-{method_name}-{img_id}.npy"
                files_missing = files_missing or not heatmap_path.exists()
            
            # Process if any required file is missing
            if files_missing:
                images_to_process.append(img)
                image_ids.append(img_id)
                labels.append(label)
                
        if not images_to_process:
            # logging.info(f"{attribution_method} already processed")
            return
            
        # Process in batches
        for i in tqdm(range(0, len(images_to_process), batch_size), 
                      desc=attribution_method, leave=False, position=0):
            end_idx = min(i + batch_size, len(images_to_process))
            
            # Concatenate images (they already have batch dim from dataloader)
            batch_images = torch.cat(images_to_process[i:end_idx], dim=0).to(self.config.DEVICE)
            batch_labels = torch.tensor(labels[i:end_idx]).to(self.config.DEVICE)
            
            # Generate attributions with mixed precision for faster computation
            if self.config.DEVICE == "cuda":
                with torch.amp.autocast(self.config.DEVICE):
                    heatmaps = method(model, batch_images, batch_labels)
            else:
                heatmaps = method(model, batch_images, batch_labels)
            
            if heatmaps is not None:
                # Save heatmaps and/or sorted indices based on config
                for j, heatmap in enumerate(heatmaps):
                    img_id = image_ids[i + j]
                    heatmap_np = heatmap.cpu().numpy()
                    
                    # Save full heatmap only if requested (saves ~50% disk space if False)
                    if self.config.SAVE_HEATMAPS:
                        heatmap_path = self.config.HEATMAP_DIR / f"{model_name}-{method_name}-{img_id}.npy"
                        np.save(heatmap_path, heatmap_np)
                    
                    # Cache sorted pixel indices

                    sorted_indices = sort_pixels(heatmap_np)
                    sorted_path = self.config.HEATMAP_DIR / f"{model_name}-{method_name}-{img_id}_sorted.npy"
                    np.save(sorted_path, sorted_indices)
                    
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def run_phase_2(self, dataset_name: str):
        """Evaluate heatmaps with occlusion."""
        logging.info(f"Starting Phase 2 - Dataset: {dataset_name}")
        
        try:
            # Load existing results to enable resume
            completed_work = self._load_existing_results()
            write_header = len(completed_work) == 0  # Write header only if starting fresh
            
            # Load dataset and judging models
            dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
            image_label_map = {
                f"image_{i:05d}": (img, lbl.item())
                for i, (img, lbl) in enumerate(dataloader)
            }
            
            judging_models = {
                name: self._get_cached_model(name) for name in self.config.JUDGING_MODELS
            }
            
            # Get heatmap files
            heatmap_files = list(self.config.HEATMAP_DIR.glob("*.npy"))
            logging.info(f"Found {len(heatmap_files)} heatmaps to evaluate")
            
            if not heatmap_files:
                logging.warning("No heatmaps found. Run Phase 1 first.")
                return
            
            # Group heatmaps by generator and method for batch processing
            heatmap_groups = self._group_heatmaps(heatmap_files)
            
            logging.info("Processing heatmap groups...")
            for i, (group_key, group_files) in enumerate(heatmap_groups.items()):
                try:
                    group_name = f"{i+1}/{len(heatmap_groups)}  {group_key[0]}, {group_key[1]}:"
                    results = self._evaluate_heatmap_batch(
                        group_files, image_label_map, judging_models, 
                        completed_work, group_name
                    )
                    
                    # Write results incrementally after each group
                    if results:
                        self._append_results_to_csv(results, write_header=write_header)
                        write_header = False  # Only write header once
                        
                        # Add new results to completed set to avoid duplicates
                        for result in results:
                            key = (result[0], result[1], result[2], result[3], result[4], result[5])
                            completed_work.add(key)
                    
                except Exception as e:
                    logging.error(f"Error processing group {group_key}: {e}")
                    continue
                
            logging.info(f"Phase 2 complete! Results saved to: {self.config.RESULTS_DIR}")
        except Exception as e:
            logging.error(f"Phase 2 failed: {e}")
            raise
        
    def _group_heatmaps(self, heatmap_files: List[Path]) -> Dict[Tuple[str, str], List[Path]]:
        """Group heatmap files by generator model and method for batch processing."""
        groups = {}
        for heatmap_path in heatmap_files:
            parts = heatmap_path.stem.split('-')
            if len(parts) >= 3:
                gen_model, method = parts[0], parts[1]
                key = (gen_model, method)
                if key not in groups:
                    groups[key] = []
                groups[key].append(heatmap_path)
        return groups
    
    def _evaluate_heatmap_batch(self, heatmap_paths: List[Path], image_label_map, judging_models, completed_work: set, group_name = ""):
        """Evaluate a batch of heatmaps (same generator and method) efficiently."""
        results = []
        batch_size = 8  # Process multiple images at once
        
        total_skipped = 0
        total_processed = 0
        
        for i in tqdm(range(0, len(heatmap_paths), batch_size), 
                      desc=group_name, leave=False, position=0):
            batch_paths = heatmap_paths[i:i+batch_size]
            
            # Prepare batch data
            batch_data = []
            for heatmap_path in batch_paths:
                try:
                    parts = heatmap_path.stem.split('-')
                    gen_model, method, img_id = parts[0], parts[1], '-'.join(parts[2:])
                    
                    original_image, true_label = image_label_map[img_id]
                    
                    # Load cached sorted indices if available (much faster!)
                    sorted_path = heatmap_path.with_name(heatmap_path.stem + "_sorted.npy")
                    if sorted_path.exists():
                        sorted_pixel_indices = np.load(sorted_path)
                    else:
                        # Fallback: compute if not cached
                        heatmap = np.load(heatmap_path)
                        sorted_pixel_indices = sort_pixels(heatmap)
                        logging.warning(f"Sorted indices not found for {heatmap_path.name}, computing on-the-fly")
                    
                    batch_data.append({
                        'gen_model': gen_model,
                        'method': method,
                        'img_id': img_id,
                        'image': original_image,
                        'label': true_label,
                        'sorted_indices': sorted_pixel_indices
                    })
                except Exception as e:
                    logging.warning(f"Error loading {heatmap_path}: {e}")
                    continue
            
            if not batch_data:
                continue
            
            # Evaluate for each judge, strategy, and level
            for judge_name, judge_model in judging_models.items():
                gen_model = batch_data[0]['gen_model']
                if judge_name == gen_model:
                    continue
                
                for strategy in self.config.FILL_STRATEGIES:
                    for p_level in self.config.OCCLUSION_LEVELS:
                        # Process batch of images together
                        batch_images = []
                        batch_labels = []
                        batch_info = []
                        
                        for data in batch_data:
                            # Check if this specific evaluation already exists
                            work_key = (data['gen_model'], data['method'], data['img_id'], 
                                       judge_name, strategy, float(p_level))
                            
                            if work_key in completed_work:
                                total_skipped += 1
                                continue
                            
                            try:
                                masked_image = apply_occlusion(
                                    image=data['image'][0],
                                    sorted_pixel_indices=data['sorted_indices'],
                                    occlusion_level=p_level,
                                    strategy=strategy
                                )
                                batch_images.append(masked_image)
                                batch_labels.append(data['label'])
                                batch_info.append(data)
                            except Exception as e:
                                logging.warning(f"Error applying occlusion: {e}")
                                continue
                        
                        if not batch_images:
                            continue
                        
                        # Stack and evaluate in batch with mixed precision
                        try:
                            batch_tensor = torch.stack(batch_images).to(self.config.DEVICE)
                            
                            with torch.no_grad():
                                # Use FP16 for faster inference 
                                if self.config.DEVICE == "cuda":
                                    with torch.amp.autocast(self.config.DEVICE):
                                        outputs = judge_model(batch_tensor)
                                else:
                                    outputs = judge_model(batch_tensor)
                                    
                                if isinstance(outputs, tuple):
                                    outputs = outputs[0]
                                if isinstance(outputs, dict):
                                    outputs = outputs['logits']
                                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                            
                            # Record results
                            for idx, (pred, info) in enumerate(zip(predictions, batch_info)):
                                is_correct = 1 if pred == batch_labels[idx] else 0
                                results.append([
                                    info['gen_model'], info['method'], info['img_id'], 
                                    judge_name, strategy, p_level, is_correct
                                ])
                                total_processed += 1
                        except Exception as e:
                            logging.warning(f"Batch evaluation error: {e}, falling back to single")
                            # Fallback to single evaluation
                            for idx, info in enumerate(batch_info):
                                try:
                                    masked_image = batch_images[idx].unsqueeze(0).to(self.config.DEVICE)
                                    
                                    # Evaluate with FP16 if available
                                    with torch.no_grad():
                                        if self.config.DEVICE == "cuda":
                                            with torch.amp.autocast(self.config.DEVICE):
                                                outputs = judge_model(masked_image)
                                        else:
                                            outputs = judge_model(masked_image)
                                        
                                        if isinstance(outputs, tuple):
                                            outputs = outputs[0]
                                        if isinstance(outputs, dict):
                                            outputs = outputs['logits']
                                        pred = torch.argmax(outputs, dim=1).item()
                                        is_correct = 1 if pred == batch_labels[idx] else 0
                                    results.append([
                                        info['gen_model'], info['method'], info['img_id'], 
                                        judge_name, strategy, p_level, is_correct
                                    ])
                                    total_processed += 1
                                except Exception as e2:
                                    logging.warning(f"Single evaluation failed: {e2}")
                                    continue
        
        # Log summary of work done
        if total_skipped > 0 or total_processed > 0:
            logging.info(f"Group complete: {total_processed} new evaluations, {total_skipped} already completed (skipped)")
        
        return results
    
    def run_phase_3(self):
        """Run analysis and visualization."""
        logging.info("Starting Phase 3 - Analysis and visualization")
        
        try:
            # Load results
            results_csv_path = self.config.RESULTS_DIR / "evaluation_results.csv"
            if not results_csv_path.exists():
                logging.error(f"Results file not found at {results_csv_path}")
                return
                
            import pandas as pd
            df = pd.read_csv(results_csv_path)
            
            if df.empty:
                logging.warning("Results CSV is empty")
                return
            
            # Calculate aggregated accuracy
            group_cols = [
                "generating_model", "attribution_method", "judging_model",
                "fill_strategy", "occlusion_level"
            ]
            agg_df = df.groupby(group_cols)['is_correct'].mean().reset_index()
            agg_df.rename(columns={'is_correct': 'mean_accuracy'}, inplace=True)
            
            # Calculate metrics
            metrics_list = []
            curve_group_cols = ["generating_model", "attribution_method", "judging_model", "fill_strategy"]
            
            for name, curve_df in agg_df.groupby(curve_group_cols):
                try:
                    gen_model, method, judge_model, fill_strat = name
                    
                    baseline_acc = curve_df[curve_df['occlusion_level'] == 0]['mean_accuracy'].iloc[0] if 0 in curve_df['occlusion_level'].values else 1.0
                    
                    accuracies = curve_df['mean_accuracy'].tolist()
                    levels = curve_df['occlusion_level'].tolist()
                    
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
                except Exception as e:
                    logging.warning(f"Error calculating metrics for {name}: {e}")
                    continue
                
            # Save results
            metrics_df = pd.DataFrame(metrics_list)
            agg_output_path = self.config.ANALYSIS_DIR / "aggregated_accuracy_curves.csv"
            metrics_output_path = self.config.ANALYSIS_DIR / "faithfulness_metrics.csv"
            
            agg_df.to_csv(agg_output_path, index=False)
            metrics_df.to_csv(metrics_output_path, index=False)
            
            # Generate plots
            plot_accuracy_degradation_curves(agg_df, output_dir=self.config.ANALYSIS_DIR)
            
            # Generate fill strategy comparison plot
            plot_fill_strategy_comparison(agg_df, output_dir=self.config.ANALYSIS_DIR)
            
            logging.info(f"Analysis complete! Results saved to {self.config.ANALYSIS_DIR}")
        except Exception as e:
            logging.error(f"Phase 3 failed: {e}")
            raise
