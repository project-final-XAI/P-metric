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
from collections import defaultdict

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.progress_tracker import ProgressTracker
from attribution.registry import get_attribution_method, get_all_methods
from models.loader import load_model
from data.loader import get_dataloader
from evaluation.occlusion import sort_pixels, apply_occlusion, evaluate_judging_model
from evaluation.metrics import calculate_auc, calculate_drop
from visualization.plotter import plot_accuracy_degradation_curves, plot_fill_strategy_comparison

# Setup logging (separate from tqdm stdout)
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)




class ExperimentRunner:
    """Main experiment orchestrator."""
    
    def __init__(self, config):
        self.config = config
        self.gpu_manager = GPUManager()
        self.gpu_manager.print_info()
        
        # Initialize file manager for centralized file operations
        self.file_manager = FileManager(config.BASE_DIR)
        
        # Create base directories
        self.file_manager.ensure_dir_exists(self.file_manager.heatmap_dir)
        self.file_manager.ensure_dir_exists(self.file_manager.results_dir)
        self.file_manager.ensure_dir_exists(self.file_manager.analysis_dir)
        
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
        if not self.config.FILL_STRATEGIES:
            raise ValueError("FILL_STRATEGIES cannot be empty")
    
    def _get_cached_model(self, model_name: str) -> torch.nn.Module:
        """Load model with caching."""
        if model_name not in self._model_cache:
            # logging.info(f"Loading model: {model_name}")
            self._model_cache[model_name] = load_model(model_name)
        return self._model_cache[model_name]
        
    def run_phase_1(self, dataset_name: str):
        """Generate heatmaps for all model-method-image combinations."""
        # Validate dataset name
        if dataset_name not in self.config.DATASET_CONFIG:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_CONFIG. "
                f"Available datasets: {list(self.config.DATASET_CONFIG.keys())}"
            )
        
        logging.info(f"Starting Phase 1 - Dataset: {dataset_name}")
        logging.info(f"Models: {len(self.config.GENERATING_MODELS)} | Methods: {len(self.config.ATTRIBUTION_METHODS)} | Storage: {'Full+Sorted' if self.config.SAVE_HEATMAPS else 'Sorted only'}")
        
        try:
            # Ensure dataset heatmap directory exists
            heatmap_dir = self.file_manager.get_heatmap_dir(dataset_name)
            self.file_manager.ensure_dir_exists(heatmap_dir)
            
            # Load dataset once
            dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
            image_label_map = {
                f"image_{i:05d}": (img, lbl.item())
                for i, (img, lbl) in enumerate(dataloader)
            }
            
            # Process each model with progress bar
            total_combinations = len(self.config.GENERATING_MODELS) * len(self.config.ATTRIBUTION_METHODS)
            with tqdm(total=total_combinations, desc="Phase 1 Progress") as pbar:
                for model_idx, model_name in enumerate(self.config.GENERATING_MODELS, 1):
                    model = self._get_cached_model(model_name)
                    
                    for method_idx, method_name in enumerate(self.config.ATTRIBUTION_METHODS, 1):
                        pbar.set_description(f"[{model_idx}/{len(self.config.GENERATING_MODELS)}] {model_name[:12]} | [{method_idx}/{len(self.config.ATTRIBUTION_METHODS)}] {method_name[:15]}")
                        try:
                            self._process_method_batch(
                                model, model_name, method_name, 
                                image_label_map, dataset_name
                            )
                        except Exception as e:
                            logging.error(f"Error: {model_name}-{method_name}: {e}")
                        finally:
                            pbar.update(1)
                    
            logging.info(f"Heatmaps saved to: {heatmap_dir}")
        except Exception as e:
            logging.error(f"Phase 1 failed: {e}")
            raise
        
    def _process_method_batch(
        self, model, model_name, method_name, 
        image_label_map, dataset_name
    ):
        """Process a batch of images for specific model-method combination."""
        method = get_attribution_method(method_name)
        batch_size = self.gpu_manager.get_batch_size(method_name)
        
        # Collect images to process
        images_to_process = []
        image_ids = []
        labels = []
        
        for img_id, (img, label) in image_label_map.items():
            # Use FileManager to get paths with dataset
            sorted_path = self.file_manager.get_heatmap_path(
                dataset_name, model_name, method_name, img_id, sorted=True
            )
            
            # Check what files are required based on config
            files_missing = not sorted_path.exists()  # sorted indices always required
            
            if self.config.SAVE_HEATMAPS:
                heatmap_path = self.file_manager.get_heatmap_path(
                    dataset_name, model_name, method_name, img_id, sorted=False
                )
                files_missing = files_missing or not heatmap_path.exists()
            
            # Process if any required file is missing
            if files_missing:
                images_to_process.append(img)
                image_ids.append(img_id)
                labels.append(label)
                
        if not images_to_process:
            return
            
        # Process in batches with inner progress bar
        for i in tqdm(range(0, len(images_to_process), batch_size),
                      desc=f"  → Processing {len(images_to_process)} images", dynamic_ncols=True):
            end_idx = min(i + batch_size, len(images_to_process))
            
            # Concatenate images
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
                        heatmap_path = self.file_manager.get_heatmap_path(
                            dataset_name, model_name, method_name, img_id, sorted=False
                        )
                        np.save(heatmap_path, heatmap_np)
                    
                    # Cache sorted pixel indices
                    sorted_indices = sort_pixels(heatmap_np)
                    sorted_path = self.file_manager.get_heatmap_path(
                        dataset_name, model_name, method_name, img_id, sorted=True
                    )
                    np.save(sorted_path, sorted_indices)
                    
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def run_phase_2(self, dataset_name: str):
        """Evaluate heatmaps with occlusion."""
        # Validate dataset name
        if dataset_name not in self.config.DATASET_CONFIG:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_CONFIG. "
                f"Available datasets: {list(self.config.DATASET_CONFIG.keys())}"
            )
        
        completed_str = ""
        
        try:
            # Initialize progress tracker for fast resume
            with ProgressTracker(self.file_manager, dataset_name) as progress:
                completed_count = progress.get_completed_count()
                if completed_count > 0:
                    completed_str = f" | Resuming: {completed_count:,} done"
                
                logging.info(f"Starting Phase 2 - Dataset: {dataset_name}{completed_str}")
                
                # Load dataset and judging models
                dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
                image_label_map = {
                    f"image_{i:05d}": (img, lbl.item())
                    for i, (img, lbl) in enumerate(dataloader)
                }
                
                judging_models = {
                    name: self._get_cached_model(name) for name in self.config.JUDGING_MODELS
                }
                
                # Get heatmap files for this dataset
                heatmap_dir = self.file_manager.get_heatmap_dir(dataset_name)
                if not heatmap_dir.exists():
                    logging.warning(f"No heatmaps found for {dataset_name}. Run Phase 1 first.")
                    return
                
                all_files = list(heatmap_dir.glob("*.npy"))
                heatmap_files = [f for f in all_files if f.stem.endswith("_sorted")]
                
                if not heatmap_files:
                    logging.warning(f"No heatmaps found for {dataset_name}. Run Phase 1 first.")
                    return
                
                # Group heatmaps by generator and method for batch processing
                heatmap_groups = self._group_heatmaps(heatmap_files)
                
                # Process with progress bar
                with tqdm(total=len(heatmap_groups), desc="Phase 2 Progress", unit="group", leave=True) as pbar:
                    for i, (group_key, group_files) in enumerate(heatmap_groups.items(), 1):
                        gen_model, method = group_key
                        pbar.set_description(f"Phase 2 [{i}/{len(heatmap_groups)}] {gen_model[:10]}/{method[:12]}")
                        try:
                            self._evaluate_heatmap_batch(
                                group_files, image_label_map, judging_models,
                                progress, dataset_name, gen_model, method
                            )
                        except Exception as e:
                            logging.error(f"Error processing {gen_model}-{method}: {e}")
                        finally:
                            pbar.update(1)
                
                result_dir = self.file_manager.get_result_dir(dataset_name)
                logging.info(f"Phase 2 complete! Results saved to: {result_dir}")
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
    
    def _evaluate_heatmap_batch(
        self, heatmap_paths: List[Path], image_label_map,
        judging_models, progress: ProgressTracker, 
        dataset_name: str, gen_model: str, method: str
    ):
        """Evaluate a batch of heatmaps (same generator and method) efficiently."""
        # Prepare batch data once
        batch_data = self._prepare_batch_data(heatmap_paths, image_label_map)
        if not batch_data:
            return
        
        # Evaluate for each judge and strategy
        for judge_name, judge_model in judging_models.items():
            if judge_name == gen_model:
                continue
            
            for strategy in self.config.FILL_STRATEGIES:
                # Evaluate this specific combination
                self._evaluate_strategy(
                    batch_data, judge_model, judge_name, strategy,
                    progress, dataset_name, gen_model, method
                )
    
    def _prepare_batch_data(
        self, heatmap_paths: List[Path], image_label_map
    ) -> List[Dict]:
        """Prepare batch data from heatmap paths."""
        batch_data = []
        for heatmap_path in heatmap_paths:
            try:
                parts = heatmap_path.stem.split('-')
                gen_model, method, img_id = parts[0], parts[1], '-'.join(parts[2:])

                if img_id.endswith('_sorted'):
                    img_id = img_id[:-len('_sorted')]
                
                # Check if image exists in dataset
                if img_id not in image_label_map:
                    logging.warning(f"Image {img_id} not found in dataset, skipping heatmap {heatmap_path.name}")
                    continue
                
                original_image, true_label = image_label_map[img_id]
                
                # Load cached sorted indices

                # if sorted_path.exists():
                sorted_pixel_indices = np.load(heatmap_path)
                # else:
                #     # Fallback: compute if not cached
                #     heatmap = np.load(heatmap_path)
                #     sorted_pixel_indices = sort_pixels(heatmap)
                #     logging.warning(f"Sorted indices not found for {heatmap_path.name}")
                #
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
        
        return batch_data
    
    def _evaluate_strategy(
        self, batch_data: List[Dict], judge_model, judge_name: str,
        strategy: str, progress: ProgressTracker, dataset_name: str,
        gen_model: str, method: str
    ):
        """Evaluate a specific judge/strategy combination for all images."""
        # Results grouped by occlusion level
        results_by_level = defaultdict(list)
        total_skipped = 0
        total_processed = 0
        
        # Inner progress bar for occlusion levels (only if many levels)
        show_inner = len(self.config.OCCLUSION_LEVELS) > 10
        for p_level in tqdm(self.config.OCCLUSION_LEVELS, 
                           desc=f"  → {judge_name[:8]}/{strategy[:6]}", 
                           leave=False, disable=not show_inner):
            # Process batch of images together
            batch_images = []
            batch_labels = []
            batch_info = []
            
            for data in batch_data:
                # Check if already completed
                if progress.is_completed(
                    data['gen_model'], data['method'], data['img_id'],
                    judge_name, strategy, p_level
                ):
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
            
            # Evaluate batch
            predictions = self._evaluate_batch(batch_images, judge_model)
            
            # Record results
            for idx, (pred, info) in enumerate(zip(predictions, batch_info)):
                # Skip invalid predictions (failed evaluations)
                if pred is None or pred < 0:
                    logging.error(f"Skipping invalid prediction for {info['img_id']} at level {p_level}")
                    continue
                
                is_correct = 1 if pred == batch_labels[idx] else 0
                results_by_level[p_level].append([
                    info['img_id'], p_level, is_correct
                ])
                
                # Mark as completed in progress tracker
                progress.mark_completed(
                    info['gen_model'], info['method'], info['img_id'],
                    judge_name, strategy, p_level
                )
                total_processed += 1
        
        # Save results to file
        if results_by_level:
            self._save_strategy_results(
                results_by_level, dataset_name, gen_model,
                judge_name, method, strategy
            )
            progress.save()  # Save progress after each strategy
    
    def _evaluate_batch(
        self, batch_images: List[torch.Tensor], judge_model
    ) -> np.ndarray:
        """Evaluate a batch of images with the judging model."""
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
            
            return predictions
        except Exception as e:
            # logging.warning(f"Batch evaluation error: {e}, falling back to single") # TODO: FIXXXXXXXXXX!
            # Fallback to single evaluation
            predictions = []
            for img in batch_images:
                try:
                    single_tensor = img.unsqueeze(0).to(self.config.DEVICE)
                    with torch.no_grad():
                        if self.config.DEVICE == "cuda":
                            with torch.amp.autocast(self.config.DEVICE):
                                outputs = judge_model(single_tensor)
                        else:
                            outputs = judge_model(single_tensor)
                        
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        if isinstance(outputs, dict):
                            outputs = outputs['logits']
                        pred = torch.argmax(outputs, dim=1).item()
                    predictions.append(pred)
                except Exception as e2:
                    logging.warning(f"Single evaluation failed: {e2}")
                    predictions.append(None)  # Invalid prediction marker
            return np.array(predictions, dtype=object)
    
    def _save_strategy_results(
        self, results_by_level: Dict, dataset_name: str,
        gen_model: str, judge_model: str, method: str, strategy: str
    ):
        """Save results for a specific strategy to CSV file."""
        # Get file path
        result_file = self.file_manager.get_result_file_path(
            dataset_name, gen_model, judge_model, method, strategy
        )
        
        # Flatten results
        all_results = []
        for level in sorted(results_by_level.keys()):
            all_results.extend(results_by_level[level])
        
        # Save to CSV (append mode if file exists)
        header = ["image_id", "occlusion_level", "is_correct"]
        self.file_manager.save_csv(
            result_file, all_results, header=header, append=result_file.exists()
        )
    
    def run_phase_3(self, dataset_name: str = None):
        """
        Run analysis and visualization.
        
        Args:
            dataset_name: If specified, analyze only this dataset.
                         If None, analyze all datasets.
        """
        try:
            import pandas as pd
            
            # Determine which datasets to analyze
            if dataset_name:
                datasets = [dataset_name]
            else:
                # Check if results directory exists
                if not self.file_manager.results_dir.exists():
                    logging.error(f"Results directory does not exist: {self.file_manager.results_dir}")
                    return
                
                # Find all datasets with results
                datasets = [
                    d.name for d in self.file_manager.results_dir.iterdir()
                    if d.is_dir() and not d.name.startswith('.')
                ]
            
            if not datasets:
                logging.error("No result datasets found")
                return
            
            logging.info(f"Starting Phase 3 - Analyzing: {', '.join(datasets)}")
            
            # Load all results from file structure
            all_data = []
            for dataset in datasets:
                result_files = self.file_manager.scan_result_files(dataset)
                
                for result_file in result_files:
                    # Parse file path to get parameters
                    params = self.file_manager.parse_result_file_path(result_file, dataset)
                    if not params:
                        continue
                    
                    # Load CSV data
                    rows = self.file_manager.load_csv(result_file, skip_header=True)
                    
                    # Add metadata to each row
                    for row in rows:
                        if len(row) >= 3:
                            try:
                                all_data.append({
                                    'dataset': dataset,
                                    'generating_model': params['gen_model'],
                                    'attribution_method': params['method'],
                                    'judging_model': params['judge_model'],
                                    'fill_strategy': params['strategy'],
                                    'image_id': row[0],
                                    'occlusion_level': float(row[1]),
                                    'is_correct': int(row[2])
                                })
                            except (ValueError, TypeError) as e:
                                logging.warning(f"Skipping corrupted row in {result_file}: {row} - {e}")
                                continue
            
            if not all_data:
                logging.warning("No results data found")
                return
            
            # Create DataFrame
            df = pd.DataFrame(all_data)
            
            # Calculate aggregated accuracy
            group_cols = [
                "dataset", "generating_model", "attribution_method",
                "judging_model", "fill_strategy", "occlusion_level"
            ]
            agg_df = df.groupby(group_cols)['is_correct'].mean().reset_index()
            agg_df.rename(columns={'is_correct': 'mean_accuracy'}, inplace=True)
            
            # Calculate metrics
            metrics_list = []
            curve_group_cols = [
                "dataset", "generating_model", "attribution_method",
                "judging_model", "fill_strategy"
            ]
            
            for name, curve_df in agg_df.groupby(curve_group_cols):
                try:
                    dataset, gen_model, method, judge_model, fill_strat = name
                    
                    baseline_acc = 1.0
                    if 0 in curve_df['occlusion_level'].values:
                        baseline_acc = curve_df[curve_df['occlusion_level'] == 0]['mean_accuracy'].iloc[0]
                    
                    accuracies = curve_df['mean_accuracy'].tolist()
                    levels = curve_df['occlusion_level'].tolist()
                    
                    auc = calculate_auc(accuracies, levels)
                    drop75 = calculate_drop(accuracies, levels, initial_accuracy=baseline_acc, drop_level=75)
                    
                    metrics_list.append({
                        "dataset": dataset,
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
            agg_output_path = self.file_manager.analysis_dir / "aggregated_accuracy_curves.csv"
            metrics_output_path = self.file_manager.analysis_dir / "faithfulness_metrics.csv"
            
            agg_df.to_csv(agg_output_path, index=False)
            metrics_df.to_csv(metrics_output_path, index=False)
            
            # Generate plots per dataset
            for dataset in datasets:
                dataset_df = agg_df[agg_df['dataset'] == dataset].copy()
                if not dataset_df.empty:
                    # Create dataset-specific output directory
                    dataset_analysis_dir = self.file_manager.analysis_dir / dataset
                    self.file_manager.ensure_dir_exists(dataset_analysis_dir)
                    
                    plot_accuracy_degradation_curves(
                        dataset_df, output_dir=dataset_analysis_dir
                    )
                    plot_fill_strategy_comparison(
                        dataset_df, output_dir=dataset_analysis_dir
                    )
            
            logging.info(f"Phase 3 complete! Results → {self.file_manager.analysis_dir}")
        except Exception as e:
            logging.error(f"Phase 3 failed: {e}")
            raise
