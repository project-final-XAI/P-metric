"""
Phase 1: Heatmap Generation Runner.

Generates attribution heatmaps for all model-method-image combinations.
This phase creates sorted pixel indices files that are used in Phase 2 for occlusion evaluation.
"""

import numpy as np
import torch
import logging
from tqdm import tqdm
from typing import Dict, Any

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.gpu_utils import transfer_to_device, clear_cache_if_needed
from attribution.registry import get_attribution_method
from data.loader import get_dataloader
from evaluation.occlusion import sort_pixels


class Phase1Runner:
    """Handles Phase 1: Heatmap generation for all model-method-image combinations."""
    
    def __init__(
        self,
        config,
        gpu_manager: GPUManager,
        file_manager: FileManager,
        model_cache: Dict[str, Any]
    ):
        """
        Initialize Phase 1 runner.
        
        Args:
            config: Configuration object
            gpu_manager: GPU resource manager
            file_manager: File path manager
            model_cache: Shared model cache dictionary
        """
        self.config = config
        self.gpu_manager = gpu_manager
        self.file_manager = file_manager
        self.model_cache = model_cache
    
    def run(self, get_cached_model_func):
        """
        Generate heatmaps for all model-method-image combinations.
        
        Args:
            get_cached_model_func: Function to get cached model by name
        """
        dataset_name = self.config.DATASET_NAME
        
        # Validate dataset name
        if dataset_name not in self.config.DATASET_CONFIG:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_CONFIG. "
                f"Available datasets: {list(self.config.DATASET_CONFIG.keys())}"
            )
        
        logging.info(f"Starting Phase 1 - Dataset: {dataset_name}")
        logging.info(
            f"Models: {len(self.config.GENERATING_MODELS)} | "
            f"Methods: {len(self.config.ATTRIBUTION_METHODS)}"
        )
        
        try:
            # Ensure dataset heatmap directory exists
            heatmap_dir = self.file_manager.get_heatmap_dir(dataset_name)
            self.file_manager.ensure_dir_exists(heatmap_dir)
            
            # Load dataset once
            dataloader = get_dataloader(dataset_name, batch_size=32, shuffle=False)
            image_label_map = {
                f"image_{i:05d}": (img, lbl.item())
                for i, (img, lbl) in enumerate(dataloader)
            }
            
            # Process each model-method combination
            total_combinations = len(self.config.GENERATING_MODELS) * len(self.config.ATTRIBUTION_METHODS)
            with tqdm(total=total_combinations, desc="Phase 1 Progress") as pbar:
                for model_idx, model_name in enumerate(self.config.GENERATING_MODELS, 1):
                    model = get_cached_model_func(model_name)
                    
                    for method_idx, method_name in enumerate(self.config.ATTRIBUTION_METHODS, 1):
                        pbar.set_description(
                            f"[{model_idx}/{len(self.config.GENERATING_MODELS)}] {model_name[:12]} | "
                            f"[{method_idx}/{len(self.config.ATTRIBUTION_METHODS)}] {method_name[:15]}"
                        )
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
        self,
        model: Any,
        model_name: str,
        method_name: str,
        image_label_map: Dict[str, tuple],
        dataset_name: str
    ):
        """
        Process a batch of images for specific model-method combination.
        
        Args:
            model: Loaded model instance
            model_name: Name of the model
            method_name: Name of the attribution method
            image_label_map: Dictionary mapping image IDs to (image, label) tuples
            dataset_name: Name of the dataset
        """
        method = get_attribution_method(method_name)
        batch_size = self.gpu_manager.get_batch_size(method_name)
        
        # Collect images that need processing (skip if already processed)
        images_to_process = []
        image_ids = []
        labels = []
        
        for img_id, (img, label) in list(image_label_map.items()):
            sorted_path = self.file_manager.get_heatmap_path(
                dataset_name, model_name, method_name, img_id, sorted=True
            )
            
            # Only process if sorted indices file doesn't exist
            if not sorted_path.exists():
                images_to_process.append(img)
                image_ids.append(img_id)
                labels.append(label)
        
        if not images_to_process:
            return
        
        # Process in batches with progress bar
        for i in tqdm(
            range(0, len(images_to_process), batch_size),
            desc=f"  → Processing {len(images_to_process)} images",
            dynamic_ncols=True
        ):
            # Periodic thermal check every 5 batches
            if i > 0 and i % (batch_size * 5) == 0:
                self.gpu_manager.check_and_throttle()
            
            end_idx = min(i + batch_size, len(images_to_process))
            
            # Prepare batch with GPU optimizations
            batch_images = torch.cat(images_to_process[i:end_idx], dim=0)
            batch_images = transfer_to_device(
                batch_images,
                self.config.DEVICE,
                non_blocking=True,
                memory_format=torch.channels_last
            )
            batch_labels = torch.tensor(labels[i:end_idx]).to(self.config.DEVICE, non_blocking=True)
            
            # Generate attributions with mixed precision for faster computation
            if self.config.DEVICE == "cuda":
                with torch.amp.autocast(self.config.DEVICE):
                    heatmaps = method.compute(model, batch_images, batch_labels)
            else:
                heatmaps = method.compute(model, batch_images, batch_labels)
            
            # Save sorted pixel indices
            if heatmaps is not None:
                for j, heatmap in enumerate(heatmaps):
                    img_id = image_ids[i + j]
                    heatmap_np = heatmap.cpu().numpy()
                    
                    # Sort pixels by importance and save
                    sorted_indices = sort_pixels(heatmap_np)
                    sorted_path = self.file_manager.get_heatmap_path(
                        dataset_name, model_name, method_name, img_id, sorted=True
                    )
                    np.save(sorted_path, sorted_indices)
        
        # Cleanup: check temperature and clear cache if needed
        self.gpu_manager.check_and_throttle()
        clear_cache_if_needed(threshold_percent=50.0)

