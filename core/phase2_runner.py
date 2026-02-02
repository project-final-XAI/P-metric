"""
Phase 2: Pre-Generate All Occluded Images.

Loads sorted heatmaps from Phase 1 and generates ALL occluded images
for all combinations (model × method × strategy × level × image).
This pre-generation makes Phase 3 super-fast (only loading and testing).
"""

import numpy as np
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Tuple, List
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
# from core.gpu_utils import clear_cache_if_needed
from core.phase1_runner import Phase1Runner
from data.loader import get_dataloader
from evaluation.occlusion import apply_occlusion_batch


class Phase2Runner:
    """Handles Phase 2: Pre-generation of all occluded images."""
    
    def __init__(
        self,
        config,
        gpu_manager: GPUManager,
        file_manager: FileManager,
        model_cache: Dict[str, Any]
    ):
        """
        Initialize Phase 2 runner.
        
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
        Pre-generate all occluded images for all combinations.
        
        Args:
            get_cached_model_func: Function to get cached model by name
        """
        dataset_name = self.config.DATASET_NAME
        
        if dataset_name not in self.config.DATASET_CONFIG:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_CONFIG. "
                f"Available datasets: {list(self.config.DATASET_CONFIG.keys())}"
            )
        
        logging.info(f"Starting Phase 2 - Dataset: {dataset_name}")
        
        # Check and generate missing heatmaps from Phase 1
        self._ensure_phase1_complete(dataset_name, get_cached_model_func)
        
        # Load dataset images
        image_label_map = self._load_dataset_images(dataset_name)
        
        # Generate all occluded images
        total_combinations = (
            len(self.config.GENERATING_MODELS) *
            len(self.config.ATTRIBUTION_METHODS) *
            len(self.config.FILL_STRATEGIES) *
            len(self.config.OCCLUSION_LEVELS)
        )
        
        with tqdm(total=total_combinations, desc="Phase 2 Progress") as pbar:
            for model_name in self.config.GENERATING_MODELS:
                for method_name in self.config.ATTRIBUTION_METHODS:
                    for strategy in self.config.FILL_STRATEGIES:
                        for level in self.config.OCCLUSION_LEVELS:
                            pbar.set_description(
                                f"{model_name[:12]}/{method_name[:12]}/{strategy}/{level}%"
                            )
                            try:
                                self._generate_occluded_images(
                                    dataset_name, model_name, method_name,
                                    strategy, level, image_label_map
                                )
                            except Exception as e:
                                logging.error(
                                    f"Error: {model_name}-{method_name}-{strategy}-{level}%: {e}"
                                )
                            finally:
                                pbar.update(1)
        
        logging.info(f"Phase 2 complete! Occluded images saved to: {self.file_manager.get_occluded_dir(dataset_name)}")
    
    def _ensure_phase1_complete(self, dataset_name: str, get_cached_model_func):
        """Check if Phase 1 is complete, run it for missing items if needed."""
        missing_items = []
        
        for model_name in self.config.GENERATING_MODELS:
            for method_name in self.config.ATTRIBUTION_METHODS:
                # Check if sorted heatmaps exist for all images
                sorted_heatmaps = self.file_manager.scan_sorted_heatmaps(
                    dataset_name, model_name, method_name
                )
                
                # Count images - use a simple approach
                batch_size = getattr(self.config, 'PHASE2_BATCH_SIZE', 128)
                dataloader = get_dataloader(dataset_name, batch_size=batch_size, shuffle=False)
                total_images = len(dataloader.dataset)
                
                if len(sorted_heatmaps) < total_images:
                    missing_items.append((model_name, method_name))
        
        if missing_items:
            logging.info(f"Running Phase 1 for {len(missing_items)} missing combinations...")
            phase1 = Phase1Runner(self.config, self.gpu_manager, self.file_manager, self.model_cache)
            phase1.run(get_cached_model_func)
    
    def _load_dataset_images(self, dataset_name: str) -> Dict[str, Tuple[torch.Tensor, int]]:
        """Load all dataset images into memory."""
        # Use larger batch size for faster loading
        batch_size = getattr(self.config, 'PHASE2_BATCH_SIZE', 128)
        dataloader = get_dataloader(dataset_name, batch_size=batch_size, shuffle=False)
        image_label_map = {}
        global_idx = 0
        
        for batch_images, batch_labels in dataloader:
            for img, lbl in zip(batch_images, batch_labels):
                image_label_map[f"image_{global_idx:05d}"] = (img, lbl.item())
                global_idx += 1
        
        return image_label_map
    
    def _generate_occluded_images(
        self,
        dataset_name: str,
        model_name: str,
        method_name: str,
        strategy: str,
        level: int,
        image_label_map: Dict[str, Tuple[torch.Tensor, int]]
    ):
        """
        Generate occluded images for a specific combination.
        
        Args:
            dataset_name: Dataset name
            model_name: Generating model name
            method_name: Attribution method name
            strategy: Fill strategy
            level: Occlusion level (0-100)
            image_label_map: Dictionary mapping image IDs to (image, label) tuples
        """
        # Collect images and sorted indices that need processing
        images_to_process = []
        sorted_indices_list = []
        image_ids = []
        
        # Pre-check which images need processing (faster than checking during loop)
        for img_id, (img, label) in image_label_map.items():
            occluded_path = self.file_manager.get_occluded_image_path(
                dataset_name, model_name, strategy, method_name, level, img_id
            )
            
            # Skip if already generated
            if occluded_path.exists():
                continue
            
            # Load sorted indices - try with category name first (for ImageNet), then fallback to old format
            sorted_path = None
            if dataset_name == "imagenet":
                # Try to get category name from label
                try:
                    from data.imagenet_class_mapping import get_cached_mapping, format_class_for_llm
                    from config import DATASET_CONFIG
                    import os
                    mapping = get_cached_mapping()
                    dataset_path = DATASET_CONFIG.get("imagenet", {}).get("path")
                    if dataset_path and os.path.exists(dataset_path):
                        synset_ids = sorted([d for d in os.listdir(dataset_path) 
                                            if os.path.isdir(os.path.join(dataset_path, d))])
                        if label < len(synset_ids):
                            synset_id = synset_ids[label]
                            category_name_full = mapping.get(synset_id, "")
                            if category_name_full:
                                category_name = format_class_for_llm(category_name_full)
                                sorted_path = self.file_manager.get_sorted_heatmap_path(
                                    dataset_name, model_name, method_name, img_id, category_name
                                )
                except Exception:
                    pass
            
            # Fallback to old format if not found
            if sorted_path is None or not sorted_path.exists():
                sorted_path = self.file_manager.get_sorted_heatmap_path(
                    dataset_name, model_name, method_name, img_id
                )
            
            if not sorted_path.exists():
                logging.warning(f"Missing sorted heatmap: {sorted_path}")
                continue
            
            sorted_indices = np.load(sorted_path)
            # img is already (C, H, W) from dataloader, no need to remove batch dimension
            images_to_process.append(img)
            sorted_indices_list.append(sorted_indices)
            image_ids.append(img_id)
        
        if not images_to_process:
            return
        
        # Use configurable batch size (default 128 for better GPU utilization)
        batch_size = getattr(self.config, 'PHASE2_BATCH_SIZE', 128)
        save_workers = getattr(self.config, 'PHASE2_SAVE_WORKERS', 4)
        
        # Process in batches with parallel saving
        for i in range(0, len(images_to_process), batch_size):
            end_idx = min(i + batch_size, len(images_to_process))
            batch_images = images_to_process[i:end_idx]
            batch_indices = sorted_indices_list[i:end_idx]
            batch_ids = image_ids[i:end_idx]
            
            # Apply occlusion (GPU processing)
            occluded_images = apply_occlusion_batch(
                batch_images,
                batch_indices,
                level,
                strategy,
                image_shape=(224, 224)
            )
            
            # Prepare save tasks for parallel execution
            save_tasks = []
            for j, occluded_img in enumerate(occluded_images):
                img_id = batch_ids[j]
                occluded_path = self.file_manager.get_occluded_image_path(
                    dataset_name, model_name, strategy, method_name, level, img_id
                )
                self.file_manager.ensure_dir_exists(occluded_path.parent)
                save_tasks.append((occluded_img, occluded_path))
            
            # Save images in parallel using ThreadPoolExecutor (I/O bound operation)
            if save_workers > 1 and len(save_tasks) > 1:
                with ThreadPoolExecutor(max_workers=save_workers) as executor:
                    futures = [executor.submit(self._save_occluded_image, img, path) 
                               for img, path in save_tasks]
                    # Wait for all saves to complete
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logging.error(f"Error saving image: {e}")
            else:
                # Sequential saving if workers <= 1 or single image
                for img, path in save_tasks:
                    self._save_occluded_image(img, path)
        
        # Cleanup GPU memory periodically
        # if len(images_to_process) > 0:
        #     clear_cache_if_needed(threshold_percent=50.0)
    
    def _save_occluded_image(self, image_tensor: torch.Tensor, path: Path):
        """
        Save occluded image tensor as PNG.
        
        Args:
            image_tensor: Image tensor (C, H, W) - normalized
            path: Path to save PNG file
        """
        # Ensure tensor is on CPU and has correct shape
        img_tensor = image_tensor.detach().cpu().clone()
        
        # Handle different tensor shapes
        if img_tensor.ndim == 4:  # (B, C, H, W) - take first image
            img_tensor = img_tensor[0]
        elif img_tensor.ndim == 2:  # (H, W) - add channel dimension
            img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # Denormalize ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Convert to numpy and PIL Image
        img_array = (img_tensor.numpy() * 255).astype(np.uint8)
        
        # Ensure correct shape: (C, H, W) -> (H, W, C)
        if img_array.ndim == 3:
            if img_array.shape[0] == 3:
                img_array = np.transpose(img_array, (1, 2, 0))
            elif img_array.shape[2] == 3:
                # Already (H, W, C)
                pass
            else:
                # Unexpected shape - try to fix
                logging.warning(f"Unexpected image shape: {img_array.shape}")
        elif img_array.ndim == 2:
            # Grayscale - convert to RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        
        Image.fromarray(img_array).save(path, 'PNG')


def main():
    """Simple main function to run Phase 2."""
    import sys
    import logging
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config
    from core.gpu_manager import GPUManager
    from core.file_manager import FileManager
    from models.loader import load_model
    from evaluation.judging.binary_llm_judge import BinaryLLMJudge
    from evaluation.judging.cosine_llm_judge import CosineSimilarityLLMJudge
    from evaluation.judging.classid_llm_judge import ClassIdLLMJudge
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    gpu_manager = GPUManager()
    file_manager = FileManager(config.BASE_DIR)
    model_cache = {}
    
    def get_cached_model(name):
        if name not in model_cache:
            if name.endswith('-binary'):
                model_cache[name] = BinaryLLMJudge(name, config.DATASET_NAME, 0.0)
            elif name.endswith('-cosine'):
                model_cache[name] = CosineSimilarityLLMJudge(name, config.DATASET_NAME, 0.1, 0.8, "nomic-embed-text")
            elif name.endswith('-classid'):
                model_cache[name] = ClassIdLLMJudge(name, config.DATASET_NAME, 0.0)
            else:
                model_cache[name] = load_model(name)
        return model_cache[name]
    
    runner = Phase2Runner(config, gpu_manager, file_manager, model_cache)
    runner.run(get_cached_model)


if __name__ == "__main__":
    main()
