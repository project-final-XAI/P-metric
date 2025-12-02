"""
Phase 1: Heatmap Generation Runner.

Generates attribution heatmaps for all model-method-image combinations.
This phase creates sorted pixel indices files that are used in Phase 2 for occlusion evaluation.
"""

import numpy as np
import torch
import logging
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.gpu_utils import clear_cache_if_needed, prepare_batch_tensor
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
            image_label_map = {}
            global_idx = 0
            for batch_images, batch_labels in dataloader:
                # Iterate through each image and label in the batch
                for img, lbl in zip(batch_images, batch_labels):
                    image_label_map[f"image_{global_idx:05d}"] = (img, lbl.item())
                    global_idx += 1
            
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
            sorted_path = self.file_manager.get_sorted_heatmap_path(
                dataset_name, model_name, method_name, img_id
            )
            regular_path = self.file_manager.get_regular_heatmap_path(
                dataset_name, model_name, method_name, img_id
            )
            
            # Only process if either file doesn't exist
            if not sorted_path.exists() or not regular_path.exists():
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
            batch_images = prepare_batch_tensor(
                images_to_process[i:end_idx],
                device=self.config.DEVICE,
                memory_format=torch.channels_last
            )
            batch_labels = torch.tensor(labels[i:end_idx]).to(self.config.DEVICE, non_blocking=True)
            
            # Generate attributions with mixed precision for faster computation
            if self.config.DEVICE == "cuda":
                with torch.amp.autocast(self.config.DEVICE):
                    heatmaps = method.compute(model, batch_images, batch_labels)
            else:
                heatmaps = method.compute(model, batch_images, batch_labels)
            
            # Save sorted pixel indices and regular PNG
            if heatmaps is not None:
                for j, heatmap in enumerate(heatmaps):
                    img_id = image_ids[i + j]
                    heatmap_np = heatmap.cpu().numpy()
                    
                    # Handle multi-channel heatmaps (take mean if needed)
                    if heatmap_np.ndim == 3:
                        heatmap_np = np.mean(heatmap_np, axis=0)
                    
                    # Sort pixels by importance and save NPY
                    sorted_indices = sort_pixels(heatmap_np)
                    sorted_path = self.file_manager.get_sorted_heatmap_path(
                        dataset_name, model_name, method_name, img_id
                    )
                    self.file_manager.ensure_dir_exists(sorted_path.parent)
                    np.save(sorted_path, sorted_indices)
                    
                    # Save regular PNG heatmap
                    regular_path = self.file_manager.get_regular_heatmap_path(
                        dataset_name, model_name, method_name, img_id
                    )
                    self.file_manager.ensure_dir_exists(regular_path.parent)
                    self._save_heatmap_png(heatmap_np, regular_path)
        
        # Cleanup: check temperature and clear cache if needed
        self.gpu_manager.check_and_throttle()
        clear_cache_if_needed(threshold_percent=50.0)
    
    def _save_heatmap_png(self, heatmap: np.ndarray, path: Path):
        """
        Save heatmap as PNG image with colormap.
        
        Args:
            heatmap: 2D numpy array representing heatmap
            path: Path to save PNG file
        """
        # Normalize to 0-1
        hmap = heatmap.copy()
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        
        # Convert to uint8
        hmap_uint8 = (hmap * 255).astype(np.uint8)
        
        # Map colormap names to OpenCV constants
        colormap_dict = {
            "hot": cv2.COLORMAP_HOT,
            "jet": cv2.COLORMAP_JET,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "turbo": cv2.COLORMAP_TURBO,
        }
        
        colormap = getattr(self.config, 'HEATMAP_COLORMAP', 'hot')
        cv_colormap = colormap_dict.get(colormap.lower(), cv2.COLORMAP_HOT)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(hmap_uint8, cv_colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Save as PNG
        Image.fromarray(heatmap_rgb).save(path, 'PNG')


def main():
    """Simple main function to run Phase 1."""
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
            else:
                model_cache[name] = load_model(name)
        return model_cache[name]
    
    runner = Phase1Runner(config, gpu_manager, file_manager, model_cache)
    runner.run(get_cached_model)


if __name__ == "__main__":
    main()

