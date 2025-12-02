"""
Base class for LLM judges with shared functionality.

Provides common methods for all LLM-based judges to avoid code duplication.
"""

import torch
import tempfile
import os
import logging
import numpy as np
from typing import List, Dict
from abc import abstractmethod
from PIL import Image
import ollama  # Import once at module level for better performance

from config import DATASET_CONFIG
from evaluation.judging.base import JudgingModel
from data.imagenet_class_mapping import get_cached_mapping, format_class_for_llm

# Disable httpx/HTTP logging completely
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.connection").setLevel(logging.CRITICAL)


# Maximum parallel workers for all LLM judges
MAX_PARALLEL_WORKERS = 24


class BaseLLMJudge(JudgingModel):
    """
    Base class for LLM judges providing shared functionality.
    
    Handles:
    - Loading class names from dataset
    - Loading ImageNet class mapping (synset ID -> readable name)
    - Converting tensors to temporary image files
    - Formatting class names for LLM prompts
    """
    
    def __init__(self, model_name: str, dataset_name: str = "imagenet"):
        """
        Initialize base LLM judge.
        
        Args:
            model_name: Ollama model name (may include -binary/-cosine suffix)
            dataset_name: Dataset name
        """
        super().__init__(model_name)
        self.dataset_name = dataset_name
        
        # Extract actual Ollama model name (remove -binary/-cosine suffix)
        # e.g., "llama3.2-vision-binary" -> "llama3.2-vision"
        if model_name.endswith('-binary'):
            self.ollama_model_name = model_name[:-7]  # Remove '-binary'
        elif model_name.endswith('-cosine'):
            self.ollama_model_name = model_name[:-7]  # Remove '-cosine'
        else:
            self.ollama_model_name = model_name
        
        # Load class names
        self.class_names = self._load_class_names()
        if not self.class_names:
            raise ValueError(f"Could not load class names for dataset: {dataset_name}")
        
        # Load ImageNet class mapping if needed
        self.class_name_mapping = self._load_class_mapping()
    
    def _load_class_names(self) -> List[str]:
        """
        Load class names for the dataset.
        
        Returns:
            List of class names (synset IDs for ImageNet, folder names for others)
        """
        if self.dataset_name == "imagenet":
            dataset_path = DATASET_CONFIG.get("imagenet", {}).get("path")
            if dataset_path and os.path.exists(dataset_path):
                class_names = sorted([d for d in os.listdir(dataset_path) 
                                    if os.path.isdir(os.path.join(dataset_path, d))])
                if len(class_names) == 1000:
                    return class_names
            
            # Fallback: try ImageFolder
            try:
                from torchvision.datasets import ImageFolder
                if dataset_path and os.path.exists(dataset_path):
                    temp_dataset = ImageFolder(root=str(dataset_path))
                    return temp_dataset.classes
            except:
                pass
            
            logging.warning("Could not load ImageNet class names from dataset.")
            return []
        
        elif self.dataset_name in ["SIPaKMeD", "SIPaKMeD_cropped"]:
            dataset_path = DATASET_CONFIG.get(self.dataset_name, {}).get("path")
            if dataset_path and os.path.exists(dataset_path):
                class_names = sorted([d for d in os.listdir(dataset_path) 
                                    if os.path.isdir(os.path.join(dataset_path, d))])
                return class_names
        
        return []
    
    def _load_class_mapping(self) -> Dict[str, str]:
        """
        Load ImageNet class mapping (synset ID -> readable name).
        
        Returns:
            Dictionary mapping synset IDs to readable names, or empty dict
        """
        if self.dataset_name == "imagenet":
            try:
                mapping = get_cached_mapping()
                logging.info(f"Loaded ImageNet class mapping with {len(mapping)} entries")
                return mapping
            except Exception as e:
                logging.warning(f"Could not load ImageNet mapping: {e}")
                return {}
        return {}
    
    def _format_class_name(self, class_name: str) -> str:
        """
        Format class name for natural language prompt.
        
        Converts synset IDs to readable names for ImageNet.
        
        Args:
            class_name: Raw class name (e.g., 'n01440764' or 'Dyskeratotic')
            
        Returns:
            Human-readable name (e.g., 'tench' or 'Dyskeratotic')
        """
        # Try to get readable name from mapping (for ImageNet synsets)
        if class_name in self.class_name_mapping:
            readable_name = self.class_name_mapping[class_name]
            # Format for LLM: take first part if there's a comma
            return format_class_for_llm(readable_name)
        
        # Replace underscores with spaces for other datasets
        return class_name.replace('_', ' ')
    
    @abstractmethod
    def predict(self, images: List[torch.Tensor], **kwargs):
        """
        Predict classes for given images.
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def _predict_single_image_from_path(
        self,
        image_path: str,
        true_label: int,
        img_index: int
    ):
        """
        Predict class for a single image using file path directly (optimized).
        
        Must be implemented by subclasses. This avoids unnecessary tensor conversions.
        
        Args:
            image_path: Path to image file (PNG/JPG)
            true_label: True class label
            img_index: Original index in batch
            
        Returns:
            Tuple of (img_index, predicted_class_index) or (img_index, predicted_class_index, similarity)
        """
        pass
    
    def _call_ollama_with_retry(
        self,
        prompt: str,
        image_path: str,
        max_retries: int = 3,
        **ollama_options
    ) -> str:
        """
        Call Ollama API with retry logic (shared helper method).
        
        Optimized with:
        - Smart exponential backoff (longer delays for connection errors)
        - Timeout handling
        - Better error classification
        - No import overhead (ollama imported at module level)
        
        Args:
            prompt: Prompt text for LLM
            image_path: Path to image file
            max_retries: Maximum number of retry attempts
            **ollama_options: Additional options for Ollama (temperature, etc.)
            
        Returns:
            Response text from LLM
            
        Raises:
            Exception: If all retries fail
        """
        import time
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Direct call - ollama already imported at module level
                response = ollama.chat(
                    model=self.ollama_model_name,
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [image_path]
                        }
                    ],
                    options=ollama_options
                )
                return response.message.content.strip()
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Classify error type for smarter backoff
                if attempt < max_retries - 1:
                    # Connection errors need longer backoff
                    if 'connection' in error_str or 'timeout' in error_str or 'network' in error_str:
                        backoff_time = 0.5 * (2 ** attempt)  # Exponential: 0.5s, 1s, 2s
                    # Rate limiting needs longer backoff
                    elif 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                        backoff_time = 1.0 * (2 ** attempt)  # Exponential: 1s, 2s, 4s
                    # Other errors use shorter backoff
                    else:
                        backoff_time = 0.1 * (attempt + 1)  # Linear: 0.1s, 0.2s, 0.3s
                    
                    logging.warning(f"Retry {attempt + 1}/{max_retries} for Ollama call (backoff: {backoff_time:.1f}s): {e}")
                    time.sleep(backoff_time)
                else:
                    # Last attempt failed - raise the exception
                    raise last_exception
        
        # Should not reach here, but just in case
        raise last_exception if last_exception else Exception("Unknown error in Ollama call")
    
    def predict_from_paths(self, image_paths: List[str], true_labels: List[int] = None, shared_executor=None, **kwargs) -> np.ndarray:
        """
        Predict classes for images given as file paths (optimized for LLM judges).
        
        This method avoids unnecessary tensor conversions when images are already on disk.
        Uses parallel processing with adaptive worker count for better performance.
        
        Optimizations:
        - Adaptive worker count based on batch size
        - Better error handling and recovery
        - Progress tracking for large batches
        
        Args:
            image_paths: List of image file paths (PNG/JPG)
            true_labels: List of true labels (optional)
            **kwargs: Additional parameters
            
        Returns:
            Array of predicted class indices (shape: [batch_size])
        """
        if len(image_paths) == 0:
            return np.array([], dtype=np.int64)

        # Get true labels if provided
        if true_labels is None:
            true_labels = [0] * len(image_paths)  # Fallback
        elif len(true_labels) != len(image_paths):
            logging.warning(
                f"true_labels length ({len(true_labels)}) doesn't match image_paths length ({len(image_paths)}). "
                f"Using fallback."
            )
            true_labels = [0] * len(image_paths)

        # Use shared executor if provided, otherwise create new one
        # Shared executor eliminates overhead between batches - CRITICAL for performance!
        batch_size = len(image_paths)
        max_workers = min(MAX_PARALLEL_WORKERS, batch_size)
        
        predictions = [None] * len(image_paths)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Use shared executor if provided (zero overhead), otherwise create new one
        should_close_executor = False
        if shared_executor is None:
            executor = ThreadPoolExecutor(max_workers=max_workers)
            should_close_executor = True
        else:
            executor = shared_executor
        
        try:
            # Submit all tasks immediately
            future_to_idx = {
                executor.submit(self._predict_single_image_from_path, path, true_labels[idx], idx): idx
                for idx, path in enumerate(image_paths)
            }

            # Process results as they complete (out-of-order is fine, we track by idx)
            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    # Handle both (img_idx, class_idx) and (img_idx, class_idx, similarity) formats
                    img_idx = result[0]
                    predictions[img_idx] = result[1]
                except Exception as e:
                    idx = future_to_idx[future]
                    logging.error(f"Unexpected error processing image {idx}: {e}")
                    predictions[idx] = -1
        finally:
            # Only close executor if we created it
            if should_close_executor:
                executor.shutdown(wait=False)

        return np.array(predictions, dtype=np.int64)

