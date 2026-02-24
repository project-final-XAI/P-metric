"""
Base class for LLM judges with shared functionality.

Provides common methods for all LLM-based judges to avoid code duplication.
"""

import logging
import numpy as np
import os
import base64
import io
from pathlib import Path
from typing import List, Dict, Union, Any, Optional
from abc import abstractmethod
import ollama  # Import once at module level for better performance

from config import DATASET_CONFIG
from evaluation.judging.base import JudgingModel
from data.imagenet_class_mapping import get_cached_mapping, format_class_for_llm

# Disable httpx/HTTP logging completely
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.connection").setLevel(logging.CRITICAL)


# Maximum parallel workers for all LLM judges (aligned with OLLAMA_NUM_PARALLEL=16)
MAX_PARALLEL_WORKERS = 16

# Ollama optimization constants
# IMPORTANT: keep_alive=0 gives best accuracy but is ~3x slower
# Alternative: Use periodic resets (see OLLAMA_RESET_INTERVAL)
OLLAMA_KEEP_ALIVE = None  # Force model unload (None = immediate unload) for max accuracy (prevents KV cache degradation)
OLLAMA_NUM_CTX = 8192  # Context window size for llama3.2-vision
OLLAMA_SEED = 42  # Fixed seed for deterministic results (critical for reproducibility!)

# Periodic reset configuration (alternative to keep_alive=0)
# Set OLLAMA_KEEP_ALIVE to "5m" and use reset every N requests for speed/accuracy balance
OLLAMA_RESET_INTERVAL = 50  # Reset model state every N requests (0 = disabled) 

# Default system prompt for all LLM judges
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise vision classification assistant. "
    "Answer concisely based only on the provided image and follow any "
    "response schema when requested."
)


class BaseLLMJudge(JudgingModel):
    """
    Base class for LLM judges providing shared functionality.
    
    Handles:
    - Loading class names from dataset
    - Loading ImageNet class mapping (synset ID -> readable name)
    - Formatting class names for LLM prompts
    - Optimized image conversion (hybrid I/O: paths vs base64)
    - Efficient Ollama API calls with keep_alive and num_ctx
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str = "imagenet",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize base LLM judge.
        
        Args:
            model_name: Ollama model name (may include -binary/-cosine suffix)
            dataset_name: Dataset name
        """
        super().__init__(model_name)
        self.dataset_name = dataset_name
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
                
        # Request counter for periodic reset (prevents KV cache degradation)
        self._request_count = 0
        
        # Extract actual Ollama model name (remove -binary/-cosine/-classid suffix)
        # e.g., "llama3.2-vision-binary" -> "llama3.2-vision"
        if model_name.endswith('-binary'):
            self.ollama_model_name = model_name[:-7]  # Remove '-binary'
        elif model_name.endswith('-cosine'):
            self.ollama_model_name = model_name[:-7]  # Remove '-cosine'
        elif model_name.endswith('-classid'):
            self.ollama_model_name = model_name[:-8]  # Remove '-classid'
        else:
            self.ollama_model_name = model_name
        
        # Load class names
        self.class_names = self._load_class_names()
        if not self.class_names:
            raise ValueError(f"Could not load class names for dataset: {dataset_name}")
        
        # Load ImageNet class mapping if needed
        self.class_name_mapping = self._load_class_mapping()
        
        # Log keep_alive configuration for verification
        logging.info(f"BaseLLMJudge initialized: keep_alive={OLLAMA_KEEP_ALIVE}, seed={OLLAMA_SEED}")
    
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
            except Exception:
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
        Ensures simple English labels (strips Latin names after comma).
        
        Args:
            class_name: Raw class name (e.g., 'n01440764' or 'Dyskeratotic')
            
        Returns:
            Human-readable name (e.g., 'tench' or 'Dyskeratotic')
        """
        # Try to get readable name from mapping (for ImageNet synsets)
        if class_name in self.class_name_mapping:
            readable_name = self.class_name_mapping[class_name]
            # Format for LLM: take first part if there's a comma (strips Latin names)
            return format_class_for_llm(readable_name)
        
        # Replace underscores with spaces for other datasets
        return class_name.replace('_', ' ')
    
    def _reset_ollama_model(self):
        """
        Reset Ollama model to clear KV cache and prevent accuracy degradation.
        
        This is called periodically when OLLAMA_RESET_INTERVAL > 0.
        Addresses known issue: ollama/ollama#4846
        """
        import subprocess
        import time
        try:
            subprocess.run(
                ["ollama", "stop", self.ollama_model_name],
                capture_output=True,
                timeout=10
            )
            time.sleep(1)  # Brief pause for clean unload
            logging.debug(f"Reset Ollama model {self.ollama_model_name} to clear KV cache")
        except Exception as e:
            logging.warning(f"Failed to reset Ollama model: {e}")
    
    @abstractmethod
    def _predict_single_image(
        self,
        image_data: Union[str, bytes],
        true_label: int,
        image_id: str,
        context: Optional[Dict] = None
    ):
        """
        Predict class for a single image using image data (path or base64 string).
        
        Must be implemented by subclasses. This avoids unnecessary disk I/O.
        
        Args:
            image_data: Image file path (str) or base64-encoded image string (str)
            true_label: True class label
            image_id: Stable identifier for the image (e.g., file stem)
            context: Optional metadata (occlusion level, fill strategy, etc.)
            
        Returns:
            Tuple of (img_index, predicted_class_index) or (img_index, predicted_class_index, similarity)
        """
        pass
    
    def _call_ollama_with_retry(
        self,
        prompt: str,
        image_data: Union[str, bytes],
        max_retries: int = 3,
        temperature: float = 0.0,
        format_schema: Optional[Dict] = None,
        classes_names: Optional[List[str]] = None,
        **additional_options
    ) -> str:
        """
        Call Ollama API with retry logic (shared helper method).
        
        Optimized with:
        - keep_alive to prevent model reloading (CRITICAL for throughput)
        - num_ctx for optimal context window
        - Smart exponential backoff (longer delays for connection errors)
        - Timeout handling
        - Better error classification
        - No import overhead (ollama imported at module level)
        - In-memory image processing (no disk I/O)
        
        Args:
            prompt: Prompt text for LLM
            image_data: Image file path (str) or base64-encoded image string (str)
            max_retries: Maximum number of retry attempts
            temperature: Temperature for LLM (0.0 = deterministic)
            format_schema: Optional Pydantic JSON schema for structured outputs
            **additional_options: Additional options for Ollama
        
        Returns:
            Response text from LLM
            
        Raises:
            Exception: If all retries fail
        """
        import time
        
        # Periodic reset to prevent KV cache degradation (if enabled)
        if OLLAMA_RESET_INTERVAL > 0 and OLLAMA_KEEP_ALIVE is not None and OLLAMA_KEEP_ALIVE != 0:
            self._request_count += 1
            if self._request_count >= OLLAMA_RESET_INTERVAL:
                self._reset_ollama_model()
                self._request_count = 0
        
        # Keep file paths as-is for Ollama (let Ollama handle loading)
        # Only convert non-path data (PIL/Tensor/numpy) to base64
        if isinstance(image_data, str):
            if not os.path.exists(image_data):
                # Assume it's already base64 - pass as-is
                pass
            # else: it's a valid file path - Ollama will load it directly
        else:
            # Convert in-memory data to base64
            image_data = self._convert_image_to_base64(image_data)
        
        # Build optimized Ollama options
        # IMPORTANT: seed is critical for deterministic results!
        ollama_options = {
            'temperature': temperature,
            'seed': OLLAMA_SEED,  # Fixed seed for reproducibility
            'num_ctx': OLLAMA_NUM_CTX,
            **additional_options  # Allow override of defaults
        }
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Build messages list
                messages = []
                
                # Optional system grounding with class names (indexed list)
                if classes_names:
                    class_list = "\n".join(
                        f"{i}: {name}" for i, name in enumerate(classes_names[:1000])
                    )
                    system_prompt = (
                        "You are given the following classes indexed from 0 upward:\n"
                        f"{class_list}\n"
                        "Use these class indices when responding."
                    )
                    messages.append({'role': 'system', 'content': system_prompt})
                
                # Append system message
                messages.append({
                    'role': 'system',
                    'content': self.system_prompt
                })                
                # Append user message with image
                messages.append({
                    'role': 'user',
                    'content': prompt,
                    'images': [image_data]
                })
                response = ollama.chat(
                    model=self.ollama_model_name,
                    messages=messages,
                    options=ollama_options,
                    keep_alive=OLLAMA_KEEP_ALIVE,  # Keep model in memory
                    format=format_schema  # Optional JSON schema for structured outputs
                )
                logging.info(f"Sending response: {response}")
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
    
    def _convert_image_to_base64(self, img: Any) -> Union[str, bytes]:
        """
        Convert an image (Tensor/Numpy/PIL/Path) to base64-encoded string or keep as path.
        
        Hybrid I/O optimization:
        - If input is a path -> keep as path (Ollama handles paths efficiently)
        - If input is Tensor/PIL/NumPy -> convert to base64 in RAM (no disk writes)
        
        Optimized for in-memory processing - no disk I/O.
        
        Args:
            img: Input image - can be:
                 - File path (str or Path) - will be kept as path for Ollama
                 - PIL Image
                 - NumPy array
                 - PyTorch Tensor
        
        Returns:
            Base64-encoded image string (for tensors/arrays/PIL) or file path (for paths)
        """
        from pathlib import Path
        
        # If it's already a path, just return it as string - Ollama handles paths efficiently
        if isinstance(img, (str, Path)):
            return str(img)
        
        # Convert to PIL Image (in-memory, no disk I/O)
        try:
            from PIL import Image
            
            if hasattr(img, 'save') and hasattr(img, 'size'):
                # Already a PIL Image
                pil_img = img
            elif isinstance(img, np.ndarray):
                # NumPy array - optimize conversion
                if img.dtype != np.uint8:
                    # Normalize if needed (assuming 0-1 range)
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = np.clip(img, 0, 255).astype(np.uint8)
                
                # Handle shape conversion
                if img.ndim == 2:
                    # Grayscale -> RGB
                    pil_img = Image.fromarray(img, mode='L').convert('RGB')
                elif img.ndim == 3:
                    if img.shape[2] == 1:
                        # (H, W, 1) -> RGB
                        pil_img = Image.fromarray(img[:, :, 0], mode='L').convert('RGB')
                    elif img.shape[2] == 3:
                        # (H, W, 3) -> RGB
                        pil_img = Image.fromarray(img, mode='RGB')
                    else:
                        raise ValueError(f"Unsupported array shape: {img.shape}")
                else:
                    raise ValueError(f"Unsupported array dimensions: {img.ndim}")
            else:
                # Assume it's a tensor-like object
                import torch
                if isinstance(img, torch.Tensor):
                    # Convert tensor to numpy then PIL (optimized)
                    # Move to CPU if needed (non-blocking for better performance)
                    if img.is_cuda:
                        img_np = img.detach().cpu().numpy()
                    else:
                        img_np = img.detach().numpy()
                    
                    # Handle different tensor shapes
                    if img_np.ndim == 2:
                        # (H, W) -> RGB
                        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_np, mode='L').convert('RGB')
                    elif img_np.ndim == 3:
                        # (C, H, W) -> (H, W, C)
                        if img_np.shape[0] == 3 or img_np.shape[0] == 1:
                            img_np = img_np.transpose(1, 2, 0)
                            if img_np.shape[2] == 1:
                                img_np = img_np[:, :, 0]
                                pil_img = Image.fromarray(img_np, mode='L').convert('RGB')
                            else:
                                # Normalize if needed (assuming 0-1 range for normalized tensors)
                                if img_np.max() <= 1.0:
                                    img_np = (img_np * 255).astype(np.uint8)
                                else:
                                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                                pil_img = Image.fromarray(img_np, mode='RGB')
                        else:
                            # (H, W, C) already
                            if img_np.max() <= 1.0:
                                img_np = (img_np * 255).astype(np.uint8)
                            else:
                                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                            pil_img = Image.fromarray(img_np, mode='RGB')
                    elif img_np.ndim == 4:
                        # (B, C, H, W) -> take first image
                        img_np = img_np[0].transpose(1, 2, 0)
                        # Normalize if needed
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_np, mode='RGB')
                    else:
                        raise ValueError(f"Unsupported tensor shape: {img_np.shape}")
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
            
            # Save PIL Image to BytesIO buffer (in-memory, no disk I/O)
            buffer = io.BytesIO()
            # Use PNG format for lossless compression
            pil_img.save(buffer, format='PNG', optimize=False)  # optimize=False for speed
            buffer.seek(0)
            
            # Encode to base64 string
            img_bytes = buffer.getvalue()
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            
            # Clean up buffer
            buffer.close()
            
            return base64_str
            
        except Exception as e:
            logging.error(f"Failed to convert image to base64: {e}")
            raise
    
    def predict(
        self,
        images: Union[List, np.ndarray, Any],
        **kwargs
    ) -> np.ndarray:
        """
        Predict classes for given images (required by JudgingModel interface).
        
        Optimized for in-memory processing - no disk I/O. Converts images to base64
        strings in memory for maximum performance.
        
        Args:
            images: Input images - can be:
                   - List of image file paths (str or Path) - kept as paths
                   - List of PIL Images - converted to base64
                   - List of NumPy arrays - converted to base64
                   - List of PyTorch Tensors - converted to base64
            **kwargs: Additional parameters:
                    - true_labels: List of true labels (optional)
                    - shared_executor: ThreadPoolExecutor for parallel processing (optional)
        
        Returns:
            Array of predicted class indices (shape: [batch_size])
        """
        # Extract true_labels from kwargs if provided
        true_labels = kwargs.get('true_labels', None)
        shared_executor = kwargs.get('shared_executor', None)
        
        # Convert images to base64 strings or keep paths (in-memory, no disk I/O)
        image_data_list = []
        
        for img in images:
            try:
                image_data = self._convert_image_to_base64(img)
                image_data_list.append(image_data)
            except Exception as e:
                logging.error(f"Failed to convert image: {e}")
                # Create a dummy entry that will fail gracefully
                image_data_list.append("")
        
        # Use predict_from_data for actual prediction
        return self.predict_from_data(image_data_list, true_labels=true_labels, shared_executor=shared_executor, **kwargs)
    
    def predict_from_data(
        self,
        image_data_list: List[Union[str, bytes]],
        true_labels: List[int] = None,
        image_ids: List[str] = None,
        context: Dict = None,
        return_details: bool = False,
        shared_executor=None,
        **kwargs
    ) -> np.ndarray:
        """
        Predict classes for images given as file paths or base64 strings (optimized for LLM judges).
        
        This method uses in-memory processing - no disk I/O. Accepts both file paths
        and base64-encoded image strings. Uses parallel processing with adaptive worker count.
        
        Optimizations:
        - In-memory processing (no disk I/O)
        - Adaptive worker count based on batch size
        - Better error handling and recovery
        - Progress tracking for large batches
        
        Args:
            image_data_list: List of image file paths (str) or base64-encoded image strings (str)
            true_labels: List of true labels (optional)
            **kwargs: Additional parameters
            
        Returns:
            Array of predicted class indices (shape: [batch_size])
        """
        if len(image_data_list) == 0:
            return np.array([], dtype=np.int64)

        # Get true labels if provided
        if true_labels is None:
            true_labels = [0] * len(image_data_list)  # Fallback
        elif len(true_labels) != len(image_data_list):
            logging.warning(
                f"true_labels length ({len(true_labels)}) doesn't match image_data_list length ({len(image_data_list)}). "
                f"Using fallback."
            )
            true_labels = [0] * len(image_data_list)

        # Stable image ids for logging and traceability
        if image_ids is None:
            image_ids = [str(i) for i in range(len(image_data_list))]
        elif len(image_ids) != len(image_data_list):
            logging.warning(
                f"image_ids length ({len(image_ids)}) doesn't match image_data_list length ({len(image_data_list)}). "
                f"Using fallback."
            )
            image_ids = [str(i) for i in range(len(image_data_list))]

        # Optional shared context per batch (occlusion level, fill strategy, etc.)
        context = context or {}

        # Use shared executor if provided, otherwise create new one
        # Shared executor eliminates overhead between batches - CRITICAL for performance!
        batch_size = len(image_data_list)
        
        # When keep_alive=None or 0, process sequentially to allow model unload between requests
        # Parallel processing prevents model unload because there's always another request using it
        if OLLAMA_KEEP_ALIVE is None or OLLAMA_KEEP_ALIVE == 0:
            max_workers = 1  # Sequential processing for keep_alive=None/0
        else:
            max_workers = min(MAX_PARALLEL_WORKERS, batch_size)
        
        predictions = [None] * len(image_data_list)
        details = {} if return_details else None

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
                executor.submit(
                    self._predict_single_image,
                    image_data,
                    true_labels[idx],
                    image_ids[idx],
                    context
                ): idx
                for idx, image_data in enumerate(image_data_list)
            }

            # Process results as they complete (out-of-order is fine, we track by idx)
            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    # Handle both (img_id, class_idx) and extended tuples with extra metadata
                    idx = future_to_idx[future]
                    predictions[idx] = result[1]
                    if return_details and len(result) > 2:
                        details[idx] = result[2:]
                except Exception as e:
                    idx = future_to_idx[future]
                    logging.error(f"Unexpected error processing image {idx}: {e}")
                    predictions[idx] = -1
        finally:
            # Only close executor if we created it
            if should_close_executor:
                executor.shutdown(wait=True)  # Wait for all tasks to complete

        if return_details:
            return np.array(predictions, dtype=np.int64), details
        return np.array(predictions, dtype=np.int64)
    
    # Backward compatibility alias
    def predict_from_paths(
        self,
        image_paths: List[str],
        true_labels: List[int] = None,
        image_ids: List[str] = None,
        context: Dict = None,
        return_details: bool = False,
        shared_executor=None,
        **kwargs
    ) -> np.ndarray:
        """
        Backward compatibility method - redirects to predict_from_data.
        
        Args:
            image_paths: List of image file paths
            true_labels: List of true labels (optional)
            image_ids: Optional list of stable ids (defaults to file stems)
            context: Optional metadata passed to per-image predictor
            **kwargs: Additional parameters
            
        Returns:
            Array of predicted class indices (shape: [batch_size])
        """
        if image_ids is None:
            image_ids = [Path(p).stem for p in image_paths]

        return self.predict_from_data(
            image_paths,
            true_labels=true_labels,
            image_ids=image_ids,
            context=context,
            return_details=return_details,
            shared_executor=shared_executor,
            **kwargs
        )
