"""
Binary LLM Judge - Yes/No approach.

Asks the model "Do you see {true_category}? Answer yes or no".
Simple and direct approach - only one question per image.
- "yes" = correct prediction
- "no" = incorrect prediction

Most efficient and straightforward evaluation method.
"""

import ollama
import torch
import numpy as np
from typing import List, Tuple
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation.judging.base_llm_judge import BaseLLMJudge, MAX_PARALLEL_WORKERS


class BinaryLLMJudge(BaseLLMJudge):
    """
    Binary LLM Judge using Yes/No questions.
    
    For each image, asks "Do you see {true_category}?" and evaluates:
    - Answer "yes" → Correct (returns true_label)
    - Answer "no" → Incorrect (returns different label)
    
    This is the most efficient approach - only one question per image.
    """
    
    def __init__(self, model_name: str, dataset_name: str = "imagenet", temperature: float = 0.0):
        """
        Initialize Binary LLM Judge.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.2-vision")
            dataset_name: Dataset name to get class names from
            temperature: Temperature for LLM (0.0 = deterministic)
        """
        super().__init__(model_name, dataset_name)
        self.temperature = temperature
        
        logging.info(f"BinaryLLMJudge initialized with {len(self.class_names)} classes, temperature={temperature}")
    
    
    def _clean_response(self, response_text: str) -> str:
        """
        Extract yes/no from LLM response.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            'yes' or 'no' (lowercase)
        """
        response_lower = response_text.lower().strip()
        
        # Direct yes/no
        if response_lower.startswith('yes'):
            return 'yes'
        if response_lower.startswith('no'):
            return 'no'
        
        # Common variations
        if 'yes' in response_lower[:10]:
            return 'yes'
        if 'no' in response_lower[:10]:
            return 'no'
        
        # Default to no if unclear
        return 'no'
    
    def _predict_single_image(
        self,
        img_tensor: torch.Tensor,
        true_label: int,
        img_index: int
    ) -> Tuple[int, int]:
        """
        Predict class for a single image using binary yes/no question.
        
        Simply asks if the LLM sees the true class in the image.
        - If "yes" → correct prediction (returns true_label)
        - If "no" → incorrect prediction (returns -1 to mark as wrong)
        
        Args:
            img_tensor: Image tensor (C, H, W)
            true_label: True class label
            img_index: Original index in batch
            
        Returns:
            Tuple of (img_index, predicted_class_index)
        """
        try:
            temp_image_path = self._tensor_to_temp_file(img_tensor)
            
            try:
                # Get the true class name
                class_name = self._format_class_name(self.class_names[true_label])
                
                # Ask simple binary question about the true class
                prompt = (
                    f"Look at this image carefully. Do you see a {class_name}? "
                    "Answer with just 'yes' or 'no'."
                )
                
                # Call Ollama API with low temperature for deterministic results
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [temp_image_path]
                        }
                    ],
                    options={
                        'temperature': self.temperature,
                    }
                )
                
                response_text = response.message.content.strip()
                answer = self._clean_response(response_text)
                
                # If LLM says "yes", it correctly identified the class
                # If LLM says "no", it failed to identify the class
                if answer == 'yes':
                    return (img_index, true_label)  # Correct!
                else:
                    # Return a different class to mark as incorrect
                    # Use -1 as a marker, or pick a different class
                    wrong_class = (true_label + 1) % len(self.class_names)
                    return (img_index, wrong_class)  # Incorrect!
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except Exception:
                        pass
        
        except Exception as e:
            logging.error(f"Error predicting image {img_index} with BinaryLLMJudge: {e}")
            return (img_index, -1)
    
    def predict(self, images: List[torch.Tensor], true_labels: List[int] = None, **kwargs) -> np.ndarray:
        """
        Predict classes for given images using binary yes/no questions.
        
        Args:
            images: List of image tensors (C, H, W) - normalized ImageNet format
            true_labels: List of true labels (optional, for targeted questioning)
            **kwargs: Additional parameters
            
        Returns:
            Array of predicted class indices (shape: [batch_size])
        """
        if len(images) == 0:
            return np.array([], dtype=np.int64)
        
        # Get true labels if provided
        if true_labels is None:
            true_labels = [0] * len(images)  # Fallback
        
        # Process images in parallel
        max_workers = min(MAX_PARALLEL_WORKERS, len(images))
        predictions = [None] * len(images)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._predict_single_image, img, true_labels[idx], idx): idx
                for idx, img in enumerate(images)
            }
            
            for future in as_completed(future_to_idx):
                try:
                    img_idx, class_idx = future.result()
                    predictions[img_idx] = class_idx
                except Exception as e:
                    idx = future_to_idx[future]
                    logging.error(f"Unexpected error processing image {idx}: {e}")
                    predictions[idx] = -1
        
        return np.array(predictions, dtype=np.int64)

