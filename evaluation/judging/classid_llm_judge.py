"""
LLM Judge - integer class prediction.

Asks the model to return an ImageNet class index (0-999) for each image.
Optimized for high-throughput inference with llama3.2-vision via Ollama.
"""

import logging
from typing import Tuple, Union

from pydantic import BaseModel, ValidationError, conint

from evaluation.judging.base_llm_judge import BaseLLMJudge


class ClassIdResponse(BaseModel):
    """Structured response schema for class-id prediction (0-999)."""
    class_id: conint(ge=0, le=999)


class ClassIdLLMJudge(BaseLLMJudge):
    """
    LLM judge that requests an ImageNet class index (0-999) per image.
    
    Optimized with:
    - Shared retry logic via _call_ollama_with_retry
    - keep_alive to prevent model reloading
    - num_ctx for optimal context window
    - Temperature=0 for deterministic results
    - Robust error handling and validation
    """

    def __init__(self, model_name: str, dataset_name: str = "imagenet", temperature: float = 0.0):
        """
        Initialize ClassId LLM Judge.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.2-vision")
            dataset_name: Dataset name to get class names from
            temperature: Temperature for LLM (0.0 = deterministic, recommended)
        """
        super().__init__(model_name, dataset_name)
        self.temperature = temperature
        logging.info(f"ClassIdLLMJudge initialized with {len(self.class_names)} classes, temperature={temperature}")

    def _predict_single_image(
            self,
            image_data: Union[str, bytes],
            true_label: int,
            img_index: int
    ) -> Tuple[int, int]:
        """
        Predict class for a single image by requesting an ImageNet class index (0-999).
        
        Uses structured outputs (JSON mode) for reliable parsing. Accepts both file paths
        and base64-encoded image strings. Uses shared retry logic for consistency and reliability.
        
        Args:
            image_data: Image file path (str) or base64-encoded image string (str)
            true_label: True class label
            img_index: Original index in batch
            
        Returns:
            Tuple of (img_index, predicted_class_index)
        """
        try:
            # Validate true_label
            if true_label < 0 or true_label >= len(self.class_names):
                logging.warning(f"Invalid true_label {true_label} for image {img_index}. Using fallback.")
                return (img_index, -1)
            
            # Prompt the LLM to return a class index 0-999 for ImageNet
            prompt = (
                "Look at the provided image and return a single integer from 0-999 "
                "representing the corresponding ImageNet class index. Return only the integer."
            )

            # Call Ollama with structured outputs using shared retry helper
            response_text = self._call_ollama_with_retry(
                prompt=prompt,
                image_data=image_data,
                max_retries=3,
                temperature=self.temperature,
                format_schema=ClassIdResponse.model_json_schema(),
                classes_names=self.class_names
            )

            # Parse structured response with robust error handling
            try:
                result = ClassIdResponse.model_validate_json(response_text)
                predicted = result.class_id
            except ValidationError as e:
                # Fallback: try to parse as plain text if JSON parsing fails
                logging.warning(f"JSON parsing failed for image {img_index}, attempting text fallback: {e}")
                try:
                    predicted = int(response_text.strip())
                except Exception:
                    logging.warning(f"Unclear response for image {img_index}: '{response_text}'. Defaulting to -1.")
                    predicted = -1

            # Clamp to valid range
            if predicted < 0 or predicted >= len(self.class_names):
                logging.debug(f"Image {img_index}: predicted class {predicted} out of range, defaulting to -1")
                predicted = -1

            return img_index, predicted

        except Exception as e:
            logging.error(f"Error predicting image {img_index} with ClassIdLLMJudge: {e}")
            return (img_index, -1)

