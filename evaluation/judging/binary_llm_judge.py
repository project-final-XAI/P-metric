"""
Binary LLM Judge - Yes/No approach.

Asks the model "Do you see {true_category}? Answer yes or no".
Simple and direct approach - only one question per image.
- "yes" = correct prediction
- "no" = incorrect prediction

Most efficient and straightforward evaluation method.
Optimized for high-throughput inference with llama3.2-vision via Ollama.
"""
import logging
from typing import Tuple, Union
from pydantic import BaseModel, ValidationError

from evaluation.judging.base_llm_judge import BaseLLMJudge


class ValidationResponse(BaseModel):
    """Structured response schema for binary validation."""
    is_match: bool


class BinaryLLMJudge(BaseLLMJudge):
    """
    Binary LLM Judge using Yes/No questions.
    
    For each image, asks "Do you see {true_category}?" and evaluates:
    - Answer "yes" → Correct (returns true_label)
    - Answer "no" → Incorrect (returns different label)
    
    This is the most efficient approach - only one question per image.
    Optimized with:
    - Shared retry logic via _call_ollama_with_retry
    - keep_alive to prevent model reloading
    - num_ctx for optimal context window
    - Temperature=0 for deterministic results
    - Robust error handling and validation
    """

    def __init__(self, model_name: str, dataset_name: str = "imagenet", temperature: float = 0.0):
        """
        Initialize Binary LLM Judge.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.2-vision")
            dataset_name: Dataset name to get class names from
            temperature: Temperature for LLM (0.0 = deterministic, recommended)
        """
        super().__init__(model_name, dataset_name)
        self.temperature = temperature
        logging.info(f"BinaryLLMJudge initialized with {len(self.class_names)} classes, temperature={temperature}")

    def _predict_single_image(
            self,
            image_data: Union[str, bytes],
            true_label: int,
            img_index: int
    ) -> Tuple[int, int]:
        """
        Predict class for a single image using binary yes/no question (optimized - in-memory processing).
        
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
            
            # Get the true class name and format it (strips Latin names after comma)
            class_name = self._format_class_name(self.class_names[true_label])

            # Optimized prompt - cognitive priming approach
            prompt = f"What do you see in the picture? Is it a {class_name} from the imagenet database?"

            # Call Ollama with structured outputs using shared retry helper
            # This ensures consistency, retry logic, keep_alive, and num_ctx optimization
            response_text = self._call_ollama_with_retry(
                prompt=prompt,
                image_data=image_data,
                max_retries=3,
                temperature=self.temperature,  # Use instance temperature (should be 0.0)
                format_schema=ValidationResponse.model_json_schema()  # JSON schema for structured output
            )

            # Parse structured response with robust error handling
            try:
                result = ValidationResponse.model_validate_json(response_text)
            except ValidationError as e:
                # Fallback: try to parse as plain text if JSON parsing fails
                logging.warning(f"JSON parsing failed for image {img_index}, attempting text fallback: {e}")
                response_lower = response_text.lower().strip()
                # Simple heuristic: look for "yes" or "no" in response
                if any(word in response_lower for word in ['yes', 'true', 'correct', 'match']):
                    result = ValidationResponse(is_match=True)
                elif any(word in response_lower for word in ['no', 'false', 'incorrect', 'not']):
                    result = ValidationResponse(is_match=False)
                else:
                    # Default to False if unclear
                    logging.warning(f"Unclear response for image {img_index}: '{response_text}'. Defaulting to False.")
                    result = ValidationResponse(is_match=False)

            # If LLM says match, it correctly identified the class
            # If LLM says no match, it failed to identify the class
            if result.is_match:
                logging.debug(f"Image {img_index} ({class_name}): LLM confirmed match")
                return img_index, true_label  # Correct!
            else:
                # Return a different class to mark as incorrect
                # Use modulo to ensure valid index
                wrong_class = (true_label + 1) % len(self.class_names)
                logging.debug(f"Image {img_index} ({class_name}): LLM said no match, returning class {wrong_class}")
                return img_index, wrong_class  # Incorrect!

        except Exception as e:
            logging.error(f"Error predicting image {img_index} with BinaryLLMJudge: {e}")
            return (img_index, -1)
