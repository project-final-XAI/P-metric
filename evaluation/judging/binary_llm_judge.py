"""
Binary LLM Judge - Yes/No approach.

Asks the model "Do you see {true_category}? Answer yes or no".
Simple and direct approach - only one question per image.
- "yes" = correct prediction
- "no" = incorrect prediction

Most efficient and straightforward evaluation method.
"""
import logging
from typing import Tuple

from evaluation.judging.base_llm_judge import BaseLLMJudge


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
        if 'yes' in response_lower:
            return 'yes'
        if 'no' in response_lower:
            return 'no'

        # Default to no if unclear
        return 'no'

    def _predict_single_image_from_path(
            self,
            image_path: str,
            true_label: int,
            img_index: int
    ) -> Tuple[int, int]:
        """
        Predict class for a single image using binary yes/no question (optimized - uses file path directly).
        
        Args:
            image_path: Path to image file (PNG/JPG)
            true_label: True class label
            img_index: Original index in batch
            
        Returns:
            Tuple of (img_index, predicted_class_index)
        """
        try:
            # Get the true class name
            class_name = self._format_class_name(self.class_names[true_label])

            # Binary question - based on research paper approach
            prompt = (f"You are an image classification expert. "
                      f"What do you see in the picture? "
                      f"Is it a {class_name} from the imagenet database? "
                      f"Answer with only 'Yes' or 'No'.")

            # Use shared retry helper method
            response_text = self._call_ollama_with_retry(
                prompt=prompt,
                image_path=image_path,
                max_retries=3,
                temperature=self.temperature
            )

            answer = self._clean_response(response_text)

            # If LLM says "yes", it correctly identified the class
            # If LLM says "no", it failed to identify the class
            logging.info(f"LLM response for {class_name}: '{response_text}' -> {answer}")
            if answer == 'yes':
                return img_index, true_label  # Correct!
            else:
                # Return a different class to mark as incorrect
                wrong_class = (true_label + 1) % len(self.class_names)
                return img_index, wrong_class  # Incorrect!

        except Exception as e:
            logging.error(f"Error predicting image {img_index} with BinaryLLMJudge: {e}")
            return (img_index, -1)
