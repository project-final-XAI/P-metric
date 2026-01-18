"""
LLM Judge - integer class prediction.

Asks the model to return a class index for each image.
Supports multiple datasets (ImageNet, SIPaKMeD, etc.) with dataset-aware prompts.
Optimized for high-throughput inference with llama3.2-vision via Ollama.
"""

import logging
import os
import time
from typing import Tuple, Union, Optional, Dict

from pydantic import BaseModel, ValidationError, conint
import ollama

from evaluation.judging.base_llm_judge import (
    BaseLLMJudge, 
    OLLAMA_KEEP_ALIVE, 
    OLLAMA_NUM_CTX, 
    OLLAMA_SEED,
    OLLAMA_RESET_INTERVAL
)


def create_class_id_response_model(max_classes: int = 1000):
    """
    Create a dynamic ClassIdResponse model with dataset-specific class limits.
    
    Args:
        max_classes: Maximum number of classes in the dataset
        
    Returns:
        Pydantic BaseModel class for class ID validation
    """
    class ClassIdResponse(BaseModel):
        """Structured response schema for class-id prediction."""
        class_id: conint(ge=1, le=max_classes)
    
    return ClassIdResponse


class ClassIdLLMJudge(BaseLLMJudge):
    """
    LLM judge that requests a class index (0-based) per image.
    
    Optimized with:
    - Shared retry logic via _call_ollama_with_retry
    - keep_alive to prevent model reloading
    - num_ctx for optimal context window
    - Temperature=0 for deterministic results
    - Robust error handling and validation
    - Dataset-aware prompts for different datasets
    """

    def _get_sipakmed_class_descriptions(self) -> str:
        """
        Get detailed descriptions of SIPaKMeD cell classes.
        
        Returns:
            Formatted string with class descriptions
        """
        descriptions = {
            "Superficial-Intermediate": (
                "1. Superficial-Intermediate (Normal)\n"
                "   Shape: Large, flat, and polygonal with well-defined margins.\n"
                "   Nucleus: Contains a central, small, and dark (pycnotic) nucleus in superficial cells, "
                "or a larger vesicular nucleus in intermediate cells.\n"
                "   Cytoplasm: Abundant and typically stains eosinophilic (pink/red) or cyanophilic (blue/green)."
            ),
            "Parabasal": (
                "2. Parabasal (Normal)\n"
                "   Shape: Smaller than superficial cells; typically round to oval with dense, distinct borders.\n"
                "   Nucleus: Large relative to the cell size (high N/C ratio), occupying roughly half the cell.\n"
                "   Cytoplasm: Fairly dense and dark-stained."
            ),
            "Metaplastic": (
                "3. Metaplastic (Benign)\n"
                "   Shape: Similar to parabasal cells but with very prominent cellular borders and an almost perfectly round cytoplasm.\n"
                "   Nucleus: Often eccentric (off-center) rather than central.\n"
                "   Distinct Feature: May contain a large intracellular vacuole; staining is often lighter in the center than at the margins."
            ),
            "Koilocytotic": (
                "4. Koilocytotic (Abnormal)\n"
                "   Shape: Variable, but often exhibits cellular enlargement.\n"
                "   Nucleus: Irregular, hyperchromatic (darker), and may show 'raisin-like' wrinkling.\n"
                "   Distinct Feature: Characterized by perinuclear haloing—a clear, transparent zone surrounding the nucleus—caused by HPV infection."
            ),
            "Dyskeratotic": (
                "5. Dyskeratotic (Abnormal)\n"
                "   Shape: Often found in thick, three-dimensional clusters where margins are difficult to distinguish.\n"
                "   Nucleus: Vesicular nuclei, often similar in appearance to koilocytotic nuclei.\n"
                "   Distinct Feature: Exhibits orangeophilic cytoplasm (brilliant orange/yellow staining) due to premature or abnormal keratinization."
            )
        }
        return "\n\n".join(descriptions.values())

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
        self.dataset_name = dataset_name
        logging.info(f"ClassIdLLMJudge initialized with {len(self.class_names)} classes, dataset={dataset_name}, temperature={temperature}")

    def _predict_single_image(
            self,
            image_data: Union[str, bytes],
            true_label: int,
            image_id: str,
            context: Optional[Dict] = None
    ) -> Tuple[str, int]:
        """
        Predict class for a single image by requesting a class index.
        
        Uses structured outputs (JSON mode) for reliable parsing. Accepts both file paths
        and base64-encoded image strings. Uses shared retry logic for consistency and reliability.
        
        Args:
            image_data: Image file path (str) or base64-encoded image string (str)
            true_label: True class label
            image_id: Stable identifier for the image (e.g., file stem)
            context: Optional metadata (occlusion level, fill strategy, etc.)
            
        Returns:
            Tuple of (image_id, predicted_class_index)
        """
        try:
            # Validate true_label
            if true_label < 0 or true_label >= len(self.class_names):
                logging.warning(f"Invalid true_label {true_label} for image {image_id}. Using fallback.")
                return (image_id, -1)
            
            # Build list of all categories with formatted names for system prompt
            # Format: "1: tench", "2: goldfish", etc. (1-based indexing)
            formatted_categories = []
            for i, class_name in enumerate(self.class_names):
                formatted_name = self._format_class_name(class_name)
                # Use 1-based indexing: class 0 becomes 1, class 1 becomes 2, etc.
                formatted_categories.append(f"{i+1}: {formatted_name}")
            
            # Reverse the list so it starts from highest number and goes down to 1
            formatted_categories.reverse()
            
            # Create dataset-aware system prompt with categories list
            class_list = "\n".join(formatted_categories)
            
            if self.dataset_name in ["SIPaKMeD", "SIPaKMeD_cropped"]:
                # Get detailed class descriptions
                class_descriptions = self._get_sipakmed_class_descriptions()
                
                categories_system_prompt = (
                    "You are an expert medical cytologist specializing in cervical cell classification. "
                    "You analyze microscopic images of cervical cells and classify them into specific cell types.\n\n"
                    "CERVICAL CELL TYPE DESCRIPTIONS:\n"
                    f"{class_descriptions}\n\n"
                    f"Here are all the cervical cell types indexed from 1 to {len(self.class_names)}:\n"
                    f"{class_list}\n\n"
                    "IMPORTANT: You must carefully examine:\n"
                    "- Cell shape and margins\n"
                    "- Nucleus size, shape, position, and staining characteristics\n"
                    "- Cytoplasm characteristics (color, density, texture)\n"
                    "- Any distinctive features (perinuclear haloing, vacuoles, orangeophilic staining, etc.)\n\n"
                    f"Return the EXACT number (1-{len(self.class_names)}) that corresponds to the correct cell type based on these characteristics."
                )
                prompt = (
                    "Carefully examine this microscopic cervical cell image. "
                    "Analyze the cell morphology, nucleus characteristics, cytoplasmic features, and any distinctive patterns. "
                    "Match the observed features to one of the cell type descriptions in the system prompt. "
                    f"Return ONLY the number (1-{len(self.class_names)}) that corresponds to the correct cell type. "
                )
            elif self.dataset_name == "imagenet":
                categories_system_prompt = (
                    "You are an expert image classifier. You will be shown an image and must identify which ImageNet category it belongs to.\n\n"
                    f"Here are all the ImageNet categories indexed from 1 to {len(self.class_names)}:\n"
                    f"{class_list}\n\n"
                    f"IMPORTANT: You must carefully examine the image and return the EXACT number (1-{len(self.class_names)}) that matches what you see in the image. "
                )
                prompt = (
                    "Carefully examine the image provided. "
                    "Identify the main object or subject in the image. "
                    "Match it to one of the categories listed in the system prompt. "
                    f"Return ONLY the number (1-{len(self.class_names)}) that corresponds to the correct ImageNet category. "
                )
            else:
                categories_system_prompt = (
                    "You are an expert image classifier. You will be shown an image and must identify which category it belongs to.\n\n"
                    f"Here are all the categories indexed from 1 to {len(self.class_names)}:\n"
                    f"{class_list}\n\n"
                    f"IMPORTANT: You must carefully examine the image and return the EXACT number (1-{len(self.class_names)}) that matches what you see in the image. "
                )
                prompt = (
                    "Carefully examine the image provided. "
                    "Identify the main object or subject in the image. "
                    "Match it to one of the categories listed in the system prompt. "
                    f"Return ONLY the number (1-{len(self.class_names)}) that corresponds to the correct category. "
                )

            # Create dynamic response model based on dataset size
            ClassIdResponseModel = create_class_id_response_model(len(self.class_names))
            
            # Call Ollama with custom system prompt containing categories list
            response_text = self._call_ollama_with_categories(
                prompt=prompt,
                image_data=image_data,
                categories_system_prompt=categories_system_prompt,
                max_retries=3,
                temperature=self.temperature,
                format_schema=ClassIdResponseModel.model_json_schema()
            )

            # Parse structured response with robust error handling
            try:
                result = ClassIdResponseModel.model_validate_json(response_text)
                predicted_1based = result.class_id  # This is 1-1000
                # Debug: log the 1-based prediction before conversion (first few times)
                if not hasattr(self, '_debug_conversion_count'):
                    self._debug_conversion_count = 0
                self._debug_conversion_count += 1
                if self._debug_conversion_count <= 5:
                    logging.info(f"DEBUG Conversion #{self._debug_conversion_count}: 1-based={predicted_1based}, converting to 0-based={predicted_1based - 1}")
                # Convert back to 0-based indexing for internal use
                predicted = predicted_1based - 1
            except ValidationError as e:
                # Fallback: try to parse as plain text if JSON parsing fails
                logging.warning(f"JSON parsing failed for image {image_id}, attempting text fallback: {e}")
                try:
                    predicted_1based = int(response_text.strip())
                    # Debug: log the 1-based prediction before conversion
                    if not hasattr(self, '_debug_response_conversion_printed'):
                        logging.info(f"DEBUG: Raw 1-based prediction from text parsing: {predicted_1based}")
                        self._debug_response_conversion_printed = True
                    # Convert from 1-based to 0-based
                    predicted = predicted_1based - 1
                except Exception:
                    logging.warning(f"Unclear response for image {image_id}: '{response_text}'. Defaulting to -1.")
                    predicted = -1

            # Clamp to valid range (0-based: 0-999)
            if predicted < 0 or predicted >= len(self.class_names):
                logging.debug(f"Image {image_id}: predicted class {predicted} out of range, defaulting to -1")
                predicted = -1

            # Debug: print comparison when prediction is wrong
            if predicted != true_label:
                true_class_name = self._format_class_name(self.class_names[true_label]) if true_label >= 0 and true_label < len(self.class_names) else "unknown"
                predicted_class_name = self._format_class_name(self.class_names[predicted]) if predicted >= 0 and predicted < len(self.class_names) else "unknown"
                logging.info(f"❌ MISMATCH for {image_id}: Predicted={predicted} ({predicted_class_name}) vs True={true_label} ({true_class_name})")
            else:
                # Also log correct predictions occasionally (every 10th correct prediction to avoid spam)
                if not hasattr(self, '_correct_count'):
                    self._correct_count = 0
                self._correct_count += 1
                if self._correct_count % 10 == 0:
                    true_class_name = self._format_class_name(self.class_names[true_label]) if true_label >= 0 and true_label < len(self.class_names) else "unknown"
                    logging.debug(f"✓ Correct for {image_id}: {predicted} ({true_class_name})")

            return image_id, predicted

        except Exception as e:
            logging.error(f"Error predicting image {image_id} with ClassIdLLMJudge: {e}")
            return (image_id, -1)
    
    def _call_ollama_with_categories(
        self,
        prompt: str,
        image_data: Union[str, bytes],
        categories_system_prompt: str,
        max_retries: int = 3,
        temperature: float = 0.0,
        format_schema: Optional[Dict] = None
    ) -> str:
        """
        Call Ollama API with custom system prompt containing categories list.
        
        This is specific to ClassIdLLMJudge and includes the categories list in system prompt.
        
        Args:
            prompt: User prompt text
            image_data: Image file path (str) or base64-encoded image string (str)
            categories_system_prompt: System prompt with categories list
            max_retries: Maximum number of retry attempts
            temperature: Temperature for LLM (0.0 = deterministic)
            format_schema: Optional Pydantic JSON schema for structured outputs
        
        Returns:
            Response text from LLM
        """
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
        ollama_options = {
            'temperature': temperature,
            'seed': OLLAMA_SEED,
            'num_ctx': OLLAMA_NUM_CTX,
        }
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Build messages list with categories in system prompt
                messages = [
                    {
                        'role': 'system',
                        'content': categories_system_prompt
                    },
                    {
                        'role': 'system',
                        'content': self.system_prompt
                    },
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_data]
                    }
                ]
                
                # Debug: print prompts once to verify they're correct (only on first call)
                if not hasattr(self, '_debug_printed'):
                    logging.info("=" * 80)
                    logging.info("CLASSID LLM JUDGE - PROMPT DEBUG")
                    logging.info("=" * 80)
                    logging.info(f"System Prompt (categories):\n{categories_system_prompt[:500]}...")  # First 500 chars
                    logging.info(f"\nSystem Prompt (base):\n{self.system_prompt}")
                    logging.info(f"\nUser Prompt:\n{prompt}")
                    # Debug: check if image_data is valid
                    if isinstance(image_data, str):
                        if os.path.exists(image_data):
                            logging.info(f"Image path: {image_data} (exists)")
                        else:
                            logging.info(f"Image data: base64 string (length: {len(image_data)} chars)")
                    else:
                        logging.info(f"Image data type: {type(image_data)}")
                    logging.info("=" * 80)
                    self._debug_printed = True
                
                response = ollama.chat(
                    model=self.ollama_model_name,
                    messages=messages,
                    options=ollama_options,
                    keep_alive=OLLAMA_KEEP_ALIVE,
                    format=format_schema
                )
                
                response_text = response.message.content.strip()
                
                # Debug: print response for first few calls to see what model returns
                if not hasattr(self, '_debug_response_count'):
                    self._debug_response_count = 0
                self._debug_response_count += 1
                if self._debug_response_count <= 5:
                    logging.info(f"DEBUG Response #{self._debug_response_count}: Raw response from Ollama: '{response_text}'")
                
                return response_text
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logging.warning(f"Ollama call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All Ollama retry attempts failed: {e}")
        
        raise last_exception or Exception("Failed to call Ollama API")

