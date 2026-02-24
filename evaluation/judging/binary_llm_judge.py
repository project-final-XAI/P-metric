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
from typing import Tuple, Union, Optional
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
                "   Shape: Irregular shape, often with dense, thick cytoplasm. May appear as a single cell or in clusters.\n" 
                "   Nucleus: Vesicular nuclei, often similar in appearance to koilocytotic nuclei.\n"
                "   Cytoplasm: Orangeophilic cytoplasm (brilliant orange/yellow staining) due to abnormal keratinization."
            )
        }
        return "\n\n".join(descriptions.values())

    def _get_system_prompt(self, dataset_name: str) -> str:
        """
        Get dataset-specific system prompt.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            System prompt string appropriate for the dataset
        """
        if dataset_name in ["SIPaKMeD", "SIPaKMeD_cropped"]:
            class_descriptions = self._get_sipakmed_class_descriptions()
            return (
                "You are an expert medical cytologist with years of experience analyzing cervical cell images. "
                "You have a systematic, analytical approach to cell classification.\n\n"
                "CERVICAL CELL TYPE DESCRIPTIONS:\n"
                f"{class_descriptions}\n\n"
                "YOUR ANALYSIS METHODOLOGY:\n"
                "1. SYSTEMATIC OBSERVATION: First, carefully observe the entire image - note what type of image it is (cell, heatmap, occluded)\n"
                "2. FEATURE IDENTIFICATION: Systematically examine each feature category:\n"
                "   - Cell shape and overall morphology\n"
                "   - Nucleus characteristics (size, shape, position, staining)\n"
                "   - Cytoplasm appearance (color, density, texture, staining patterns)\n"
                "   - Distinctive features (haloing, vacuoles, special staining)\n"
                "3. PATTERN MATCHING: Compare observed features against the cell type descriptions\n"
                "4. DECISION LOGIC:\n"
                "   - If you see KEY DISTINCTIVE FEATURES of the given cell type → YES\n"
                "   - If features are CONSISTENT with the cell type (even partially visible) → YES\n"
                "   - If you see features of a DIFFERENT cell type → NO\n"
                "   - If no cell features are visible at all → NO\n"
                "5. BE THOROUGH: Consider all visible features, not just one aspect\n\n"
                "Return your response as JSON with a boolean 'is_match' field. "
                "Set is_match=true if your systematic analysis indicates the image shows the given cell type, false otherwise."
            )
        elif dataset_name == "imagenet":
            return (
                "You are an image classifier. Use the ImageNet categories to classify images. "
                "Return your response as JSON with a boolean 'is_match' field."
            )
        else:
            return (
                "You are an image classifier. Analyze the image and determine if it matches the given category. "
                "Return your response as JSON with a boolean 'is_match' field."
            )

    def __init__(self, model_name: str, dataset_name: str = "imagenet", temperature: float = 0.0):
        """
        Initialize Binary LLM Judge.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.2-vision")
            dataset_name: Dataset name to get class names from
            temperature: Temperature for LLM (0.0 = deterministic, recommended)
        """
        system_prompt = self._get_system_prompt(dataset_name)
        super().__init__(model_name, dataset_name, system_prompt=system_prompt)
        self.temperature = temperature
        self.dataset_name = dataset_name
        logging.info(f"BinaryLLMJudge initialized with {len(self.class_names)} classes, dataset={dataset_name}, temperature={temperature}")

    def _predict_single_image(
            self,
            image_data: Union[str, bytes],
            true_label: int,
            image_id: str,
            context: Optional[dict] = None
    ) -> Tuple[str, int, str, str, str, Union[int, str], Union[str, int], str]:
        """
        Predict class for a single image using binary yes/no question (optimized - in-memory processing).
        
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
            context = context or {}
            occl = context.get("occlusion_level", "?")
            strategy = context.get("fill_strategy", "?")
            method = context.get("method", "?")

            # Validate true_label
            if true_label < 0 or true_label >= len(self.class_names):
                logging.warning(f"Invalid true_label {true_label} for image {image_id}. Using fallback.")
                # Return consistent 8-tuple with error marker
                return (image_id, -1, "invalid_label", "error", "unknown", occl, strategy, "")

            # Get the true class name and format it (strips Latin names after comma)
            class_name = self._format_class_name(self.class_names[true_label])

            # Dataset-aware prompt - cognitive priming approach
            if self.dataset_name in ["SIPaKMeD", "SIPaKMeD_cropped"]:
                # Get specific description for this class (handle name variations)
                class_name_lower = class_name.lower().replace("-", " ").replace("_", " ")
                class_specific_desc = ""
                
                if "superficial" in class_name_lower or "intermediate" in class_name_lower:
                    class_specific_desc = (
                        "KEY FEATURES: Large, flat, polygonal shape with clear margins. "
                        "Small dark nucleus (superficial) OR larger vesicular nucleus (intermediate). "
                        "Abundant cytoplasm - pink/red or blue/green staining. "
                        "If you see these features (even partially), it's likely Superficial-Intermediate."
                    )
                elif "parabasal" in class_name_lower:
                    class_specific_desc = (
                        "KEY FEATURES: Smaller round/oval cells with distinct borders. "
                        "Large nucleus taking up ~50% of cell (high nucleus-to-cytoplasm ratio). "
                        "Dense, dark-stained cytoplasm. "
                        "If you see a small cell with a large nucleus, it's likely Parabasal."
                    )
                elif "metaplastic" in class_name_lower:
                    class_specific_desc = (
                        "KEY FEATURES: Round cells with very prominent, clear borders. "
                        "Nucleus is off-center (eccentric), not in the middle. "
                        "May have a large vacuole (empty space) inside. "
                        "Staining is lighter in center, darker at edges. "
                        "If you see round cells with off-center nucleus, it's likely Metaplastic."
                    )
                elif "koilocytotic" in class_name_lower or "koilocyte" in class_name_lower:
                    class_specific_desc = (
                        "KEY FEATURES: Larger than normal cells. "
                        "Dark, irregular nucleus (may look wrinkled). "
                        "MOST IMPORTANT: Clear halo/ring around the nucleus (perinuclear haloing) - this is the defining feature! "
                        "If you see a clear zone around a dark nucleus, it's likely Koilocytotic."
                    )
                elif "dyskeratotic" in class_name_lower or "dyskeratosis" in class_name_lower:
                    class_specific_desc = (
                        "KEY FEATURES: Dense, irregular cytoplasm. "
                        "Vesicular (bubbly-looking) nuclei. "
                        "MOST IMPORTANT: Orange/yellow/orangeophilic staining - this is the defining feature! "
                        "If you see orange or yellow coloring in the cytoplasm, it's likely Dyskeratotic."
                    )
                else:
                    # Fallback: use general description
                    class_specific_desc = (
                        "Examine the cell shape, nucleus characteristics, and cytoplasmic features carefully. "
                        "Compare against the detailed descriptions provided in the system prompt."
                    )
                
                prompt = (
                    f"Analyze this image systematically to determine if it shows a {class_name} cell type.\n\n"
                    f"EXPECTED FEATURES FOR {class_name}:\n{class_specific_desc}\n\n"
                    f"STEP-BY-STEP ANALYSIS:\n"
                    f"Step 1 - IMAGE TYPE: What type of image is this? (cell image, heatmap, occluded/partial image)\n"
                    f"Step 2 - CELL SHAPE: What shape do you observe? (large/flat, small/round, irregular, etc.)\n"
                    f"Step 3 - NUCLEUS: What do you see about the nucleus? (size relative to cell, position, shape, staining)\n"
                    f"Step 4 - CYTOPLASM: What do you observe about the cytoplasm? (color, staining, density, texture)\n"
                    f"Step 5 - DISTINCTIVE FEATURES: Do you see any special features? (haloing, vacuoles, orange/yellow staining, etc.)\n"
                    f"Step 6 - COMPARISON: Compare your observations from Steps 2-5 against the expected features for {class_name} above\n"
                    f"Step 7 - DECISION:\n"
                    f"   - If you observed KEY FEATURES of {class_name} → is_match = true\n"
                    f"   - If your observations are CONSISTENT with {class_name} (even if some features are partially visible) → is_match = true\n"
                    f"   - If you clearly see features of a DIFFERENT cell type → is_match = false\n"
                    f"   - If no cell features are visible → is_match = false\n\n"
                    f"Think through each step carefully. Return JSON: {{\"is_match\": true/false}} based on your systematic analysis."
                )
            elif self.dataset_name == "imagenet":
                prompt = f"What do you see in the picture? Is it a {class_name} from the ImageNet database?"
            else:
                prompt = f"What do you see in the picture? Is it a {class_name}?"

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
                logging.warning(f"JSON parsing failed for image {image_id}, attempting text fallback: {e}")
                response_lower = response_text.lower().strip()
                # Simple heuristic: look for "yes" or "no" in response
                if any(word in response_lower for word in ['yes', 'true', 'correct', 'match']):
                    result = ValidationResponse(is_match=True)
                elif any(word in response_lower for word in ['no', 'false', 'incorrect', 'not']):
                    result = ValidationResponse(is_match=False)
                else:
                    # Default to False if unclear
                    logging.warning(f"Unclear response for image {image_id}: '{response_text}'. Defaulting to False.")
                    result = ValidationResponse(is_match=False)

            # Compact response for logging
            response_clean = " ".join(response_text.split())

            # Clean, readable log format: level% - category - method - fill: {response} -> YES/NO
            if result.is_match:
                logging.info(f"{occl}% - {class_name} - {method} - {strategy}: {response_clean} -> YES")
                return image_id, true_label, response_clean, "yes", class_name, occl, strategy, prompt  # Correct!
            else:
                # Return a different class to mark as incorrect
                wrong_class = (true_label + 1) % len(self.class_names)
                logging.info(f"{occl}% - {class_name} - {method} - {strategy}: {response_clean} -> NO")
                return image_id, wrong_class, response_clean, "no", class_name, occl, strategy, prompt  # Incorrect!

        except Exception as e:
            logging.error(f"Error predicting image {image_id} with BinaryLLMJudge: {e}")
            # Return consistent 8-tuple with error marker (-1 for predicted class)
            context = context or {}
            return (image_id, -1, str(e), "error", "unknown", context.get("occlusion_level", "?"),
                    context.get("fill_strategy", "?"), "")
