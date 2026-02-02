"""
Test ClassIdLLMJudge with only the first 10 ImageNet categories.

This test helps verify if the model can correctly classify when given
a smaller set of categories (10 instead of 1000).
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from evaluation.judging.classid_llm_judge import ClassIdLLMJudge
from data.loader import get_dataloader
from data.imagenet_class_mapping import get_cached_mapping, format_class_for_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Number of categories to test
NUM_CATEGORIES = 50
# Number of images per category to test
IMAGES_PER_CATEGORY = 3


class LimitedClassIdLLMJudge(ClassIdLLMJudge):
    """
    ClassIdLLMJudge limited to first N categories.
    
    This class overrides the category list building to only include
    the first N categories in the system prompt.
    """
    
    def __init__(self, model_name: str, dataset_name: str = "imagenet", 
                 temperature: float = 0.0, num_categories: int = 10):
        """
        Initialize with limited number of categories.
        
        Args:
            model_name: Ollama model name
            dataset_name: Dataset name
            temperature: Temperature for LLM
            num_categories: Number of categories to use (first N)
        """
        # Initialize parent with full dataset
        super().__init__(model_name, dataset_name, temperature)
        
        # Limit to first N categories
        self.num_categories = num_categories
        self.original_class_names = self.class_names.copy()
        # Store limited class names for building prompts
        self.limited_class_names = self.class_names[:num_categories]
        
        logging.info(f"LimitedClassIdLLMJudge: Using first {num_categories} categories out of {len(self.original_class_names)}")
        logging.info(f"Categories: {[self._format_class_name(name) for name in self.limited_class_names]}")
    
    def _predict_single_image(
            self,
            image_data,
            true_label: int,
            image_id: str,
            context=None
    ):
        """
        Predict with category limitation.
        
        Overrides the parent to use only first N categories in the prompt.
        """
        # If true_label is outside our limited range, skip
        if true_label >= self.num_categories:
            logging.warning(f"Image {image_id}: true_label {true_label} is outside limited range (0-{self.num_categories-1})")
            return (image_id, -1)
        
        # Temporarily replace class_names with limited version
        original_class_names = self.class_names
        self.class_names = self.limited_class_names
        
        try:
            # Call parent method (which will use self.class_names)
            result = super()._predict_single_image(image_data, true_label, image_id, context)
        finally:
            # Restore original class_names
            self.class_names = original_class_names
        
        return result


def load_test_images(num_categories: int = 50, images_per_category: int = 3) -> List[Tuple[str, int, str]]:
    """
    Load test images from the first N categories.
    
    Returns:
        List of (image_path, true_label, class_name) tuples
    """
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    
    dataset_path = config.DATASET_CONFIG["imagenet"]["path"]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    dataset = ImageFolder(root=str(dataset_path), transform=transform)
    mapping = get_cached_mapping()
    
    test_images = []
    images_by_class = {}
    
    # Group images by class
    for idx, (image, label) in enumerate(dataset):
        if label >= num_categories:
            continue  # Skip categories outside our range
        
        if label not in images_by_class:
            images_by_class[label] = []
        
        # Get image path
        image_path = dataset.samples[idx][0]
        images_by_class[label].append((str(image_path), label))
    
    # Select images from each category
    for label in range(min(num_categories, len(images_by_class))):
        if label not in images_by_class:
            continue
        
        class_images = images_by_class[label][:images_per_category]
        synset_id = dataset.classes[label]
        class_name = format_class_for_llm(mapping.get(synset_id, synset_id))
        
        for image_path, _ in class_images:
            image_id = Path(image_path).stem
            test_images.append((image_path, label, class_name))
            logging.debug(f"Added test image: {image_id} -> label {label} ({class_name})")
    
    return test_images


def run_test():
    """Run the test with first 10 categories."""
    logging.info("=" * 80)
    logging.info("TESTING ClassIdLLMJudge with First 10 Categories")
    logging.info("=" * 80)
    
    # Load test images
    logging.info(f"Loading test images (first {NUM_CATEGORIES} categories, {IMAGES_PER_CATEGORY} images each)...")
    test_images = load_test_images(NUM_CATEGORIES, IMAGES_PER_CATEGORY)
    logging.info(f"Loaded {len(test_images)} test images")
    
    if not test_images:
        logging.error("No test images found! Check dataset path.")
        return
    
    # Initialize judge with limited categories
    logging.info("Initializing LimitedClassIdLLMJudge...")
    judge = LimitedClassIdLLMJudge(
        "llama3.2-vision-classid",
        "imagenet",
        temperature=0.0,
        num_categories=NUM_CATEGORIES
    )
    
    # Run predictions
    logging.info("=" * 80)
    logging.info("Running predictions...")
    logging.info("=" * 80)
    
    correct = 0
    total = 0
    results_by_class = {}
    
    for image_path, true_label, class_name in test_images:
        image_id = Path(image_path).stem
        
        try:
            # Predict
            result = judge._predict_single_image(
                image_path,
                true_label,
                image_id
            )
            
            predicted_label = result[1]
            total += 1
            
            # Track results
            if true_label not in results_by_class:
                results_by_class[true_label] = {"correct": 0, "total": 0, "name": class_name}
            results_by_class[true_label]["total"] += 1
            
            if predicted_label == true_label:
                correct += 1
                results_by_class[true_label]["correct"] += 1
                logging.info(f"✓ {image_id}: Correct! Predicted={predicted_label} ({class_name})")
            else:
                predicted_name = judge._format_class_name(judge.class_names[predicted_label]) if 0 <= predicted_label < len(judge.class_names) else "unknown"
                logging.info(f"✗ {image_id}: Wrong! Predicted={predicted_label} ({predicted_name}) vs True={true_label} ({class_name})")
        
        except Exception as e:
            logging.error(f"Error processing {image_id}: {e}")
            total += 1
    
    # Print summary
    logging.info("=" * 80)
    logging.info("RESULTS SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total images tested: {total}")
    logging.info(f"Correct predictions: {correct}")
    logging.info(f"Accuracy: {correct/total*100:.1f}%")
    logging.info("")
    logging.info("Results by category:")
    for label in sorted(results_by_class.keys()):
        info = results_by_class[label]
        acc = info["correct"] / info["total"] * 100 if info["total"] > 0 else 0
        logging.info(f"  Category {label} ({info['name']}): {info['correct']}/{info['total']} ({acc:.1f}%)")
    logging.info("=" * 80)


if __name__ == "__main__":
    run_test()

