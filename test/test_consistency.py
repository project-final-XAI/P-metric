"""
Consistency Test for LLM Judge.

Tests if the same images give consistent results across multiple runs.
This helps identify if there's state degradation in Ollama over time.
"""

import sys
import logging
import time
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from evaluation.judging.binary_llm_judge import BinaryLLMJudge
from data.imagenet_class_mapping import get_cached_mapping, format_class_for_llm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Images that got TRUE in the original run (easy ones that should always succeed)
TEST_IMAGES = [
    ("image_00000", 0, "tench"),           # Fish - very distinctive
    ("image_00001", 1, "goldfish"),        # Fish - iconic
    ("image_00003", 3, "tiger shark"),     # Shark - distinctive
    ("image_00007", 7, "cock"),            # Rooster - distinctive
    ("image_00009", 9, "ostrich"),         # Bird - very distinctive
    ("image_00033", 33, "loggerhead"),     # Turtle - distinctive
    ("image_00071", 71, "scorpion"),       # Arachnid - distinctive
    ("image_00130", 130, "flamingo"),      # Bird - iconic pink
    ("image_00291", 291, "lion"),          # Mammal - iconic
    ("image_00292", 292, "tiger"),         # Mammal - iconic stripes
]

# Also test some that FAILED (to see if they consistently fail)
FAILED_IMAGES = [
    ("image_00022", 22, "bald eagle"),     # Should be easy but failed!
    ("image_00049", 49, "African crocodile"),  # Should be easy but failed!
    ("image_00051", 51, "triceratops"),    # Should be easy but failed!
]


def find_image_path(image_id: str, base_dir: Path) -> Path:
    """Find the occluded image path for a given image ID."""
    # Structure: results/occluded/imagenet/mobilenet_v2/gray/saliency/0/{model}-{method}-{image_id}.png
    occluded_dir = base_dir / "results" / "occluded" / "imagenet" / "mobilenet_v2" / "gray" / "saliency" / "0"
    
    # Look for the file
    pattern = f"*-{image_id}.png"
    matches = list(occluded_dir.glob(pattern))
    
    if matches:
        return matches[0]
    
    # Try alternative pattern
    pattern2 = f"mobilenet_v2-saliency-{image_id}.png"
    direct_path = occluded_dir / pattern2
    if direct_path.exists():
        return direct_path
    
    return None


def run_consistency_test(
    num_rounds: int = 10,
    include_failed: bool = True
):
    """
    Run consistency test - same images multiple times.
    
    Args:
        num_rounds: Number of times to test each image
        include_failed: Whether to also test images that failed originally
    """
    base_dir = Path(config.BASE_DIR)
    
    # Initialize judge
    logging.info("Initializing BinaryLLMJudge...")
    judge = BinaryLLMJudge("llama3.2-vision-binary", "imagenet", temperature=0.0)
    
    # Prepare test images
    test_set = TEST_IMAGES.copy()
    if include_failed:
        test_set.extend(FAILED_IMAGES)
    
    # Find actual image paths
    test_data = []
    for image_id, label, class_name in test_set:
        path = find_image_path(image_id, base_dir)
        if path and path.exists():
            test_data.append((image_id, label, class_name, str(path)))
            logging.info(f"Found: {image_id} -> {path.name}")
        else:
            logging.warning(f"Image not found: {image_id}")
    
    if not test_data:
        logging.error("No test images found! Check paths.")
        return
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting consistency test: {len(test_data)} images × {num_rounds} rounds")
    logging.info(f"{'='*60}\n")
    
    # Track results
    results = defaultdict(list)  # image_id -> list of (round, result, response)
    
    # Run multiple rounds
    for round_num in range(1, num_rounds + 1):
        logging.info(f"\n--- Round {round_num}/{num_rounds} ---")
        
        for image_id, label, class_name, image_path in test_data:
            start_time = time.time()
            
            # Call judge
            try:
                result = judge._predict_single_image(
                    image_data=image_path,
                    true_label=label,
                    image_id=f"test-{image_id}",
                    context={"occlusion_level": 0, "fill_strategy": "gray", "method": "saliency"}
                )
                
                # Extract result
                if len(result) >= 4:
                    _, pred_label, response, conclusion = result[0], result[1], result[2], result[3]
                    is_correct = (pred_label == label)
                else:
                    is_correct = False
                    conclusion = "error"
                    response = str(result)
                
                elapsed = time.time() - start_time
                results[image_id].append((round_num, is_correct, conclusion, elapsed))
                
                status = "✓" if is_correct else "✗"
                logging.info(f"  {status} {class_name}: {conclusion} ({elapsed:.2f}s)")
                
            except Exception as e:
                logging.error(f"  Error testing {image_id}: {e}")
                results[image_id].append((round_num, False, "error", 0))
        
        # Small delay between rounds
        if round_num < num_rounds:
            time.sleep(1)
    
    # Print summary
    print_summary(results, test_data, num_rounds)


def print_summary(results: dict, test_data: list, num_rounds: int):
    """Print summary of consistency test results."""
    logging.info(f"\n{'='*60}")
    logging.info("CONSISTENCY TEST SUMMARY")
    logging.info(f"{'='*60}\n")
    
    consistent_count = 0
    inconsistent_count = 0
    
    for image_id, label, class_name, _ in test_data:
        image_results = results.get(image_id, [])
        
        if not image_results:
            logging.warning(f"{class_name} ({image_id}): No results")
            continue
        
        # Count TRUE/FALSE
        true_count = sum(1 for _, is_correct, _, _ in image_results if is_correct)
        false_count = len(image_results) - true_count
        
        # Check consistency
        all_same = (true_count == 0 or false_count == 0)
        
        if all_same:
            consistent_count += 1
            status = "CONSISTENT"
            emoji = "✓" if true_count > 0 else "✗"
        else:
            inconsistent_count += 1
            status = "INCONSISTENT!"
            emoji = "⚠"
        
        # Calculate average time
        avg_time = sum(t for _, _, _, t in image_results) / len(image_results)
        
        logging.info(
            f"{emoji} {class_name:25s} ({image_id}): "
            f"TRUE={true_count}/{num_rounds}, FALSE={false_count}/{num_rounds} "
            f"[{status}] (avg {avg_time:.2f}s)"
        )
    
    # Overall summary
    logging.info(f"\n{'='*60}")
    logging.info(f"OVERALL: {consistent_count} consistent, {inconsistent_count} inconsistent")
    
    if inconsistent_count > 0:
        logging.warning(
            f"\n⚠ WARNING: {inconsistent_count} images gave DIFFERENT results across runs!"
            f"\n   This suggests non-deterministic behavior despite temperature=0.0"
        )
    else:
        logging.info(
            f"\n✓ All images gave consistent results across {num_rounds} runs."
            f"\n   The model is deterministic - failures are due to model capability, not state issues."
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM Judge consistency")
    parser.add_argument("--rounds", type=int, default=10, help="Number of test rounds")
    parser.add_argument("--no-failed", action="store_true", help="Skip testing failed images")
    
    args = parser.parse_args()
    
    run_consistency_test(
        num_rounds=args.rounds,
        include_failed=not args.no_failed
    )

