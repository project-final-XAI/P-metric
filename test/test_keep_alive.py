"""
Test different KEEP_ALIVE values to find optimal balance between speed and accuracy.

Background:
- Ollama uses keep_alive to control how long models stay in memory
- Reports show that long keep_alive times can cause "state degradation" 
- This manifests as accuracy dropping over time within a session
- Related GitHub issues: ollama/ollama#4846, ollama/ollama#5272

Tests: 0 (no keep), 30s, 1m, 5m, 10m
For each value, runs multiple rounds and measures:
- Accuracy
- Consistency
- Speed
"""
import sys
import logging
import time
import subprocess
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test images with known expected results
TEST_IMAGES = [
    ("image_00000", 0, "tench"),
    ("image_00001", 1, "goldfish"),
    ("image_00003", 3, "tiger shark"),
    ("image_00007", 7, "cock"),
    ("image_00009", 9, "ostrich"),
    ("image_00033", 33, "loggerhead"),
    ("image_00071", 71, "scorpion"),
    ("image_00130", 130, "flamingo"),
    ("image_00291", 291, "lion"),
    ("image_00292", 292, "tiger"),
]

# Different keep_alive values to test (order: fastest to slowest)
KEEP_ALIVE_VALUES = [0, "30s", "1m", "5m"]
ROUNDS_PER_VALUE = 3


def find_image_path(image_id: str) -> Path:
    """Find the occluded image file."""
    base_path = Path("results/occluded/imagenet/mobilenet_v2/gray/saliency/0")
    for f in base_path.glob(f"*{image_id}*.png"):
        return f
    return None


def stop_ollama():
    """Stop the Ollama model to clear state."""
    try:
        result = subprocess.run(
            ["ollama", "stop", "llama3.2-vision"], 
            capture_output=True, 
            timeout=10,
            text=True
        )
        logger.info("Stopped Ollama model")
        time.sleep(2)
    except Exception as e:
        logger.warning(f"Failed to stop Ollama: {e}")


def run_test_with_keep_alive(keep_alive_value, num_rounds: int = 3):
    """Run consistency test with a specific keep_alive value."""
    # Import here to allow patching
    import evaluation.judging.base_llm_judge as base_judge
    
    # Patch the KEEP_ALIVE value
    original_keep_alive = base_judge.OLLAMA_KEEP_ALIVE
    base_judge.OLLAMA_KEEP_ALIVE = keep_alive_value
    
    # Import fresh (after patching)
    from evaluation.judging.binary_llm_judge import BinaryLLMJudge
    
    # Stop Ollama to start fresh
    stop_ollama()
    
    # Initialize judge
    judge = BinaryLLMJudge(
        model_name="llama3.2-vision",
        dataset_name="imagenet"
    )
    
    # Find test images
    test_data = []
    for image_id, label, class_name in TEST_IMAGES:
        image_path = find_image_path(image_id)
        if image_path:
            test_data.append((image_id, label, class_name, image_path))
    
    if not test_data:
        logger.error("No test images found!")
        return None
    
    # Track results
    results = defaultdict(list)
    round_accuracies = []
    total_time = 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing keep_alive={keep_alive_value}")
    logger.info(f"{'='*60}")
    
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n--- Round {round_num}/{num_rounds} ---")
        round_correct = 0
        
        for image_id, label, class_name, image_path in test_data:
            start_time = time.time()
            
            try:
                result = judge._predict_single_image(
                    image_data=str(image_path),
                    true_label=label,
                    image_id=f"test-{image_id}",
                    context={"occlusion_level": 0, "fill_strategy": "gray", "method": "saliency"}
                )
                
                if len(result) >= 4:
                    pred_label = result[1]
                    is_correct = (pred_label == label)
                else:
                    is_correct = False
                
                elapsed = time.time() - start_time
                total_time += elapsed
                results[image_id].append(is_correct)
                
                if is_correct:
                    round_correct += 1
                
                status = "✓" if is_correct else "✗"
                logger.info(f"  {status} {class_name}: {'yes' if is_correct else 'no'} ({elapsed:.2f}s)")
                
            except Exception as e:
                logger.error(f"  Error: {e}")
                results[image_id].append(False)
        
        round_acc = round_correct / len(test_data) * 100
        round_accuracies.append(round_acc)
        logger.info(f"  Round {round_num} accuracy: {round_acc:.1f}%")
        
        time.sleep(1)
    
    # Restore original value
    base_judge.OLLAMA_KEEP_ALIVE = original_keep_alive
    
    # Calculate statistics
    consistent_count = 0
    true_count = 0
    total_predictions = 0
    
    for image_id, predictions in results.items():
        total_predictions += len(predictions)
        true_count += sum(predictions)
        if all(p == predictions[0] for p in predictions):
            consistent_count += 1
    
    accuracy = true_count / total_predictions * 100 if total_predictions > 0 else 0
    consistency = consistent_count / len(results) * 100 if results else 0
    avg_time = total_time / total_predictions if total_predictions > 0 else 0
    
    # Check for degradation (accuracy dropping across rounds)
    degradation = round_accuracies[0] - round_accuracies[-1] if len(round_accuracies) >= 2 else 0
    
    return {
        'keep_alive': keep_alive_value,
        'accuracy': accuracy,
        'consistency': consistency,
        'avg_time': avg_time,
        'true_count': true_count,
        'total': total_predictions,
        'consistent_images': consistent_count,
        'total_images': len(results),
        'round_accuracies': round_accuracies,
        'degradation': degradation
    }


def main():
    logger.info("="*70)
    logger.info("KEEP_ALIVE OPTIMIZATION TEST")
    logger.info("="*70)
    logger.info("\nBackground:")
    logger.info("  - Ollama can have 'state degradation' when keep_alive is too long")
    logger.info("  - This causes accuracy to drop over time within a session")
    logger.info("  - GitHub issues: ollama/ollama#4846, ollama/ollama#5272")
    logger.info("="*70)
    
    all_results = []
    
    for keep_alive in KEEP_ALIVE_VALUES:
        try:
            result = run_test_with_keep_alive(keep_alive, ROUNDS_PER_VALUE)
            if result:
                all_results.append(result)
                
                logger.info(f"\n>>> Results for keep_alive={keep_alive}:")
                logger.info(f"    Accuracy: {result['accuracy']:.1f}%")
                logger.info(f"    Consistency: {result['consistency']:.1f}%")
                logger.info(f"    Avg time: {result['avg_time']:.2f}s")
                logger.info(f"    Degradation: {result['degradation']:.1f}% (R1-R{ROUNDS_PER_VALUE})")
                logger.info(f"    Round accuracies: {[f'{a:.0f}%' for a in result['round_accuracies']]}")
                
        except Exception as e:
            logger.error(f"Failed to test keep_alive={keep_alive}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        logger.error("No results collected!")
        return
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"{'Keep Alive':<12} {'Accuracy':<12} {'Consistency':<14} {'Avg Time':<10} {'Degradation':<12}")
    logger.info("-"*60)
    
    for r in all_results:
        logger.info(f"{r['keep_alive']:<12} {r['accuracy']:.1f}%{'':<6} {r['consistency']:.1f}%{'':<8} {r['avg_time']:.2f}s{'':<4} {r['degradation']:.1f}%")
    
    # Analysis
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS")
    logger.info("="*70)
    
    # Find best accuracy
    best_accuracy = max(all_results, key=lambda x: x['accuracy'])
    logger.info(f"Best accuracy: keep_alive={best_accuracy['keep_alive']} ({best_accuracy['accuracy']:.1f}%)")
    
    # Find best consistency
    best_consistency = max(all_results, key=lambda x: x['consistency'])
    logger.info(f"Best consistency: keep_alive={best_consistency['keep_alive']} ({best_consistency['consistency']:.1f}%)")
    
    # Find fastest
    fastest = min(all_results, key=lambda x: x['avg_time'])
    logger.info(f"Fastest: keep_alive={fastest['keep_alive']} ({fastest['avg_time']:.2f}s)")
    
    # Find least degradation
    least_degradation = min(all_results, key=lambda x: x['degradation'])
    logger.info(f"Least degradation: keep_alive={least_degradation['keep_alive']} ({least_degradation['degradation']:.1f}%)")
    
    # Recommendation - balance score
    # Score = (accuracy * consistency) / (time * (1 + degradation/100))
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATION")
    logger.info("="*70)
    
    for r in all_results:
        # Higher is better: accuracy and consistency
        # Lower is better: time and degradation
        score = (r['accuracy'] * r['consistency']) / (r['avg_time'] * (1 + r['degradation']/10 + 0.1))
        r['score'] = score
    
    best = max(all_results, key=lambda x: x['score'])
    logger.info(f"\nBest balance: keep_alive={best['keep_alive']}")
    logger.info(f"  Accuracy: {best['accuracy']:.1f}%")
    logger.info(f"  Consistency: {best['consistency']:.1f}%")
    logger.info(f"  Avg time: {best['avg_time']:.2f}s")
    logger.info(f"  Degradation: {best['degradation']:.1f}%")
    
    # Specific recommendations
    logger.info("\n" + "-"*60)
    logger.info("SPECIFIC RECOMMENDATIONS:")
    logger.info("-"*60)
    
    if best['keep_alive'] == 0:
        logger.info("→ Use keep_alive='0' for maximum accuracy/consistency")
        logger.info("  Tradeoff: ~3x slower, but no state degradation")
        logger.info("  Consider: Periodic 'ollama stop' every N requests as alternative")
    else:
        logger.info(f"→ Use keep_alive='{best['keep_alive']}' for good balance")
        logger.info(f"  If accuracy drops over time, decrease keep_alive or add periodic resets")


if __name__ == '__main__':
    main()
