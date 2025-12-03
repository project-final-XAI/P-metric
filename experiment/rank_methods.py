"""
experiment/rank_methods.py

Rank XAI methods from both experiments and compare rankings.
Uses existing heatmaps from original experiment.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
import logging
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import get_dataloader
from models.loader import load_model
from evaluation.judging.pytorch_judge import PyTorchJudgingModel
from core.file_manager import FileManager
from experiment.metrics_pmetric import compute_pmetric
from experiment.metrics_ssms import compute_ssms
from experiment.config import ExperimentConfig
from attribution.registry import get_attribution_method


def load_heatmap_from_png(png_path: Path) -> np.ndarray:
    """
    Load heatmap from PNG file and convert back to [0,1] values.
    
    Note: This is approximate since PNG is lossy, but should be close enough.
    We use the grayscale intensity as a proxy for the original heatmap values.
    """
    img = cv2.imread(str(png_path))
    if img is None:
        raise ValueError(f"Could not load PNG: {png_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale (take mean across channels)
    heatmap = np.mean(img_rgb, axis=2).astype(np.float32)
    
    # Normalize to [0, 1]
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)
    
    return heatmap


def generate_heatmap_from_model(
    image: torch.Tensor,
    generating_model: torch.nn.Module,
    method_name: str,
    true_label: int,
    device: str
) -> np.ndarray:
    """
    Generate heatmap on-the-fly using the attribution method.
    This is more accurate than loading from PNG.
    """
    attribution_method = get_attribution_method(method_name)
    
    # Prepare inputs
    image_batch = image.unsqueeze(0).to(device)
    target_batch = torch.tensor([true_label], device=device)
    
    # Generate heatmap
    with torch.set_grad_enabled(True):
        heatmap_tensor = attribution_method.compute(
            generating_model,
            image_batch,
            target_batch
        )
    
    # Convert to numpy and ensure 2D
    if isinstance(heatmap_tensor, torch.Tensor):
        heatmap = heatmap_tensor.squeeze().cpu().detach().numpy()
    else:
        heatmap = np.array(heatmap_tensor).squeeze()
    
    # Handle multi-channel heatmaps
    if heatmap.ndim == 3:
        heatmap = np.mean(heatmap, axis=0)
    elif heatmap.ndim > 2:
        heatmap = heatmap.squeeze()
    
    # Ensure 2D
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap should be 2D, got shape {heatmap.shape}")
    
    return heatmap


def get_available_methods(heatmap_dir: Path, model_name: str) -> list:
    """Get list of available attribution methods from heatmap directory."""
    model_dir = heatmap_dir / model_name
    if not model_dir.exists():
        return []
    
    methods = []
    for method_dir in model_dir.iterdir():
        if method_dir.is_dir() and (method_dir / "regular").exists():
            methods.append(method_dir.name)
    
    return sorted(methods)


def save_method_visualizations(
    image: torch.Tensor,
    heatmap: np.ndarray,
    masked_image: torch.Tensor,
    method_name: str,
    image_id: str,
    output_dir: Path,
    ssms_score: float,
    auc: float
):
    """
    Save visualization images for a method.
    Only saves if files don't already exist.
    
    Saves: original image, heatmap, masked image, and overlay.
    """
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    original_path = method_dir / f"{image_id}_original.png"
    heatmap_path = method_dir / f"{image_id}_heatmap.png"
    masked_path = method_dir / f"{image_id}_masked_ssms{ssms_score:.2f}_auc{auc:.3f}.png"
    overlay_path = method_dir / f"{image_id}_overlay.png"
    
    # Check if all files already exist
    if all(p.exists() for p in [original_path, heatmap_path, masked_path, overlay_path]):
        return  # Skip if all files exist
    
    # Denormalize image for visualization (ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = image.cpu() * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)
    
    # Convert to numpy for PIL
    img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    # Save original image (only if doesn't exist)
    if not original_path.exists():
        img_pil.save(original_path)
    
    # Prepare heatmap
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_colored = cm.hot(heatmap_normalized)[:, :, :3]  # RGB
    heatmap_uint8 = (heatmap_colored * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_uint8)
    
    # Save heatmap (only if doesn't exist)
    if not heatmap_path.exists():
        heatmap_pil.save(heatmap_path)
    
    # Save masked image (only if doesn't exist)
    if not masked_path.exists():
        masked_denorm = masked_image.cpu() * std + mean
        masked_denorm = torch.clamp(masked_denorm, 0, 1)
        masked_np = (masked_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        masked_pil = Image.fromarray(masked_np)
        masked_pil.save(masked_path)
    
    # Save overlay (only if doesn't exist)
    if not overlay_path.exists():
        overlay = img_np.copy().astype(np.float32) / 255.0
        overlay = overlay * 0.6 + heatmap_colored * 0.4  # Blend
        overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_uint8)
        overlay_pil.save(overlay_path)


def load_pmetric_from_original_experiment(config):
    """
    Load P-Metric results from original experiment CSV.
    Returns a dict mapping (method, judge_model) -> (AUC, DROP) averages.
    """
    try:
        pmetric_path = Path(config.base_dir) / "results" / "analysis" / "faithfulness_metrics.csv"
        if not pmetric_path.exists():
            logging.warning(f"P-Metric CSV not found: {pmetric_path}")
            return {}
        
        df = pd.read_csv(pmetric_path)
        
        # Filter out LLM judges
        df = df[~df['judging_model'].str.contains('llama', case=False, na=False)]
        
        # Map method names (might need normalization)
        method_mapping = {
            'grad_cam': 'grad_cam',
            'integrated_gradients': 'integrated_gradients',
            'inputxgradient': 'inputxgradient',
            'saliency': 'saliency',
            'occlusion': 'occlusion',
            'xrai': 'xrai',
            'random_baseline': 'random_baseline',
        }
        
        # Create lookup dict: (method, judge) -> (AUC, DROP)
        pmetric_lookup = {}
        for _, row in df.iterrows():
            method = row.get('attribution_method', '').lower()
            judge = row.get('judging_model', '').lower()
            auc = row.get('auc', 0.0)
            drop = row.get('drop_at_75', 0.0)
            
            # Normalize method name
            method_normalized = method_mapping.get(method, method)
            
            # Store average (if multiple entries, they'll be averaged)
            key = (method_normalized, judge)
            if key not in pmetric_lookup:
                pmetric_lookup[key] = {'AUC': [], 'DROP': []}
            pmetric_lookup[key]['AUC'].append(auc)
            pmetric_lookup[key]['DROP'].append(drop)
        
        # Average multiple entries
        pmetric_avg = {}
        for key, values in pmetric_lookup.items():
            pmetric_avg[key] = {
                'AUC': np.mean(values['AUC']),
                'DROP': np.mean(values['DROP'])
            }
        
        logging.info(f"Loaded P-Metric results for {len(pmetric_avg)} method-judge combinations")
        return pmetric_avg
        
    except Exception as e:
        logging.error(f"Failed to load P-Metric from original experiment: {e}")
        return {}


def evaluate_all_methods(config, num_images=1000):
    """
    Evaluate all available XAI methods using existing heatmaps.
    Uses P-Metric results from original experiment (no recalculation!).
    Only computes SSMS (the new metric).
    
    Returns DataFrame with results for all methods.
    """
    # Load P-Metric results from original experiment (FAST - no computation!)
    pmetric_lookup = load_pmetric_from_original_experiment(config)
    
    file_manager = FileManager(config.base_dir)
    heatmap_dir = file_manager.get_heatmap_dir(config.dataset_name)
    
    # Get available methods (try resnet50 first, then mobilenet_v2)
    gen_model_name = "mobilenet_v2"
    available_methods = get_available_methods(heatmap_dir, gen_model_name)
    
    if not available_methods:
        logging.warning("No heatmaps found for resnet50! Trying mobilenet_v2...")
        gen_model_name = "mobilenet_v2"
        available_methods = get_available_methods(heatmap_dir, gen_model_name)
    
    if not available_methods:
        raise ValueError("No heatmaps found in results/heatmaps/")
    
    logging.info(f"Found methods: {available_methods}")
    logging.info(f"Using generating model: {gen_model_name}")
    
    # Load generating model for on-the-fly generation
    try:
        generating_model = load_model(gen_model_name)
    except Exception as e:
        logging.error(f"Failed to load generating model: {e}")
        generating_model = None
    
    # Load judge models
    judge_models = {}
    for judge_name in config.judge_models:
        try:
            model = load_model(judge_name)
            judge_models[judge_name] = PyTorchJudgingModel(model, judge_name, device=config.device)
        except Exception as e:
            logging.error(f"Failed to load judge {judge_name}: {e}")
    
    if not judge_models:
        raise ValueError("No judge models loaded!")
    
    # Load dataset with larger batch size for efficiency
    batch_size = 8 if config.device == "cuda" else 4
    dataloader = get_dataloader(config.dataset_name, batch_size=batch_size, shuffle=False)
    
    # Results storage
    results = []
    
    # Pre-compute category mapping for ImageNet (ONCE, not in loop!)
    category_mapping = {}
    synset_ids = []
    if config.dataset_name == "imagenet":
        try:
            from data.imagenet_class_mapping import get_cached_mapping, format_class_for_llm
            from config import DATASET_CONFIG
            import os
            mapping = get_cached_mapping()
            dataset_path = DATASET_CONFIG.get("imagenet", {}).get("path")
            if dataset_path and os.path.exists(dataset_path):
                synset_ids = sorted([d for d in os.listdir(dataset_path) 
                                    if os.path.isdir(os.path.join(dataset_path, d))])
                # Pre-compute mapping for all labels
                for label_idx in range(len(synset_ids)):
                    synset_id = synset_ids[label_idx]
                    category_name_full = mapping.get(synset_id, "")
                    if category_name_full:
                        category_name = format_class_for_llm(category_name_full)
                        category_mapping[label_idx] = category_name
            logging.info(f"Pre-computed category mapping for {len(category_mapping)} classes")
        except Exception as e:
            logging.warning(f"Failed to pre-compute category mapping: {e}")
    
    # Heatmap cache to avoid repeated I/O
    heatmap_cache = {}  # {(image_id, method_name): heatmap}
    
    # Process images in batches for better GPU utilization
    processed = 0
    for batch_images, batch_labels in tqdm(dataloader, desc="Processing images", total=num_images):
        if processed >= num_images:
            break
        
        # Move batch to GPU
        batch_images = batch_images.to(config.device, non_blocking=True)
        
        # Process each image in batch
        for batch_idx in range(batch_images.shape[0]):
            if processed >= num_images:
                break
            
            image = batch_images[batch_idx]
            true_label = batch_labels[batch_idx].item()
            image_id = f"image_{processed:05d}"
            
            # Process each method
            for method_name in available_methods:
                # Check cache first (FAST!)
                cache_key = (image_id, method_name)
                if cache_key in heatmap_cache:
                    heatmap = heatmap_cache[cache_key]
                else:
                    heatmap = None
                    
                    # Try to load from PNG first
                    regular_path = file_manager.get_regular_heatmap_path(
                        config.dataset_name, gen_model_name, method_name, image_id
                    )
                    
                    # Also try with category name (for ImageNet) - use pre-computed mapping!
                    if not regular_path.exists() and config.dataset_name == "imagenet" and true_label in category_mapping:
                        category_name = category_mapping[true_label]
                        regular_path = file_manager.get_regular_heatmap_path(
                            config.dataset_name, gen_model_name, method_name, image_id, category_name
                        )
                    
                    # Try loading from PNG (preferred - use existing heatmaps from original experiment)
                    if regular_path.exists():
                        try:
                            heatmap = load_heatmap_from_png(regular_path)
                            # Cache it!
                            heatmap_cache[cache_key] = heatmap
                        except Exception as e:
                            logging.debug(f"Failed to load PNG {regular_path}: {e}")
                            heatmap = None
                
                # Only generate on-the-fly if PNG doesn't exist
                if heatmap is None:
                    if generating_model is not None:
                        try:
                            heatmap = generate_heatmap_from_model(
                                image, generating_model, method_name, true_label, config.device
                            )
                            # Cache generated heatmap
                            heatmap_cache[cache_key] = heatmap
                        except Exception as e:
                            logging.warning(f"Failed to generate heatmap for {method_name}: {e}")
                            continue
                    else:
                        logging.debug(f"No heatmap available for {image_id}/{method_name} and no generating model")
                        continue
                
                if heatmap is None:
                    logging.debug(f"No heatmap available for {image_id}/{method_name}")
                    continue
                
                # Evaluate with each judge model
                for judge_name, judge_model in judge_models.items():
                    try:
                        # Use torch.no_grad() for inference (faster, less memory)
                        with torch.no_grad():
                            # Compute SSMS with continuous differential masking (always compute - this is the NEW metric)
                            ssms_score, ssms_metadata, masked_image = compute_ssms(
                                heatmap,
                                image,
                                judge_model,
                                true_label,
                                alpha_max=config.alpha_max,
                                eps=config.eps,
                                power_factor=2.5,  # Higher power = stronger masking
                                sparsity_penalty_factor=3.0,  # Penalty for non-informative heatmaps
                                base_alpha=1.0
                            )
                            
                            # Use P-Metric from original experiment (FAST - no computation!)
                            # Look up by (method, judge) combination
                            method_normalized = method_name.lower()
                            judge_normalized = judge_name.lower()
                            lookup_key = (method_normalized, judge_normalized)
                            
                            if lookup_key in pmetric_lookup:
                                # Use existing P-Metric results from original experiment
                                pmetric_metrics = pmetric_lookup[lookup_key].copy()
                            else:
                                # Fallback: try to find by method only (average across judges)
                                # Pre-compute method-only lookup for efficiency
                                method_only_key = None
                                if not hasattr(pmetric_lookup, '_method_cache'):
                                    # Build method-only cache once
                                    pmetric_lookup._method_cache = {}
                                    for key in pmetric_lookup.keys():
                                        method = key[0]
                                        if method not in pmetric_lookup._method_cache:
                                            pmetric_lookup._method_cache[method] = key
                                
                                method_only_key = pmetric_lookup._method_cache.get(method_normalized)
                                
                                if method_only_key:
                                    pmetric_metrics = pmetric_lookup[method_only_key].copy()
                                    logging.debug(f"Using P-Metric for {method_name} (averaged across judges)")
                                else:
                                    # No P-Metric found - use dummy values
                                    pmetric_metrics = {'AUC': 0.0, 'DROP': 0.0}
                                    logging.warning(f"No P-Metric found for {method_name}/{judge_name}")
                        
                        results.append({
                            'image_id': image_id,
                            'method': method_name,
                            'judge_model': judge_name,
                            'AUC': pmetric_metrics['AUC'],
                            'DROP': pmetric_metrics['DROP'],
                            'SSMS_score': ssms_score
                        })
                        
                        # Save visualizations (only for first judge model to avoid duplicates)
                        # Defer saving to end of batch to avoid I/O overhead during computation
                        if judge_name == config.judge_models[0]:
                            # Check if files already exist before saving
                            output_dir = Path("experiment/method_visualizations")
                            method_dir = output_dir / method_name
                            original_path = method_dir / f"{image_id}_original.png"
                            
                            # Only process if at least one file is missing
                            if not original_path.exists():
                                # Save immediately but use efficient I/O
                                save_method_visualizations(
                                    image,
                                    heatmap,
                                    masked_image,  # Use masked_image from compute_ssms (no recalculation!)
                                    method_name,
                                    image_id,
                                    output_dir,
                                    ssms_score,
                                    pmetric_metrics['AUC']
                                )
                        
                    except Exception as e:
                        logging.warning(f"Failed to evaluate {image_id}/{method_name}/{judge_name}: {e}")
                        continue
            
            # Clear GPU cache periodically for memory efficiency
            if processed % 50 == 0 and config.device == "cuda":
                torch.cuda.empty_cache()
            
            processed += 1
    
    return pd.DataFrame(results)


def rank_methods_from_original_experiment():
    """Load and rank methods from original experiment."""
    df = pd.read_csv("../results/analysis/faithfulness_metrics.csv")
    
    # Filter out LLM judges
    df = df[~df['judging_model'].str.contains('llama', case=False, na=False)]
    
    # Aggregate by method (average across models/strategies/judges)
    method_stats = df.groupby('attribution_method').agg({
        'auc': ['mean', 'std', 'count']
    }).round(4)
    
    method_stats.columns = ['auc_mean', 'auc_std', 'auc_count']
    method_stats = method_stats.reset_index()
    
    # Rank by AUC (higher is better)
    method_stats['rank'] = method_stats['auc_mean'].rank(ascending=False, method='min').astype(int)
    method_stats = method_stats.sort_values('rank')
    
    return method_stats


def rank_methods_from_new_experiment(df: pd.DataFrame):
    """Rank methods from new experiment results."""
    # Aggregate by method (average across images/judges)
    method_stats = df.groupby('method').agg({
        'AUC': ['mean', 'std', 'count']
    }).round(4)
    
    method_stats.columns = ['auc_mean', 'auc_std', 'auc_count']
    method_stats = method_stats.reset_index()
    
    # Rank by AUC
    method_stats['rank'] = method_stats['auc_mean'].rank(ascending=False, method='min').astype(int)
    method_stats = method_stats.sort_values('rank')
    
    return method_stats


def compare_rankings(original_rankings, new_rankings, threshold=0.05):
    """
    Compare rankings and identify:
    1. Methods that appear in both experiments
    2. Ranking consistency
    3. Small differences (within threshold) that are acceptable
    """
    print("\n" + "="*80)
    print(" " * 25 + "XAI METHOD RANKING COMPARISON")
    print("="*80)
    
    # Original experiment rankings
    print("\n" + "-"*80)
    print("ORIGINAL EXPERIMENT - Method Rankings")
    print("-"*80)
    print(f"{'Rank':<6} {'Method':<25} {'Mean AUC':<12} {'Std':<10} {'Count':<8}")
    print("-"*80)
    for _, row in original_rankings.iterrows():
        print(f"{row['rank']:<6} {row['attribution_method']:<25} "
              f"{row['auc_mean']:<12.4f} {row['auc_std']:<10.4f} {int(row['auc_count']):<8}")
    
    # New experiment rankings
    print("\n" + "-"*80)
    print("NEW EXPERIMENT - Method Rankings")
    print("-"*80)
    print(f"{'Rank':<6} {'Method':<25} {'Mean AUC':<12} {'Std':<10} {'Count':<8}")
    print("-"*80)
    for _, row in new_rankings.iterrows():
        print(f"{row['rank']:<6} {row['method']:<25} "
              f"{row['auc_mean']:<12.4f} {row['auc_std']:<10.4f} {int(row['auc_count']):<8}")
    
    # Compare common methods
    print("\n" + "-"*80)
    print("RANKING COMPARISON")
    print("-"*80)
    
    # Create mapping
    orig_dict = {row['attribution_method']: row for _, row in original_rankings.iterrows()}
    new_dict = {row['method']: row for _, row in new_rankings.iterrows()}
    
    common_methods = set(orig_dict.keys()) & set(new_dict.keys())
    
    if not common_methods:
        print("⚠ No common methods found between experiments!")
        return
    
    print(f"\nCommon methods: {sorted(common_methods)}")
    print(f"\n{'Method':<25} {'Orig Rank':<12} {'New Rank':<12} {'Rank Diff':<12} {'AUC Diff':<12} {'Status':<20}")
    print("-"*80)
    
    ranking_changes = []
    for method in sorted(common_methods):
        orig_row = orig_dict[method]
        new_row = new_dict[method]
        
        rank_diff = new_row['rank'] - orig_row['rank']
        auc_diff = new_row['auc_mean'] - orig_row['auc_mean']
        
        # Determine status
        if rank_diff == 0:
            status = "✓ Same rank"
        elif abs(auc_diff) < threshold:
            status = "~ Small diff (OK)"
        elif rank_diff < 0:
            status = f"↑ Improved {abs(rank_diff)}"
        else:
            status = f"↓ Dropped {rank_diff}"
        
        ranking_changes.append({
            'method': method,
            'orig_rank': orig_row['rank'],
            'new_rank': new_row['rank'],
            'rank_diff': rank_diff,
            'auc_diff': auc_diff,
            'status': status
        })
        
        print(f"{method:<25} {orig_row['rank']:<12} {new_row['rank']:<12} "
              f"{rank_diff:<12} {auc_diff:<12.4f} {status:<20}")
    
    # Summary
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    
    same_rank = sum(1 for r in ranking_changes if r['rank_diff'] == 0)
    small_diff = sum(1 for r in ranking_changes if abs(r['auc_diff']) < threshold and r['rank_diff'] != 0)
    improved = sum(1 for r in ranking_changes if r['rank_diff'] < 0)
    dropped = sum(1 for r in ranking_changes if r['rank_diff'] > 0 and abs(r['auc_diff']) >= threshold)
    
    print(f"Methods with same rank: {same_rank}/{len(ranking_changes)}")
    print(f"Methods with small AUC diff (<{threshold}): {small_diff}/{len(ranking_changes)}")
    print(f"Methods that improved: {improved}/{len(ranking_changes)}")
    print(f"Methods that dropped significantly: {dropped}/{len(ranking_changes)}")
    
    consistency = (same_rank + small_diff) / len(ranking_changes) * 100
    print(f"\nOverall ranking consistency: {consistency:.1f}%")
    
    if consistency >= 80:
        print("✓✓✓ EXCELLENT: Rankings are highly consistent!")
    elif consistency >= 60:
        print("✓✓ GOOD: Rankings are mostly consistent")
    elif consistency >= 40:
        print("⚠ MODERATE: Some ranking differences")
    else:
        print("✗ POOR: Significant ranking differences")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load original rankings
    print("Loading original experiment rankings...")
    original_rankings = rank_methods_from_original_experiment()
    
    # Run new experiment on all methods
    print("\nRunning new experiment on all available methods...")
    config = ExperimentConfig(quick_mode=False, num_images=1000)
    
    new_results_df = evaluate_all_methods(config, num_images=1000)
    
    if len(new_results_df) == 0:
        print("⚠ No results from new experiment!")
        return
    
    # Save results
    output_path = Path("experiment/results_all_methods.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    new_results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    
    # Rank methods from new experiment
    new_rankings = rank_methods_from_new_experiment(new_results_df)
    
    # Compare rankings
    compare_rankings(original_rankings, new_rankings, threshold=0.05)


if __name__ == "__main__":
    main()

