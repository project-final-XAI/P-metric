"""
Single file for visualizing LLM judging process.

Shows the complete flow: image -> heatmap -> occluded image -> prompt -> LLM response.

Usage:
    python test/test_llm_visualization.py
    python test/test_llm_visualization.py --image-idx 5 --occlusion 70
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import get_dataloader, get_default_transforms
from models.loader import load_model
from attribution.registry import get_attribution_method
from evaluation.occlusion import sort_pixels, apply_occlusion
from evaluation.judging.binary_llm_judge import BinaryLLMJudge
from evaluation.judging.base_llm_judge import tensor_to_pil_image
from config import DEVICE, DATASET_NAME, DATASET_CONFIG
from data.imagenet_class_mapping import get_cached_mapping, format_class_for_llm
import ollama


def tensor_to_displayable(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor to displayable numpy array."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    
    img = tensor.cpu().detach().numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img


def print_step_header(step_num: int, step_name: str):
    """Print a formatted step header."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'='*70}")


def print_success(message: str):
    """Print a success message."""
    print(f"  [OK] {message}")


def print_info(message: str):
    """Print an info message."""
    print(f"  [INFO] {message}")


def visualize_llm_process(
    image_tensor: torch.Tensor,
    true_label: int,
    occlusion_level: int = 50,
    fill_strategy: str = "mean",
    attribution_method: str = "grad_cam",
    model_name: str = "resnet50",
    llm_model: str = "llama3.2-vision-binary",
    save_path: str = None
):
    """
    Visualize the complete LLM judging process.
    
    Args:
        image_tensor: Preprocessed image tensor (C, H, W)
        true_label: True class label
        occlusion_level: Percentage of pixels to occlude
        fill_strategy: Fill strategy for occlusion
        attribution_method: XAI method to use
        model_name: Model name for generating heatmap
        llm_model: LLM judge model name
        save_path: Path to save visualization (optional)
    """
    print("\n" + "="*70)
    print(" " * 15 + "LLM PROCESS VISUALIZATION")
    print("="*70)
    
    # Step 1: Generate heatmap
    print_step_header(1, f"GENERATING HEATMAP ({attribution_method.upper()})")
    print_info(f"Loading model: {model_name}")
    model = load_model(model_name)
    model.eval()
    
    attribution_method_obj = get_attribution_method(attribution_method)
    target_tensor = torch.tensor([true_label], device=DEVICE)
    
    print_info(f"Computing attribution map using {attribution_method}...")
    with torch.no_grad():
        heatmap_tensor = attribution_method_obj.compute(
            model, 
            image_tensor.unsqueeze(0), 
            target_tensor
        )
    
    if heatmap_tensor is None:
        raise ValueError(f"Failed to generate heatmap with {attribution_method}")
    
    heatmap = heatmap_tensor.squeeze().cpu().detach().numpy()
    if heatmap.ndim == 3:
        heatmap = np.mean(heatmap, axis=0)
    
    print_success(f"Heatmap generated successfully (shape: {heatmap.shape})")
    
    # Step 2: Create occluded image
    print_step_header(2, f"CREATING OCCLUDED IMAGE")
    print_info(f"Occlusion level: {occlusion_level}%")
    print_info(f"Fill strategy: {fill_strategy}")
    
    sorted_indices = sort_pixels(heatmap)
    occluded_tensor = apply_occlusion(
        image_tensor,
        sorted_indices,
        occlusion_level,
        fill_strategy,
        image_shape=(224, 224)
    )
    occluded_image_pil = tensor_to_pil_image(occluded_tensor)
    print_success("Occluded image created successfully")
    
    # Step 3: Get class name
    print_step_header(3, "GETTING CLASS INFORMATION")
    dataset_name = DATASET_NAME
    if dataset_name == "imagenet":
        dataset_path = DATASET_CONFIG["imagenet"]["path"]
        from torchvision.datasets import ImageFolder
        temp_dataset = ImageFolder(root=str(dataset_path))
        class_names = temp_dataset.classes
        class_name = class_names[true_label]
        
        # Get readable name
        mapping = get_cached_mapping()
        readable_name = mapping.get(class_name, class_name)
        formatted_name = format_class_for_llm(readable_name)
    else:
        # For other datasets
        dataset_path = DATASET_CONFIG[dataset_name]["path"]
        class_names = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        class_name = class_names[true_label].name
        formatted_name = class_name.replace('_', ' ')
    
    print_info(f"True label: {true_label}")
    print_info(f"Class name: {formatted_name}")
    print_success("Class information retrieved")
    
    # Step 4: Create prompt
    print_step_header(4, "CREATING PROMPT FOR LLM")
    prompt = (
        f"Look at this image carefully. Do you see a {formatted_name} in this image? "
        "Answer with only 'yes' or 'no'."
    )
    print_info("Prompt created:")
    print(f"  \"{prompt}\"")
    print_success("Prompt ready for LLM")
    
    # Step 5: Call LLM
    print_step_header(5, f"CALLING LLM ({llm_model.upper()})")
    print_info(f"Initializing LLM judge: {llm_model}")
    llm_judge = BinaryLLMJudge(llm_model, dataset_name=dataset_name)
    
    # Convert occluded tensor to list format expected by judge
    occluded_list = [occluded_tensor]
    true_labels_list = [true_label]
    
    print_info("Sending image and prompt to LLM...")
    # Get prediction
    predictions = llm_judge.predict(occluded_list, true_labels=true_labels_list)
    predicted_label = predictions[0]
    
    # Get actual response by calling ollama directly
    temp_image_path = llm_judge._tensor_to_temp_file(occluded_tensor)
    try:
        response = ollama.chat(
            model=llm_judge.ollama_model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [temp_image_path]
                }
            ],
            options={'temperature': 0.0}
        )
        llm_response = response.message.content.strip()
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    
    print_info(f"LLM Response: \"{llm_response}\"")
    print_info(f"Predicted Label: {predicted_label}")
    
    is_correct = predicted_label == true_label
    status = "CORRECT" if is_correct else "INCORRECT"
    print_success(f"Prediction: {status}")
    
    # Step 6: Create visualization
    print_step_header(6, "CREATING VISUALIZATION")
    print_info("Generating visualization figure...")
    
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2,
                          left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    original_img = tensor_to_displayable(image_tensor)
    ax1.imshow(original_img)
    ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    ax2.imshow(heatmap_normalized, cmap='hot')
    ax2.set_title(f"Heatmap ({attribution_method})", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Occluded image
    ax3 = fig.add_subplot(gs[0, 2])
    occluded_img = tensor_to_displayable(occluded_tensor)
    ax3.imshow(occluded_img)
    ax3.set_title(f"Occluded Image\n({occlusion_level}% occluded, {fill_strategy})", 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Prompt display
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    ax4.text(0.05, 0.9, "Prompt sent to LLM:", fontsize=14, fontweight='bold',
             transform=ax4.transAxes, verticalalignment='top')
    ax4.text(0.05, 0.7, f'"{prompt}"', fontsize=11, style='italic',
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    ax4.text(0.05, 0.4, "LLM Response:", fontsize=14, fontweight='bold',
             transform=ax4.transAxes, verticalalignment='top')
    response_color = '#2ecc71' if is_correct else '#e74c3c'
    ax4.text(0.05, 0.2, f'"{llm_response}"', fontsize=12,
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=response_color, alpha=0.3))
    
    status_symbol = "[CORRECT]" if is_correct else "[INCORRECT]"
    ax4.text(0.05, 0.05, f"Prediction: Label {predicted_label} {status_symbol}",
             fontsize=12, fontweight='bold', color=response_color,
             transform=ax4.transAxes, verticalalignment='top')
    
    # Main title
    fig.suptitle(
        f"LLM Judging Process Visualization\n"
        f"Model: {model_name} | LLM: {llm_model} | True Class: {formatted_name}",
        fontsize=14, fontweight='bold', y=0.98
    )
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print_success(f"Visualization saved to: {save_path}")
    
    # Try to show plot, but don't fail if GUI backend is not available
    try:
        import matplotlib
        backend = matplotlib.get_backend()
        if backend.lower() not in ['agg', 'pdf', 'svg', 'ps']:
            plt.show(block=False)
            print_success("Visualization displayed")
        else:
            print_info("Non-interactive backend detected, file saved only")
    except Exception as e:
        print_info(f"GUI display skipped: {e}")
    finally:
        plt.close(fig)
    
    print("\n" + "="*70)
    print(" " * 25 + "VISUALIZATION COMPLETE")
    print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize LLM judging process',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/test_llm_visualization.py
  python test/test_llm_visualization.py --image-idx 5
  python test/test_llm_visualization.py --occlusion 70 --fill-strategy gray
  python test/test_llm_visualization.py --attribution inputxgradient --model mobilenet_v2
        """
    )
    parser.add_argument('--image-idx', type=int, default=0,
                       help='Image index from dataset (default: 0)')
    parser.add_argument('--occlusion', type=int, default=50,
                       help='Occlusion level percentage (default: 50)')
    parser.add_argument('--fill-strategy', type=str, default='mean',
                       choices=['mean', 'gray', 'black', 'white', 'blur', 'random_noise'],
                       help='Fill strategy (default: mean)')
    parser.add_argument('--attribution', type=str, default='grad_cam',
                       help='Attribution method (default: grad_cam)')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model for heatmap generation (default: resnet50)')
    parser.add_argument('--llm', type=str, default='llama3.2-vision-binary',
                       help='LLM judge model (default: llama3.2-vision-binary)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization (optional)')
    
    args = parser.parse_args()
    
    # Load dataset
    print("\n" + "="*70)
    print(" " * 25 + "LOADING DATASET")
    print("="*70)
    print_info(f"Dataset: {DATASET_NAME}")
    
    from config import DATASET_NAME
    dataloader = get_dataloader(DATASET_NAME, batch_size=1, shuffle=False)
    
    # Get specific image
    image_idx = args.image_idx
    for i, (img, lbl) in enumerate(dataloader):
        if i == image_idx:
            image_tensor = img.squeeze(0).to(DEVICE)
            true_label = lbl.item()
            break
    else:
        raise ValueError(f"Image index {image_idx} not found in dataset")
    
    print_success(f"Loaded image {image_idx}, true label: {true_label}")
    
    # Run visualization
    save_path = args.save or f"results/llm_process_test_{image_idx}.png"
    
    visualize_llm_process(
        image_tensor=image_tensor,
        true_label=true_label,
        occlusion_level=args.occlusion,
        fill_strategy=args.fill_strategy,
        attribution_method=args.attribution,
        model_name=args.model,
        llm_model=args.llm,
        save_path=save_path
    )


if __name__ == "__main__":
    main()

