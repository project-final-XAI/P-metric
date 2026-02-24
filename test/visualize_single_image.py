"""
Simple XAI Visualization Tool - Single Image Analysis

This tool allows you to visualize how different attribution methods explain
a model's prediction on a single image. Shows all methods in one window.

Edit the configuration variables below to analyze your own images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2

# Import project modules
from data.loader import get_default_transforms
from models.loader import load_model
from attribution.registry import get_attribution_method
from evaluation.occlusion import sort_pixels, apply_occlusion
from config import DEVICE, ATTRIBUTION_METHODS

# ===========================
# USER CONFIGURATION
# Edit these variables to analyze different images
# ===========================

# Path to your image (can be absolute or relative)
IMAGE_PATH = "../data/imagenet/n01491361/n01491361_tiger_shark.JPEG"
# IMAGE_PATH = "../data/SIPaKMeD/Dyskeratotic/002.jpeg"

# Model to use for analysis
MODEL_NAME = "mobilenet_v2"
# MODEL_NAME = "sipakmed_resnet50.pth"

# Dataset name for proper class mapping
DATASET_NAME = "imagenet" # "SIPaKMeD"
# DATASET_NAME = "SIPaKMeD"

# Occlusion fill strategy
FILL_STRATEGY = "black"  # Options: "gray", "blur", "black", "white", "random_noise", "mean"

# Occlusion level to display (percentage)
OCCLUSION_LEVEL = 20

# True class label (set to None to auto-predict using model)
TRUE_CLASS = None

# Colormap for heatmaps
# Options: "jet" (blue-cyan-yellow-red), "hot" (black-red-yellow-white), 
#          "viridis" (purple-blue-green-yellow), "rainbow" (rainbow colors),
#          "turbo" (vibrant multi-color)
COLORMAP = "hot"

# ===========================
# HELPER FUNCTIONS
# ===========================

def load_and_preprocess_image(image_path: str) -> tuple[torch.Tensor, Image.Image]:
    """
    Load image from path and preprocess for model input.
    
    Returns:
        tuple: (preprocessed_tensor, original_pil_image)
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load original image
    original_image = Image.open(image_path).convert("RGB")
    
    # Apply transforms
    transform = get_default_transforms()
    image_tensor = transform(original_image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    return image_tensor, original_image


def get_true_class(image_tensor: torch.Tensor, model: torch.nn.Module, manual_class: int = None) -> int:
    """
    Get the true class for the image.
    
    Args:
        image_tensor: Preprocessed image tensor
        model: Model to use for prediction
        manual_class: Manual class override (if provided)
    
    Returns:
        Class index
    """
    if manual_class is not None:
        return manual_class
    
    # Auto-predict using the model
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, dict):
            output = output['logits']
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class


def tensor_to_displayable(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor to displayable numpy array (RGB, 0-255).
    
    Reverses ImageNet normalization and converts to uint8.
    """
    # Remove batch dimension if present
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    
    # Move to CPU and convert to numpy
    img = tensor.cpu().detach().numpy()
    
    # Denormalize (reverse ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img * std + mean
    
    # Clip to valid range and convert to uint8
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    # Convert from CHW to HWC
    img = np.transpose(img, (1, 2, 0))
    
    return img


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize heatmap to 0-1 range."""
    hmap = heatmap.copy()
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    return hmap


def heatmap_to_colormap(heatmap: np.ndarray, target_size: tuple = None, colormap: str = "jet") -> np.ndarray:
    """
    Convert heatmap to colored visualization.
    
    Args:
        heatmap: 2D heatmap array
        target_size: (width, height) to resize to, or None to keep original
        colormap: Name of colormap to use (jet, hot, viridis, rainbow, turbo)
    
    Returns:
        RGB image as numpy array
    """
    # Normalize
    heatmap_norm = normalize_heatmap(heatmap)
    
    # Resize if needed
    if target_size is not None and heatmap_norm.shape[:2] != (target_size[1], target_size[0]):
        heatmap_norm = cv2.resize(heatmap_norm, target_size)
    
    # Convert to uint8 for better color mapping
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    
    # Map colormap names to OpenCV constants
    colormap_dict = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "turbo": cv2.COLORMAP_TURBO,
        "cool": cv2.COLORMAP_COOL,
        "summer": cv2.COLORMAP_SUMMER,
        "autumn": cv2.COLORMAP_AUTUMN,
    }
    
    # Get colormap or default to JET
    cv_colormap = colormap_dict.get(colormap.lower(), cv2.COLORMAP_JET)
    
    # Apply colormap using OpenCV for vibrant colors
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv_colormap)
    
    # Convert BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    return heatmap_colored


# ===========================
# MAIN VISUALIZATION
# ===========================

def visualize_all_methods(
    image_tensor: torch.Tensor,
    original_image: Image.Image,
    model: torch.nn.Module,
    model_name: str,
    true_class: int,
    methods: list[str],
    fill_strategy: str,
    occlusion_level: int,
    colormap: str = "jet",
    image_path: str = None
):
    """
    Create a single visualization showing all attribution methods.
    
    Layout:
    - Left side: Original image (spans all rows)
    - Right side: Grid of methods (2-3 columns based on count)
    - Each method: Heatmap + Occluded image side by side
    """
    print(f"\n🎨 Generating visualizations for all methods...")
    
    # Compute results for all methods
    method_results = []
    original_np = tensor_to_displayable(image_tensor)
    
    for method_name in methods:
        print(f"  Processing: {method_name}")
        
        try:
            # Get attribution method
            attribution_method = get_attribution_method(method_name)
            
            # Compute heatmap
            target_tensor = torch.tensor([true_class], device=DEVICE)
            heatmap_tensor = attribution_method(model, image_tensor, target_tensor)
            
            if heatmap_tensor is None:
                print(f"    ⚠️  Skipping {method_name}: failed to compute heatmap")
                continue
            
            # Convert to numpy
            heatmap = heatmap_tensor.squeeze().cpu().detach().numpy()
            if heatmap.ndim == 3:
                heatmap = np.mean(heatmap, axis=0)
            
            # Sort pixels and create occluded image
            sorted_indices = sort_pixels(heatmap)
            occluded_tensor = apply_occlusion(
                image_tensor.squeeze(0),
                sorted_indices,
                occlusion_level,
                fill_strategy,
                image_shape=(224, 224)
            )
            occluded_np = tensor_to_displayable(occluded_tensor)
            
            # Convert heatmap to colormap
            heatmap_colored = heatmap_to_colormap(heatmap, target_size=(224, 224), colormap=colormap)
            
            method_results.append({
                'name': method_name,
                'heatmap': heatmap_colored,
                'occluded': occluded_np
            })
            
            print(f"    ✓ Done")
            
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            continue
    
    if not method_results:
        print("  ❌ No results to display!")
        return
    
    # Create visualization
    print(f"\n📊 Creating final visualization with {len(method_results)} methods...")
    
    # Calculate dynamic grid dimensions
    num_methods = len(method_results)
    # Aim for roughly square grid: 2 cols for ≤6 methods, 3 cols for >6
    grid_cols = 2 if num_methods <= 6 else 3
    grid_rows = (num_methods + grid_cols - 1) // grid_cols  # Ceiling division
    
    # Extract filename from image path
    filename = Path(image_path).name if image_path else "Unknown"
    
    # Create figure with presentation-quality styling
    # Layout: Original image on left, grid of methods on right
    fig_width = 18 if grid_cols == 2 else 22
    fig_height = max(10, grid_rows * 4)
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='#ffffff')
    
    # Create grid spec: 1 column for original image + grid_cols * 2 for methods
    # Each method needs 2 sub-columns (heatmap + occluded)
    total_cols = 3 + grid_cols * 2  # 3 for original + spacing, rest for methods grid
    gs = fig.add_gridspec(grid_rows, total_cols, 
                          hspace=0.4, wspace=0.15,
                          left=0.02, right=0.98, top=0.90, bottom=0.03)
    
    # Left side: Original image (spans all rows, uses first 2 columns)
    ax_original = fig.add_subplot(gs[:, :2])
    ax_original.imshow(original_np)
    ax_original.set_title(
        f"Original Image\n{filename}", 
        fontsize=14, fontweight='bold', pad=20,
        color='#1a1a1a', family='sans-serif'
    )
    ax_original.axis('off')
    # Add elegant border
    for spine in ax_original.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#3498db')
        spine.set_linewidth(3)
    
    # Right side: Grid of methods
    # Each method occupies a grid cell with (heatmap + occluded) side by side
    for idx, result in enumerate(method_results):
        method_name = result['name']
        heatmap_colored = result['heatmap']
        occluded_img = result['occluded']
        
        # Calculate grid position
        grid_row = idx // grid_cols
        grid_col = idx % grid_cols
        
        # Each method cell spans 2 columns (for heatmap + occluded)
        col_start = 3 + grid_col * 2  # Start after original image + spacing
        col_end = col_start + 2
        
        # Create subplot for this method (spanning 2 columns)
        ax_method = fig.add_subplot(gs[grid_row, col_start:col_end])
        
        # Combine heatmap and occluded side by side with spacing
        spacing_width = 10  # pixels of white space between images
        spacing = np.ones((heatmap_colored.shape[0], spacing_width, 3), dtype=np.uint8) * 255
        combined = np.hstack([heatmap_colored, spacing, occluded_img])
        ax_method.imshow(combined)
        
        # Add method name as text on the left side (inside the image)
        ax_method.text(
            -0.02, 0.5, method_name,
            transform=ax_method.transAxes,
            fontsize=11, fontweight='bold',
            color='#2c3e50', family='sans-serif',
            rotation=90, va='center', ha='right'
        )
        
        # Add subtle labels for heatmap and occluded
        ax_method.text(
            0.25, -0.02, 'Heatmap',
            transform=ax_method.transAxes,
            fontsize=9, color='#7f8c8d',
            ha='center', va='top'
        )
        ax_method.text(
            0.75, -0.02, f'{occlusion_level}% Occluded',
            transform=ax_method.transAxes,
            fontsize=9, color='#7f8c8d',
            ha='center', va='top'
        )
        
        ax_method.axis('off')
    
    # Main title with presentation quality
    title_text = (
        f"XAI Attribution Methods Comparison\n"
        f"Model: {model_name}  |  Fill Strategy: {fill_strategy}  |  Colormap: {colormap}"
    )
    fig.suptitle(
        title_text,
        fontsize=16,
        fontweight='bold',
        color='#1a1a1a',
        family='sans-serif',
        y=0.97
    )
    
    # Position the window at a fixed location (top-left of screen with offset)
    plt.tight_layout()
    
    # Save figure to results directory
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    # Extract clean filename (without extension)
    if image_path:
        base_name = Path(image_path).stem  # e.g., "002" from "002.jpeg"
    else:
        base_name = "unknown_image"

    # Create unique output file name
    output_path = results_dir / f"{base_name}_xai_visualization.png"

    try:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Visualization saved to: {output_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to save visualization: {e}")
    
    # Get the window backend and set position
    manager = plt.get_current_fig_manager()
    try:
        # Try to set window position (works on most backends)
        if hasattr(manager, 'window'):
            # Set window position to (50, 30) pixels from top-left for better visibility
            if hasattr(manager.window, 'wm_geometry'):
                # For TkAgg backend
                manager.window.wm_geometry("+50+30")
            elif hasattr(manager.window, 'setGeometry'):
                # For Qt backend
                window_width = int(fig_width * 100)
                window_height = int(fig_height * 100)
                manager.window.setGeometry(50, 30, window_width, window_height)
    except:
        pass  # If positioning fails, just show at default location
    
    plt.show()
    
    print(f"  ✓ Visualization displayed!")


# ===========================
# MAIN ENTRY POINT
# ===========================

if __name__ == "__main__":
    print("=" * 60)
    print("XAI Single Image Visualizer")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Image: {IMAGE_PATH}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Fill Strategy: {FILL_STRATEGY}")
    print(f"  Occlusion Level: {OCCLUSION_LEVEL}%")
    print(f"  Colormap: {COLORMAP}")
    print(f"  Attribution Methods: {len(ATTRIBUTION_METHODS)}")
    
    # Load and preprocess image
    print(f"\n📷 Loading image...")
    try:
        image_tensor, original_image = load_and_preprocess_image(IMAGE_PATH)
        print(f"  ✓ Image loaded: {original_image.size}")
    except Exception as e:
        print(f"  ❌ Failed to load image: {e}")
        exit(1)
    
    # Load model
    print(f"\n🤖 Loading model...")
    try:
        model = load_model(MODEL_NAME)
        print(f"  ✓ Model loaded: {MODEL_NAME}")
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        exit(1)
    
    # Get true class
    print(f"\n🎯 Determining true class...")
    try:
        true_class = get_true_class(image_tensor, model, TRUE_CLASS)
        print(f"  ✓ True class: {true_class}")
    except Exception as e:
        print(f"  ❌ Failed to determine class: {e}")
        exit(1)
    
    # Create visualization
    try:
        visualize_all_methods(
            image_tensor=image_tensor,
            original_image=original_image,
            model=model,
            model_name=MODEL_NAME,
            true_class=true_class,
            methods=ATTRIBUTION_METHODS,
            fill_strategy=FILL_STRATEGY,
            occlusion_level=OCCLUSION_LEVEL,
            colormap=COLORMAP,
            image_path=IMAGE_PATH
        )
    except Exception as e:
        print(f"\n❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print("=" * 60)
