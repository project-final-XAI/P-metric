"""
ImageNet class mapping: Synset ID to human-readable names.

Maps synset IDs (like n01440764) to readable class names (like "tench, Tinca tinca").
"""

from typing import Dict, Optional
import logging
import os

from config import DATASET_CONFIG


def get_imagenet_class_mapping() -> Dict[str, str]:
    """
    Get mapping from synset ID to human-readable class name.
    
    First tries to use torchvision's built-in mapping, then falls back to
    creating a basic mapping from file names.
    
    Returns:
        Dictionary mapping synset ID to class name
    """
    # Try to use torchvision's ImageNet class info
    try:
        # This is a built-in mapping in some torchvision versions
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V1
        
        # Get class names (these are in order 0-999)
        class_names = weights.meta["categories"]
        
        # Now we need to map these to synset IDs
        # The order matches ImageNet's synset order (alphabetically sorted)
        # We'll need the synset IDs from the dataset folder
        dataset_path = DATASET_CONFIG.get("imagenet", {}).get("path")
        if dataset_path and os.path.exists(dataset_path):
            synset_ids = sorted([d for d in os.listdir(dataset_path) 
                               if os.path.isdir(os.path.join(dataset_path, d))])
            
            if len(synset_ids) == len(class_names):
                mapping = {synset_id: class_name for synset_id, class_name in zip(synset_ids, class_names)}
                logging.info(f"Loaded ImageNet mapping using torchvision: {len(mapping)} classes")
                return mapping
    
    except Exception as e:
        logging.debug(f"Could not load ImageNet mapping from torchvision: {e}")
    
    # Fallback: Try to extract names from file names
    try:
        dataset_path = DATASET_CONFIG.get("imagenet", {}).get("path")
        if dataset_path and os.path.exists(dataset_path):
            mapping = {}
            
            for class_folder in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, class_folder)
                if not os.path.isdir(class_path):
                    continue
                
                # Try to get name from first image file
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Example: n01440764_tench.JPEG -> extract "tench"
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            # Remove extension
                            name = parts[1].rsplit('.', 1)[0]
                            # Clean up: replace underscores with spaces, capitalize
                            name = name.replace('_', ' ')
                            mapping[class_folder] = name
                            break
                
                # If no name found from filename, use synset ID
                if class_folder not in mapping:
                    mapping[class_folder] = class_folder
            
            if mapping:
                logging.info(f"Loaded ImageNet mapping from filenames: {len(mapping)} classes")
                return mapping
    
    except Exception as e:
        logging.debug(f"Could not create ImageNet mapping from filenames: {e}")
    
    # Last fallback: return empty dict (will use synset IDs as-is)
    logging.warning("Could not create ImageNet class mapping. Will use synset IDs.")
    return {}


def synset_to_readable(synset_id: str, mapping: Optional[Dict[str, str]] = None) -> str:
    """
    Convert synset ID to human-readable name.
    
    Args:
        synset_id: Synset ID (e.g., "n01440764")
        mapping: Optional pre-loaded mapping dictionary
        
    Returns:
        Human-readable class name
    """
    if mapping is None:
        mapping = get_imagenet_class_mapping()
    
    # Return mapped name if available, otherwise return synset ID
    return mapping.get(synset_id, synset_id)


# Cache for the mapping to avoid recomputing
_cached_mapping: Optional[Dict[str, str]] = None


def get_cached_mapping() -> Dict[str, str]:
    """
    Get cached ImageNet class mapping (computes once, then reuses).
    
    Returns:
        Dictionary mapping synset ID to class name
    """
    global _cached_mapping
    
    if _cached_mapping is None:
        _cached_mapping = get_imagenet_class_mapping()
    
    return _cached_mapping


# For convenience: a function to format class name for LLM
def format_class_for_llm(class_name: str) -> str:
    """
    Format class name for LLM prompts.
    
    Handles cases like:
    - "tench, Tinca tinca" -> "tench"
    - "goldfish, Carassius auratus" -> "goldfish"
    - "great white shark" -> "great white shark"
    
    Args:
        class_name: Class name (possibly with scientific name)
        
    Returns:
        Formatted name suitable for LLM
    """
    # If there's a comma, take the first part (common name)
    if ',' in class_name:
        class_name = class_name.split(',')[0].strip()
    
    # Clean up
    class_name = class_name.strip()
    
    return class_name


if __name__ == "__main__":
    # Test the mapping
    import logging
    logging.basicConfig(level=logging.INFO)
    
    mapping = get_imagenet_class_mapping()
    
    print(f"Loaded {len(mapping)} class mappings")
    print("\nFirst 10 mappings:")
    for i, (synset_id, name) in enumerate(list(mapping.items())[:10]):
        formatted = format_class_for_llm(name)
        print(f"  {synset_id} -> {name} -> {formatted}")

