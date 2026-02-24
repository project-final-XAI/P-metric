"""
Check if the label mapping is correct.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.imagenet_class_mapping import get_cached_mapping, format_class_for_llm


def main():
    # Import here to avoid multiprocessing issues on Windows
    from data.loader import get_dataloader
    
    # Load the actual dataset labels (with num_workers=0 for Windows compatibility)
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    dataset_path = config.DATASET_CONFIG["imagenet"]["path"]
    image_folder = datasets.ImageFolder(root=str(dataset_path), transform=transform)
    
    # Use num_workers=0 to avoid multiprocessing issues on Windows
    dataloader = DataLoader(image_folder, batch_size=1, shuffle=False, num_workers=0)
    
    # Get class mapping
    mapping = get_cached_mapping()
    
    # Get class names from ImageFolder
    class_names = dataloader.dataset.classes
    
    print(f"Total images: {len(dataloader.dataset)}")
    print(f"Total classes: {len(class_names)}")
    print(f"\nFirst 20 image labels:\n")
    
    # Check first 20 images
    for i, (img, label) in enumerate(dataloader):
        if i >= 20:
            break
        label_int = label.item()
        synset_id = class_names[label_int]
        readable_name = mapping.get(synset_id, synset_id)
        formatted_name = format_class_for_llm(readable_name)
        
        print(f"image_{i:05d} -> label={label_int:3d} -> synset={synset_id} -> {formatted_name}")
    
    print("\n\nChecking specific images from the test:")
    test_indices = [0, 1, 3, 7, 9, 22, 33, 49, 51, 71, 130, 291, 292]
    
    # Reload dataloader to iterate from beginning
    dataloader = DataLoader(image_folder, batch_size=1, shuffle=False, num_workers=0)
    
    for i, (img, label) in enumerate(dataloader):
        if i in test_indices:
            label_int = label.item()
            synset_id = class_names[label_int]
            readable_name = mapping.get(synset_id, synset_id)
            formatted_name = format_class_for_llm(readable_name)
            print(f"image_{i:05d} -> label={label_int:3d} -> {formatted_name}")
        if i > max(test_indices):
            break


if __name__ == '__main__':
    main()
