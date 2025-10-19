# main.py

import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config import (
    CNN_MODEL_FUNC, CNN_WEIGHTS, MODEL_NAME, MODEL_TYPE, GRADCAM_TARGET_LAYER, device,
    INPUT_FOLDER, HEATMAP_FOLDER, OUTPUT_FOLDER, THRESHOLDS
)
from stages import generate_heatmaps, generate_occlusions_from_heatmaps

if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get list of images
    image_files = [
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"[Model] NAME={MODEL_NAME} TYPE={MODEL_TYPE}  CAM_LAYER={GRADCAM_TARGET_LAYER}")
    print(f"Processing {len(image_files)} images...")

    # Load model ONCE and warmup
    model = CNN_MODEL_FUNC(weights=CNN_WEIGHTS).to(device).eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, 224, 224, device=device))

    # --------------------
    # Stage 1: Heatmaps Generation
    # --------------------
    max_visuals     = 10  # only save visual JPGs for first N images
    skipped_count   = 0
    processed_count = 0

    # Use a low worker count for Stage 1 to avoid potential CUDA OOM issues or hook contention
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(generate_heatmaps, img_path, HEATMAP_FOLDER, model, index=i, max_visuals=max_visuals): img_path
            for i, img_path in enumerate(image_files)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing heatmaps"):
            try:
                result = future.result()
                if result and result.startswith("SKIPPED"):
                    skipped_count += 1
                elif result and result.startswith("COMPLETED"):
                    processed_count += 1
            except Exception as e:
                print(f"Error processing image: {e}")

    print(f"Stage 1 Summary: {processed_count} processed, {skipped_count} skipped")

    # --------------------
    # Stage 2: Occlusions from Heatmaps
    # --------------------
    # Stage 2 is mostly CPU/IO-bound, can use more workers
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(generate_occlusions_from_heatmaps, img_path, HEATMAP_FOLDER, OUTPUT_FOLDER, THRESHOLDS): img_path
            for img_path in image_files
        }
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Occlusions created"):
            pass

    print("Stage 2: All heatmaps and occlusion images have been generated.")