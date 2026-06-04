"""
Test Utility: Phase 3 Baseline (0% Occlusion) Evaluator.
Evaluates only the baseline accuracy of select judging models on a custom data directory.
"""
import time
import torch
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Tuple

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Import your framework modules
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from data.loader import get_base_transforms, get_dataset_handler
from models.loader import get_model_provider
import config as config_module


class BaselineConfig:
    DATASET_NAME = config_module.DATASET_NAME
    CUSTOM_DATA_DIR = Path(config_module.DATASET_CONFIG[DATASET_NAME]["path"])

    # List the judge models you want to test baselines for from the config file
    JUDGING_MODELS = config_module.JUDGING_MODELS

    # Runtime Tuning Performance arguments
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    DEVICE = config_module.DEVICE
    USE_FP16_INFERENCE = config_module.USE_FP16_INFERENCE


# =====================================================================
# FRAMEWORK DIRECT MATCH DATASET
# =====================================================================
class DirectBaselineDataset(Dataset):
    """
    Scans the custom folder directly and extracts items that match
    the framework's image_label_map naming indices (e.g., 'image_00000').
    """
    def __init__(self, data_dir: Path, image_label_map: Dict[str, int], transform):
        self.transform = transform
        self.pairs: List[Tuple[Path, int]] = []

        valid_extensions = {".jpg", ".jpeg", ".png"}
        # Pull and sort files from your chosen directory
        all_files = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_extensions)

        # Cross-reference using the structure format of Phase 3
        for idx, path in enumerate(all_files):
            img_id = path.stem  # Looks for names matching "image_xxxxx"
            if img_id in image_label_map:
                self.pairs.append((path, image_label_map[img_id]))
            else:
                # If original filenames are used, map them by sorted position alignment
                virtual_id = f"image_{idx:05d}"
                if virtual_id in image_label_map:
                    self.pairs.append((path, image_label_map[virtual_id]))

        if not self.pairs:
            logging.warning(
                f"Zero files inside '{data_dir}' matched the format of the framework's image_label_map index strings."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, target = self.pairs[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


# =====================================================================
# RUNNER LOGIC
# =====================================================================
def run_baseline_evaluation():
    logging.basicConfig(level=logging.INFO)
    print(f"Initializing Framework Baseline Tester...")
    print(f"Target Directory: {BaselineConfig.CUSTOM_DATA_DIR.resolve()}")

    # Instantiate management components using your system classes
    gpu_manager = GPUManager()
    file_manager = FileManager(config_module.BASE_DIR)
    dataset_handler = get_dataset_handler(BaselineConfig.DATASET_NAME)
    model_provider = get_model_provider(BaselineConfig.DATASET_NAME)

    device = torch.device(BaselineConfig.DEVICE)

    # 1. Build the true image_label_map exactly how Phase 3 builds it
    print("Building image target label mapping from framework...")
    framework_loader = dataset_handler.get_dataloader(batch_size=BaselineConfig.BATCH_SIZE, shuffle=False)
    image_label_map: Dict[str, int] = {}
    global_idx = 0

    for _, batch_labels in framework_loader:
        for lbl in batch_labels:
            image_label_map[f"image_{global_idx:05d}"] = lbl.item()
            global_idx += 1

    # 2. Iterate through specified judges
    for judge_name in BaselineConfig.JUDGING_MODELS:
        print(f"\n>>> Evaluating 0% Occlusion Baseline: {judge_name}")

        try:
            # Load judge model using unified framework provider
            model = model_provider.get_model(judge_name).to(device)
            model.eval()
            
            # Align preprocessing to be exactly the same as the evaluation pipeline (model-specific transforms)
            transform = getattr(model, "transforms", get_base_transforms())

            # Prepare custom directory dataloader with model-specific transform
            test_dataset = DirectBaselineDataset(
                data_dir=BaselineConfig.CUSTOM_DATA_DIR,
                image_label_map=image_label_map,
                transform=transform
            )

            if len(test_dataset) == 0:
                print(f"Execution skipped for {judge_name}: Dataset mapping is empty.")
                continue

            dataloader = DataLoader(
                test_dataset,
                batch_size=BaselineConfig.BATCH_SIZE,
                num_workers=BaselineConfig.NUM_WORKERS,
                pin_memory=(device.type == "cuda"),
                shuffle=False
            )

            print(f"Matched {len(test_dataset)} images ready for baseline assessment.")
            print("=" * 70)

            # Cast model down if running FP16 inference
            if BaselineConfig.USE_FP16_INFERENCE and device.type == "cuda" and gpu_manager.supports_fp16():
                model = model.half()
                print("    * Cast model weights to half-precision (FP16)")

            top1_correct = 0
            top5_correct = 0
            total_samples = 0

            start_time = time.time()

            with torch.inference_mode():
                for imgs, targets in tqdm(dataloader, desc=f"      Predicting {judge_name}", leave=False):
                    if BaselineConfig.USE_FP16_INFERENCE and device.type == "cuda":
                        imgs = imgs.half()

                    imgs = imgs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    # Unified topk implementation for fast inference matching Phase 3 approach
                    outputs = model(imgs)
                    
                    # Handle models with fewer than 5 classes (like SIPaKMeD which has 5 classes, or other custom models)
                    num_classes = outputs.shape[1]
                    k = min(5, num_classes)
                    _, top5_preds = outputs.topk(k, dim=1)

                    top1_correct += (top5_preds[:, 0] == targets).sum().item()
                    top5_correct += (top5_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
                    total_samples += targets.size(0)

            elapsed_time = time.time() - start_time

            print(f"    [RESULTS - 0% OCCLUSION] for {judge_name}:")
            print(f"    --> Sample Size Processed  : {total_samples}")
            print(f"    --> Top-1 Baseline Accuracy: {top1_correct / total_samples:.4f} ({top1_correct}/{total_samples})")
            print(f"    --> Top-5 Baseline Accuracy: {top5_correct / total_samples:.4f} ({top5_correct}/{total_samples})")
            print(f"    --> Runtime Execution Time : {elapsed_time:.2f}s")

        except Exception as e:
            logging.error(f"Error checking model {judge_name}: {e}", exc_info=True)

        finally:
            # Prevent lingering VRAM leak artifacts across sequential models
            if 'model' in locals():
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Finished evaluating target directory baselines.")


if __name__ == "__main__":
    run_baseline_evaluation()