import os
import csv
import urllib.request
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np

# ==================== CONFIG ====================
TEST_MODE = False
TEST_FILE_LIMIT = 2
BATCH_SIZE = 350

# Toggle: include naive_occ in plots or not
INCLUDE_NAIVE_OCC_IN_PLOTS = False

MODEL_CONFIG = {
    "resnet18": {"model_func": models.resnet18, "weights": models.ResNet18_Weights.IMAGENET1K_V1, "name": "ResNet18"},
    "mobilenet_v2": {"model_func": models.mobilenet_v2, "weights": models.MobileNet_V2_Weights.IMAGENET1K_V1, "name": "MobileNetV2"},
    "resnet50": {"model_func": models.resnet50, "weights": models.ResNet50_Weights.IMAGENET1K_V1, "name": "ResNet50"},
    "efficientnet_b0": {"model_func": models.efficientnet_b0, "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1, "name": "EfficientNetB0"},
    "vit_b_16": {"model_func": models.vit_b_16, "weights": models.ViT_B_16_Weights.IMAGENET1K_V1, "name": "ViT-B/16"},
    "swin_t": {"model_func": models.swin_t, "weights": models.Swin_T_Weights.IMAGENET1K_V1, "name": "Swin-T"}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True

# ===== Load ImageNet Labels =====
with urllib.request.urlopen("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as url:
    idx_to_label = [line.decode("utf-8").strip().lower() for line in url]

# Percentages now include 0 and 100
percentages = [0] + list(range(5, 100, 5)) + [100]
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# keep naive_occ for CSV, only filter other unwanted methods
to_remove = {"grad_pp", "aveall", "avefour", "vit_layercam_pytorch", "vit_ig_plus_attentionxgrad"}

# ==================== HELPERS ====================
def load_image_and_class(img_path, method_suffixes, method, folder):
    try:
        root = os.path.dirname(img_path)
        file = os.path.basename(img_path)
        if root != folder:
            class_name = os.path.basename(root).lower().replace("_", " ")
        else:
            filename_parts = file.split('_')
            if len(filename_parts) >= 2 and filename_parts[0].startswith('n'):
                class_part = '_'.join(filename_parts[1:]).rsplit('.', 1)[0]
                for suffix in method_suffixes:
                    if class_part.endswith('_' + suffix):
                        class_name = class_part[:-len('_' + suffix)].lower().replace("_", " ")
                        break
                else:
                    class_name = class_part.lower().replace("_", " ")
            else:
                class_name = os.path.splitext(file)[0].lower().replace("_", " ")

        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        return tensor, class_name, file
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

color_map = {
    "saliency": "#0072B2", "smoothgrad": "#556B2F", "inputxgradient": "#8c564b",
    "guided_backprop": "#CC79A7", "occlusion": "#A60628", "xrai": "#E41A1C",
    "random_baseline": "black", "gradcam": "#E69F00", "vit_gradcam_pytorch": "#E69F00",
    "vit_gradcam_pytorch_aveln": "#6A3D9A", "attention_x_gradient": "#FFD700",
    "integrated_gradients": "#FF33CC", "vit_ig": "#FF33CC", "vit_guided_ig": "#009688",
    "gradientshap": "#228B22",
}
marker_map = {
    "saliency": "o", "smoothgrad": "s", "inputxgradient": "D", "guided_backprop": "^",
    "occlusion": "X", "xrai": "P", "random_baseline": "v", "gradcam": "h",
    "vit_gradcam_pytorch": "h", "vit_gradcam_pytorch_aveln": "H", "attention_x_gradient": "*",
    "integrated_gradients": "<", "vit_ig": "<", "vit_guided_ig": ">", "gradientshap": "p",
}
label_map = {
    "gradientshap": "GradientShap", "guided_backprop": "Guided Backprop", "inputxgradient": "Input × Gradient",
    "integrated_gradients": "Integrated Gradients", "occlusion": "Occlusion", "random_baseline": "Random Baseline",
    "saliency": "Saliency", "smoothgrad": "SmoothGrad", "xrai": "XRAI", "gradcam": "Grad-CAM",
    "guided_gradcam": "Guided Grad-CAM", "expected_gradcam": "Expected Grad-CAM", "naive_occ": "Naive Occlusion",
    "vit_gradcam_pytorch": "Grad-CAM", "vit_gradcam_pytorch_aveln": "Grad-CAM (AveLN1)",
    "vit_ig": "Integrated Gradients", "vit_guided_ig": "Integrated Gradients",
    "attention_x_gradient": "Attention × Gradient",
}

def plot_results(results, title, filename, ylabel, sorted_methods):
    plt.figure(figsize=(12, 8))
    for m in sorted_methods:
        if results[m] and any(r is not None for r in results[m]):
            plt.plot(results["Percentage"], results[m],
                     marker=marker_map.get(m, "o"), markersize=5, alpha=0.9,
                     label=label_map.get(m, m), linewidth=2, color=color_map.get(m, None))

    plt.xlabel("Percentage of Pixels Removed")
    plt.ylabel(ylabel)
    plt.title(title, fontsize=14, fontweight="bold")

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: x[0]))
    legend = plt.legend(handles, labels, fontsize=9, ncol=1, loc='upper right',
                        bbox_to_anchor=(1, 1), borderaxespad=0,
                        frameon=True, facecolor='white', framealpha=1.0)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ==================== MAIN LOOP ====================
RESULTS_DIR = Path(r"C:\evaluation_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_PARENT = Path.cwd()
occlusion_dirs = [
    d for d in BASE_PARENT.iterdir()
    if d.is_dir() and not d.name.lower().startswith(".") and not d.name.startswith("__")
]

for SELECTED_MODEL, model_config in MODEL_CONFIG.items():
    MODEL_NAME = model_config["name"]
    print(f"\n\n##############################")
    print(f"Running evaluation with model: {MODEL_NAME}")
    print(f"##############################")

    model = model_config["model_func"](weights=model_config["weights"]).to(device)
    model.eval()

    for occ_dir in occlusion_dirs:
        occlusion_mode = occ_dir.name
        csv_file = RESULTS_DIR / f"{occlusion_mode}_{SELECTED_MODEL.lower()}_results.csv"

        if csv_file.exists():
            print(f"⏭ Skipping {occlusion_mode} for {MODEL_NAME} (results already exist: {csv_file})")
            continue

        print(f"\n===== Processing occlusion: {occlusion_mode} =====")

        XAI_DIRS = {
            d.name: str(d)
            for d in occ_dir.iterdir()
            if d.is_dir() and not d.name.lower().startswith(".") and not d.name.startswith("__")
        }
        display_names = {k: k.replace("_", " ").title() for k in XAI_DIRS}

        sorted_methods = sorted([m for m in XAI_DIRS.keys() if m not in to_remove],
                                key=lambda m: display_names[m].lower())

        if "vit_ig" in sorted_methods and "integrated_gradients" in sorted_methods:
            sorted_methods.remove("integrated_gradients")

        results_top1 = {"Percentage": percentages}
        results_top5 = {"Percentage": percentages}
        for m in sorted_methods:
            results_top1[m], results_top5[m] = [], []

        raw_suffixes = [f"{m}{pct:02}" for m in sorted_methods for pct in range(5, 100, 5)]
        method_suffixes = sorted(raw_suffixes, key=len, reverse=True)

        # ========== EVALUATION ==========
        for method in sorted_methods:
            display_method = display_names[method]

            for pct in percentages:
                if pct == 0:
                    folder = r"C:\imagenet"
                elif pct == 100:
                    if occlusion_mode == "blur":
                        folder = r"C:\imagenet_blur"
                    else:
                        results_top1[method].append(0.0)
                        results_top5[method].append(0.0)
                        print(f"{display_method}100: Top-1 = 0.000, Top-5 = 0.000 (forced)")
                        continue
                else:
                    folder = os.path.join(XAI_DIRS[method], f"{method}{pct:02}")
                    if not os.path.exists(folder):
                        results_top1[method].append(None)
                        results_top5[method].append(None)
                        continue

                valid_files = [os.path.join(root, f) for root, _, files in os.walk(folder)
                               for f in files if os.path.splitext(f)[1].lower() in valid_extensions]

                if TEST_MODE:
                    valid_files = valid_files[:TEST_FILE_LIMIT]

                if not valid_files:
                    results_top1[method].append(None)
                    results_top5[method].append(None)
                    continue

                images, class_names = [], []
                max_workers = min(32, os.cpu_count() * 2)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(tqdm(executor.map(lambda f: load_image_and_class(f, method_suffixes, method, folder),
                                                     valid_files),
                                        total=len(valid_files), desc=f"{display_method}{pct:02} loading"))
                for res in results:
                    if res:
                        tensor, class_name, _ = res
                        images.append(tensor.squeeze(0))
                        class_names.append(class_name)

                if not images:
                    results_top1[method].append(None)
                    results_top5[method].append(None)
                    continue

                dummy_batch = torch.stack([images[0]] * min(BATCH_SIZE, len(images))).to(device)
                with torch.no_grad():
                    _ = model(dummy_batch)
                del dummy_batch

                top1_correct, top5_correct, total = 0, 0, 0
                for i in range(0, len(images), BATCH_SIZE):
                    batch = images[i:i + BATCH_SIZE]
                    batch_tensors = torch.stack(batch).to(device, non_blocking=True)
                    with torch.no_grad():
                        outputs = model(batch_tensors)
                        _, top1 = outputs.topk(1, dim=1)
                        _, top5 = outputs.topk(5, dim=1)

                    top1_cpu, top5_cpu = top1.cpu(), top5.cpu()
                    for j in range(batch_tensors.size(0)):
                        cls_name = class_names[i + j]
                        pred1 = idx_to_label[top1_cpu[j].item()]
                        preds5 = [idx_to_label[idx] for idx in top5_cpu[j]]
                        variants = [cls_name, cls_name.replace("_", " "), cls_name.replace(" ", "_")]
                        if any(v in pred1 for v in variants):
                            top1_correct += 1
                        if any(any(v in lbl for v in variants) for lbl in preds5):
                            top5_correct += 1
                        total += 1
                    del batch_tensors, outputs, top1, top5, top1_cpu, top5_cpu
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                acc1 = top1_correct / total if total > 0 else None
                acc5 = top5_correct / total if total > 0 else None
                results_top1[method].append(acc1)
                results_top5[method].append(acc5)
                print(f"{display_method}{pct:02}: Top-1 = {acc1 if acc1 else 'NA'}, Top-5 = {acc5 if acc5 else 'NA'} ({total} images)")

        # ==================== CSV OUTPUT ====================
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"Performance Results by Method and Metric - {MODEL_NAME} (Occlusion: {occlusion_mode})"])
            writer.writerow([])
            writer.writerow(["Top1 Results"])
            writer.writerow(["Method"] + percentages)
            for m in sorted_methods:
                writer.writerow([display_names[m]] + results_top1[m])
            writer.writerow([])
            writer.writerow(["Top5 Results"])
            writer.writerow(["Method"] + percentages)
            for m in sorted_methods:
                writer.writerow([display_names[m]] + results_top5[m])

        print(f"Results saved to: {csv_file}")

        # ==================== PLOTTING ====================
        if INCLUDE_NAIVE_OCC_IN_PLOTS:
            plot_methods = sorted_methods
        else:
            plot_methods = [m for m in sorted_methods if m != "naive_occ"]

        plot_results(results_top1,
                     f"XAI Method Comparison - Top-1 Accuracy ({MODEL_NAME}, {occlusion_mode})",
                     RESULTS_DIR / f"{occlusion_mode}_{SELECTED_MODEL.lower()}_top1_accuracy.png",
                     "Top-1 Accuracy", plot_methods)

        plot_results(results_top5,
                     f"XAI Method Comparison - Top-5 Accuracy ({MODEL_NAME}, {occlusion_mode})",
                     RESULTS_DIR / f"{occlusion_mode}_{SELECTED_MODEL.lower()}_top5_accuracy.png",
                     "Top-5 Accuracy", plot_methods)

print("\nAll models and occlusion directories processed. Evaluation complete!")
