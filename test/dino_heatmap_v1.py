import torch
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from transformers import AutoImageProcessor, Dinov2Model
import random
import glob
import os


def get_random_local_image(base_path):
    """
    סורק תיקייה באופן רקוסיבי ומחזיר נתיב אקראי לתמונה (jpg, jpeg, png).
    """
    extensions = ('/**/*.jpg', '/**/*.jpeg', '/**/*.png')
    all_files = []
    print(f"Scanning {base_path}...")
    for ext in extensions:
        all_files.extend(glob.glob(base_path + ext, recursive=True))
    if not all_files:
        raise FileNotFoundError(f"No images found in {base_path}")
    return random.choice(all_files)


def load_image(path_or_url):
    """
    טוען תמונה מכתובת אינטרנט או מנתיב מקומי במחשב.
    ממיר את התמונה לפורמט RGB.
    """
    if path_or_url.startswith('http'):
        response = requests.get(path_or_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"File not found: {path_or_url}")
        img = Image.open(path_or_url).convert('RGB')
    return img


def create_soft_heatmap(binary_mask, sigma=10):
    """
    הופך מסיכה בינארית למפת חום רכה.
    משתמש ב-Distance Transform כדי לתת עוצמה למרכז האובייקט
    וב-Gaussian Filter כדי ליצור טשטוש המדמה פלט של XAI.
    """
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    # חישוב מרחק מהרקע (נותן עוצמה גבוהה במרכז האובייקט)
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    # החלקה ליצירת מראה טבעי
    heatmap = gaussian_filter(dist_transform, sigma=sigma)
    # נרמול לטווח 0-1
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap


# הגדרת מודל DINOv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = Dinov2Model.from_pretrained(model_name).to(device)


def process_with_pca(image_path):
    """
    מבצע את התהליך המלא:
    1. חילוץ תכונות (Features) מהמודל.
    2. ביצוע PCA לבידוד האובייקט המרכזי מהרקע.
    3. יצירת מפת חום מבוססת סגמנטציה.
    """
    img_orig = load_image(image_path)
    filename = os.path.basename(image_path)

    inputs = processor(images=img_orig, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # שולפים את ה-Features של ה-patches (בלי ה-CLS token)
        features = outputs.last_hidden_state[:, 1:, :]

        # עיבוד PCA
    features = features.squeeze(0).cpu().numpy()
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)

    # חישוב גודל הגריד (עבור 'base' זה לרוב 16x16 ברזולוציית 224)
    num_patches = features.shape[0]
    grid_size = int(np.sqrt(num_patches))

    # המרכיב הראשון (PC1) מפריד בדרך כלל בין האובייקט לרקע
    obj_mask = pca_features[:, 0].reshape(grid_size, grid_size)

    # נירמול המסיכה
    obj_mask = (obj_mask - obj_mask.min()) / (obj_mask.max() - obj_mask.min())

    # תיקון "היפוך צבעים" - מוודאים שהאובייקט בהיר והרקע כהה
    if obj_mask[0, 0] > 0.5:
        obj_mask = 1 - obj_mask

    # יצירת מסיכה בינארית (Threshold מבוסס אחוזון)
    binary_mask = (obj_mask > np.quantile(obj_mask, 0.85)).astype(float)
    binary_mask = cv2.resize(binary_mask, (img_orig.size[0], img_orig.size[1]))

    # יצירת מפת חום סופית
    final_heatmap = create_soft_heatmap(binary_mask)

    show_results(img_orig, obj_mask, final_heatmap, filename)


def show_results(img, pca_map, heatmap, title_name):
    """
    מציג את התוצאות בשלושה פאנלים: תמונה מקורית, מפת ה-PCA ומפת החום הסופית.
    """
    fig = plt.figure(figsize=(15, 6))
    # הוספת שם הקובץ ככותרת על לכל הגרף
    fig.suptitle(f"File: {title_name}", fontsize=14, fontweight='bold')

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pca_map, cmap='viridis')
    plt.title("PCA Component 1 (Object Isolation)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Final Soft Heatmap")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# הרצה
try:
    path = get_random_local_image('../data/SIPaKMed_cropped')
    process_with_pca(path)
except Exception as e:
    print(f"Error encountered: {e}")