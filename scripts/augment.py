import os
import re
from collections import defaultdict
from PIL import Image
from torchvision import transforms

# --- Folder Path ---
folder_path = '/home/rafael/kuliah/pattern_recognition/proyek_akhir/dataset-sampah/new-dataset-plastic'

# --- Label Normalization Map
label_map = {
    # Add more if needed
}

# --- Augmentation Pipeline
aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # Tambahan baru
    transforms.RandomRotation(15),  # Lebih fleksibel dari 10
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),  # Lebih beragam
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Tambahan: scale dan shear
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Tambahan baru
    transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),  # Tambahan baru
])


# --- Step 1: Augment Images
def augment_images(folder_path, n_augments=3):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for fname in image_files:
        try:
            img_path = os.path.join(folder_path, fname)
            img = Image.open(img_path).convert("RGB")

            for i in range(1, n_augments + 1):
                aug_img = aug(img)
                aug_fname = f"{os.path.splitext(fname)[0]}_aug{i}{os.path.splitext(fname)[1]}"
                aug_img.save(os.path.join(folder_path, aug_fname))
                print(f"✅ Augmented: {aug_fname}")
        except Exception as e:
            print(f"⚠️ Error augmenting {fname}: {e}")

# --- Step 2: Normalize & Reorder All Images
def normalize_label(raw_name):
    base = os.path.splitext(raw_name)[0]
    label = re.sub(r'\d+|_aug\d+', '', base).strip().lower().replace('_', ' ')
    return label_map.get(label, label)

def rename_and_reorder(folder_path):
    files_by_label = defaultdict(list)

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            label = normalize_label(fname)
            files_by_label[label].append(fname)

    for label, files in files_by_label.items():
        files.sort()  # optional: can sort based on augmentation or name
        for i, fname in enumerate(files, start=1):
            ext = os.path.splitext(fname)[1]
            new_name = f"{label}{i}{ext}"
            src = os.path.join(folder_path, fname)
            dst = os.path.join(folder_path, new_name)
            if src != dst:
                os.rename(src, dst)
                print(f"🔁 Renamed: {fname} → {new_name}")

# --- Run Everything
augment_images(folder_path, n_augments=3)
rename_and_reorder(folder_path)
