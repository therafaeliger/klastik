import os
import re
from collections import defaultdict

label_map = {
    'container for household chemicals': 'drink bottle',

}

def normalize_label(filename):
    name = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    # Remove trailing numbers (e.g., 'bottle12' → 'bottle')
    label = re.sub(r'\d+', '', name).strip().lower().replace('_', ' ')
    
    # Normalize label using the map
    norm_label = label_map.get(label, label)  # default to itself if not in map
    return norm_label, ext

def rename_and_reorder(folder_path):
    files_by_label = defaultdict(list)

    # Step 1: Group files by normalized label
    for fname in os.listdir(folder_path):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            label, ext = normalize_label(fname)
            files_by_label[label].append((fname, ext))

    # Step 2: Rename files per label group
    for label, files in files_by_label.items():
        # Sort to ensure consistent order
        files.sort(key=lambda x: int(re.findall(r'\d+', x[0])[-1]) if re.findall(r'\d+', x[0]) else 0)

        for i, (original_name, ext) in enumerate(files, start=1):
            new_name = f"{label}{i}{ext}"
            src = os.path.join(folder_path, original_name)
            dst = os.path.join(folder_path, new_name)

            if src != dst:
                os.rename(src, dst)
                print(f"Renamed: {original_name} → {new_name}")

# 🔧 Set your folder path here
folder_path = '/home/rafael/kuliah/pattern_recognition/proyek_akhir/dataset-sampah/new-dataset-plastic'
rename_and_reorder(folder_path)
