import os
from ultralytics import YOLO
from roboflow import Roboflow

model = YOLO("yolov8m.pt")

def rename_images_with_yolo(folder_path):
    counter = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)

            # Run YOLO on the image
            results = model(image_path)

            if results and len(results[0].boxes) > 0:
                top_class_id = int(results[0].boxes.cls[0])
                class_name = model.names[top_class_id]

                counter.setdefault(class_name, 0)
                counter[class_name] += 1
                new_filename = f"{class_name}_{counter[class_name]}.jpg"

                new_path = os.path.join(folder_path, new_filename)
                os.rename(image_path, new_path)
                print(f"Renamed: {filename} → {new_filename}")
            else:
                print(f"No object detected in {filename}, skipped.")

# Example usage
rename_images_with_yolo("/home/rafael/kuliah/pattern_recognition/proyek_akhir/dataset-sampah/dataset-plastic")
