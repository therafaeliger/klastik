import os
from roboflow import Roboflow

rf = Roboflow(api_key="580eO7BkLjveCaG01yCB")
project = rf.workspace().project("yolo-waste-detection")
model = project.version(1).model

def rename_images_with_roboflow(folder_path):
    counter = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)

            results = model.predict(image_path, confidence=40, overlap=30).json()

            if len(results['predictions']) > 0:
                top_prediction = results['predictions'][0]
                class_name = top_prediction['class']

                counter.setdefault(class_name, 0)
                counter[class_name] += 1
                new_filename = f"{class_name}_{counter[class_name]}.jpg"

                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} → {new_filename}")
            else:
                print(f"No object detected in {filename}, skipped.")

# Example usage
rename_images_with_roboflow("/home/rafael/kuliah/pattern_recognition/proyek_akhir/dataset-sampah/dataset-plastic")
