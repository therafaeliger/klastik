import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import random
import pickle
from PIL import Image

from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix


# Label Extraction
def get_label_from_filename(filename):
    name = os.path.splitext(filename)[0]
    label = re.split(r'\d+', name)[0]
    return label.lower()


# Preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    eq = cv2.equalizeHist(resized)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)
    return blur


# Feature Extraction
def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, channel_axis=None)
    return features
def extract_color_hist(image, bins=(8, 8, 8)):
    resized = cv2.resize(image, (128, 128))  # ensure fixed size
    hist = cv2.calcHist([resized], [0, 1, 2], None, bins,
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        return np.zeros(128)
    return descriptors.flatten()[:128]
def extract_lbp_features(gray_image, P=8, R=1, method='uniform'):
    lbp = local_binary_pattern(gray_image, P, R, method)
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


# Feature Selection
def extract_features(image_path, mode='hog'):
    gray_image = preprocess_image(image_path)
    color_image = None
    if 'color' in mode or mode == 'all_combined':
        color_image = cv2.imread(image_path)
        if color_image is None:
            raise ValueError(f"Could not read color image: {image_path}")
    if mode == 'hog':
        return extract_hog_features(gray_image)
    elif mode == 'sift':
        return extract_sift_features(gray_image)
    elif mode == 'color':
        if color_image is None: color_image = cv2.imread(image_path)
        return extract_color_hist(color_image)
    elif mode == 'lbp':
        return extract_lbp_features(gray_image)
    elif mode == 'hog+color':
        if color_image is None: color_image = cv2.imread(image_path)
        return np.hstack([
            extract_hog_features(gray_image),
            extract_color_hist(color_image)
        ])
    elif mode == 'hog+lbp':
        return np.hstack([
            extract_hog_features(gray_image),
            extract_lbp_features(gray_image)
        ])
    elif mode == 'all_combined':
        if color_image is None: color_image = cv2.imread(image_path)
        sift_feat = extract_sift_features(gray_image)
        return np.hstack([
            extract_hog_features(gray_image).flatten(),
            extract_color_hist(color_image).flatten(),
            extract_lbp_features(gray_image).flatten(),
        ])
    else:
        raise ValueError(f"Unsupported feature mode: {mode}")


# Load Data
features, labels = [], []
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for fname in image_files:
    label = get_label_from_filename(fname)
    full_path = os.path.join(folder_path, fname)
    feat = extract_features(full_path, mode=use_feature)
    features.append(feat)
    labels.append(label)

# print(features)
# print(labels)

# Encode Labels
le = LabelEncoder()
y = le.fit_transform(labels)
X_original = np.array(features)

#  Scaling Features
scaler = StandardScaler()
X = scaler.fit_transform(X_original)


# Calculate the distribution of labels
label_counts = {}
for label in labels:
    label_counts[label] = label_counts.get(label, 0) + 1

# Prepare data for plotting
labels_plot = list(label_counts.keys())
counts_plot = list(label_counts.values())

# Create the distribution chart
plt.figure(figsize=(6, 7))
bars = plt.bar(labels_plot, counts_plot, color='skyblue')
plt.xlabel('Label')
plt.ylabel('Total Count')
plt.title('Distribusi Label')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()

# Add exact total values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval , int(yval), ha='center', va='bottom')

plt.show()

# Print the exact total values for each label
print("Exact total value for each label:")
for label, count in label_counts.items():
    print(f"{label}: {count}")


# Train/Test Split
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_original, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_orig)
X_test = scaler.transform(X_test_orig)

unique_labels_in_test = np.unique(y_test)
target_names_in_test = le.inverse_transform(unique_labels_in_test)


def plot_distribution(y_data, title, le):
    unique_labels, counts = np.unique(y_data, return_counts=True)
    labels = le.inverse_transform(unique_labels)
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


plot_distribution(y_train, "Training Data Distribution", le)
plot_distribution(y_test, "Test Data Distribution", le)


# Models
# models = {
#     "SVM (Linear)": SVC(kernel='linear', probability=True),
#     "SVM (RBF)": SVC(kernel='rbf', probability=True),
#     "Random Forest": RandomForestClassifier(n_estimators=100),
#     "KNN": KNeighborsClassifier(n_neighbors=3)
# }

models_to_tune = {
    "SVM (Linear)": (SVC(probability=True), {
        'kernel': ['linear'],
        'C': [0.1, 1, 10, 100]
    }),
    "SVM (RBF)": (SVC(probability=True), {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
    }),
    "Random Forest": (RandomForestClassifier(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    "KNN": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    })
}


# Train & Evaluate
# for name, model in models.items():
#     print(f"\n\nEvaluating model: {name}")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {acc:.4f}")
#     print(classification_report(y_test, y_pred, labels=unique_labels_in_test, target_names=target_names_in_test))

#     cm = confusion_matrix(y_test, y_pred, labels=unique_labels_in_test)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names_in_test)
#     disp.plot(xticks_rotation='vertical', cmap='Blues')
#     plt.title(f"Confusion Matrix: {name}")
#     plt.tight_layout()
#     plt.show()

best_models = {}
for name, (model, params) in models_to_tune.items():
    print(f"\n\nTuning and Evaluating model: {name}")
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1) # cv=5 untuk 5-fold cross-validation
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy for {name}: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    best_models[name] = best_model # Simpan model terbaik

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy for {name} (with best params): {acc:.4f}")

    # Dapatkan unique labels dari y_test yang sudah diencode dan target names yang sesuai
    current_unique_labels_in_test = np.unique(y_test)
    current_target_names_in_test = le.inverse_transform(current_unique_labels_in_test)

    print(classification_report(y_test, y_pred, labels=current_unique_labels_in_test, target_names=current_target_names_in_test, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=current_unique_labels_in_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=current_target_names_in_test)
    disp.plot(xticks_rotation='vertical', cmap='Blues')
    plt.title(f"Confusion Matrix: {name} (Best Params)")
    plt.tight_layout()
    plt.show()


def predict_new_image(image_path, model, mode='all_combined'):
    feat = extract_features(image_path, mode=mode).reshape(1, -1)

    pred_encoded = model.predict(feat)[0]
    label = le.inverse_transform([pred_encoded])[0]
    
    return label

# Example usage
image_path = "/content/drive/MyDrive/Datasets/dataset-plastic/drink bottle8.jpg"
predicted_label = predict_new_image(image_path, best_models['Random Forest'], mode='all_combined')
print(f"Predicted: {predicted_label}")