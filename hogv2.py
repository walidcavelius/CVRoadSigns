"""
Experiment using the HOG (Histogram of Oriented Gradients) feature descriptor
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
import skimage.color as color
import skimage.feature as feature
from skimage import data, io, util
import skimage.transform as transform
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.draw import rectangle
import imutils
import pickle
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from collections import Counter
import csv

CLASSES = [
    "none",
    "danger",
    "interdiction",
    "obligation",
    "stop",
    "ceder",
    "frouge",
    "forange",
    "fvert",
]


class Dataset:
    """
    Class to load images from the dataset

    Args:
        path (str): path to the dataset
    """

    data = []

    def __init__(self, path: str, image_size=(64, 64), augment_data=False):
        self.path = path

        self.image_size = image_size

        for subdir, _, files in os.walk(self.path):
            if not os.path.basename(subdir) == "labels":
                continue

            for file in tqdm(files, desc="Processing images"):
                imid = int(os.path.splitext(os.path.basename(file))[0])
                image = util.img_as_float(color.rgb2gray(self._load_image(imid)))

                labels = self._load_label(imid)

                if labels.empty:
                    panel = transform.resize(image, self.image_size)
                    ft = np.array(
                        feature.hog(
                            panel,
                            orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            feature_vector=True,
                        )
                    )
                    self.data.append(
                        [f"{imid}_0", panel, ft, self.class_to_index("none")]
                    )
                else:
                    for index, row in labels.iterrows():
                        try:
                            x1, y1, x2, y2, label = row

                            if x1 >= x2 or y1 >= y2:
                                tqdm.write(
                                    f"Invalid bounding box for image {imid}, index {index}. Skipping..."
                                )
                                continue

                            panel = image[y1:y2, x1:x2]

                            if panel.size == 0:
                                tqdm.write(
                                    f"Zero-sized panel for image {imid}, index {index}. Skipping..."
                                )
                                continue

                            panel = transform.resize(panel, self.image_size)

                            ft = np.array(
                                feature.hog(
                                    panel,
                                    orientations=9,
                                    pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2),
                                    feature_vector=True,
                                )
                            )
                            self.data.append(
                                [
                                    f"{imid}_{index}",
                                    panel,
                                    ft,
                                    self.class_to_index(label),
                                ]
                            )
                        except (ValueError, IndexError):
                            pass

        print(f"Loaded {len(self.data)} panels.")

        if augment_data:
            self._augment_data()
    def _augment_data(self):
        augmented_data = []
        for imid, panel, ft, label in self.data:
            # Rotation
            rotated = transform.rotate(panel, angle=15)  # Rotation de 15 degrés
            ft_rotated = np.array(feature.hog(rotated, orientations=9, pixels_per_cell=(8, 8),
                                              cells_per_block=(2, 2), feature_vector=True))
            augmented_data.append([f"{imid}_rot15", rotated, ft_rotated, label])

            # Flipping horizontally
            flipped_h = np.fliplr(panel)
            ft_flipped_h = np.array(feature.hog(flipped_h, orientations=9, pixels_per_cell=(8, 8),
                                                cells_per_block=(2, 2), feature_vector=True))
            augmented_data.append([f"{imid}_fliph", flipped_h, ft_flipped_h, label])

            # Flipping vertically
            flipped_v = np.flipud(panel)
            ft_flipped_v = np.array(feature.hog(flipped_v, orientations=9, pixels_per_cell=(8, 8),
                                                cells_per_block=(2, 2), feature_vector=True))
            augmented_data.append([f"{imid}_flipv", flipped_v, ft_flipped_v, label])

            # Adding noise
            noisy = util.random_noise(panel, mode='gaussian')
            ft_noisy = np.array(feature.hog(noisy, orientations=9, pixels_per_cell=(8, 8),
                                            cells_per_block=(2, 2), feature_vector=True))
            augmented_data.append([f"{imid}_noise", noisy, ft_noisy, label])

        self.data.extend(augmented_data)
        print(f"Added {len(augmented_data)} augmented images.")


    def _load_image(self, imid: int):
        return io.imread(f"{self.path}/images/{imid:04d}.jpg")

    def _load_label(self, imid: int):
        df = pd.read_csv(
            f"{self.path}/labels/{imid:04d}.csv",
            header=None,
            names=["x1", "y1", "x2", "y2", "label"],
        )

        return df

    @staticmethod
    def class_to_index(label: str):
        return CLASSES.index(label)

    @staticmethod
    def index_to_class(index: int):
        return CLASSES[index]

    def get_features(self):
        return [d[2] for d in self.data]

    def get_labels(self):
        return [d[3] for d in self.data]


def sliding_window(img, step_size, window_size):
    for y in range(0, img.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, img.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, img[y : y + window_size[1], x : x + window_size[0]])


def improved_non_max_suppression(boxes, overlap_thresh=0.3, min_size=(30, 30), max_size=(300, 300), image_shape=None):
    if len(boxes) == 0:
        return []

    # convertit les boîtes en format [x1, y1, x2, y2, class, score]
    boxes_array = np.array([[x, y, x+w, y+h, cls, score] for (x, y, w, h, cls, score) in boxes])

    order = boxes_array[:, 5].argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(boxes_array[i, 0], boxes_array[order[1:], 0])
        yy1 = np.maximum(boxes_array[i, 1], boxes_array[order[1:], 1])
        xx2 = np.minimum(boxes_array[i, 2], boxes_array[order[1:], 2])
        yy2 = np.minimum(boxes_array[i, 3], boxes_array[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / ((boxes_array[i, 2] - boxes_array[i, 0] + 1) * (boxes_array[i, 3] - boxes_array[i, 1] + 1) +
                       (boxes_array[order[1:], 2] - boxes_array[order[1:], 0] + 1) * (boxes_array[order[1:], 3] - boxes_array[order[1:], 1] + 1) - inter)

        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]

    # verif plus proches
    final_boxes = []
    for i, box in enumerate(keep):
        x1, y1, x2, y2, cls, score = boxes_array[box]
        is_valid = True
        for j, other_box in enumerate(keep):
            if i != j:
                ox1, oy1, ox2, oy2, ocls, oscore = boxes_array[other_box]
                if calculate_iou((x1, y1, x2, y2), (ox1, oy1, ox2, oy2)) > 0.1:
                    if oscore > score:
                        is_valid = False
                        break
        if is_valid:
            final_boxes.append(box)

    return [filter_by_size_and_position(boxes_array[k], min_size, max_size, image_shape) for k in final_boxes if filter_by_size_and_position(boxes_array[k], min_size, max_size, image_shape) is not None]

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    
    xi1, yi1, xi2, yi2 = max(x1, x1p), max(y1, y1p), min(x2, x2p), min(y2, y2p)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2p - x1p) * (y2p - y1p)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def filter_by_size_and_position(box, min_size, max_size, image_shape):
    x1, y1, x2, y2, cls, score = box
    w, h = x2 - x1, y2 - y1

    # taille filtrage
    if w < min_size[0] or h < min_size[1] or w > max_size[0] or h > max_size[1]:
        return None

    # filtrage position (si image_shape est fourni)
    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        
        # threshold pour les detections trop proches
        border_threshold = 10
        if x1 < border_threshold or y1 < border_threshold or x2 > img_w - border_threshold or y2 > img_h - border_threshold:
            return None

        if y2 < img_h / 4:
            return None

        # Éliminer les détections de feux de circulation trop hautes
        if cls in [6, 7, 8]:
            if y1 < img_h / 2:
                return None

    return (int(x1), int(y1), int(w), int(h), int(cls), score)


def detect_roi(image, min_size=(30, 30), max_size=(300, 300)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    roi = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_size[0] <= w <= max_size[0] and min_size[1] <= h <= max_size[1]:
            roi.append((x, y, w, h))
    
    return roi

def classify_roi(image, roi, model, scaler):
    detections = []
    for x, y, w, h in roi:
        window = image[y:y+h, x:x+w]
        window_normalized = window.astype(np.float32) / 255.0
        window_gray = cv2.cvtColor(window_normalized, cv2.COLOR_BGR2GRAY)
        window_resized = cv2.resize(window_gray, (64, 64))
        features, _ = feature.hog(window_resized, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), block_norm='L2-Hys',
                                  visualize=True, transform_sqrt=True)
        features = features.reshape(1, -1)
        proba = model.predict_proba(features)[0]
        predicted_class = np.argmax(proba)
        if proba[predicted_class] >= 0.8 and predicted_class != 0:  # 0 est l'index pour "none"
            detections.append((x, y, w, h, predicted_class, proba[predicted_class]))
    
    return detections

def detect_panels(image, model, scaler):
    # Étape 1 : Détection rapide des ROI
    roi = detect_roi(image)
    
    # Étape 2 : Classification des ROI
    detections = classify_roi(image, roi, model, scaler)
    
    # Appliquer le NMS amélioré et le filtrage
    detections = improved_non_max_suppression(detections, overlap_thresh=0.3, min_size=(30, 30), max_size=(300, 300), image_shape=image.shape)
    
    return detections

def load_or_train_model(data, model_filename='test5_model.pkl'):
    if os.path.exists(model_filename):
        print(f"Loading existing model from {model_filename}")
        with open(model_filename, 'rb') as model_file:
            return pickle.load(model_file)
    else:
        print("Training new model...")
        X = np.asarray(data.get_features())
        Y = np.asarray(data.get_labels())

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
        clf = SVC(probability=True)
        clf.fit(X_train, Y_train)


        model_filename = 'svc_model.pkl'  # nom du fichier o le modèle sera sauvegardé
        with open(model_filename, 'wb') as model_file:
            pickle.dump(clf, model_file)
        
        print(f"Model saved to {model_filename}")
        return clf


def process_test_set(model, scaler, test_dir, output_file):
    print(f"Processing test set in directory: {test_dir}")
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        print(f"Found {len(image_files)} images to process")
        
        for image_file in image_files:
            print(f"Processing image: {image_file}")
            image_number = int(os.path.splitext(image_file)[0])
            image_path = os.path.join(test_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            boxes = detect_panels(image, model, scaler)
            
            print(f"Found {len(boxes)} detections for image {image_file}")
            
            for (x, y, w, h, predicted_class, confidence) in boxes:
                label_text = data.index_to_class(predicted_class)

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f"{label_text} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


                row = [image_number, x, y, x + w, y + h, confidence, label_text]
                csvwriter.writerow(row)
                print(f"Writing row: {row}")
            
            cv2.imshow('Detection', image)
            cv2.waitKey(3000)
            csvfile.flush()

    print(f"Finished processing. Check {output_file} for results.")

data = Dataset("dataset/train")

model = load_or_train_model(data)

test_dir = "dataset/test"
output_file = "detection.csv"

process_test_set(model, data, test_dir, output_file)

print(f"Détections enregistrées dans {output_file}")
