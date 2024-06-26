# Detecting Traffic Signs Using Two Different Methods

## Introduction

This project done as part of a course at the UTC implements traffic sign detection and classification algorithms as part of the SY32 project at UTC. The goal is to develop a system that can reliably identify different types of traffic signs from camera images, including warning signs, speed limits, and traffic lights.

Two technical approaches were explored:

1. Classical computer vision methods: Extracting visual features from images and using machine learning models to classify signs.
2. Deep learning techniques: Using artificial neural networks to analyze images in a more complex way.

The work involved experimentation and analysis to understand the strengths and weaknesses of each method, using metrics like precision and recall to measure performance.

## Dataset

The initial dataset consisted of 703 annotated images of traffic signs and 87 photos for validation. However, quality issues with annotations were identified and corrected manually. The final dataset used was a collectively corrected version of higher quality.

## Classical Model

### Data Preprocessing

- Images were loaded using an OOP approach with a Dataset class.
- Normalization was applied to adjust pixel values.
- HOG (Histogram of Oriented Gradients) method was used for feature extraction.

### Model Training

An SVM classifier was chosen and implemented using scikit-learn's SVC class.

### Panel Detection Algorithm

A two-stage detection process was implemented:

1. Fast Region of Interest (ROI) detection:
   - Convert image to grayscale
   - Apply Gaussian blur
   - Use Canny edge detection
   - Extract and filter contours

2. Detailed classification:
   - Apply trained SVM model to each ROI

An improved Non-Maximum Suppression (NMS) algorithm was used to refine results.

### Performance Analysis

Results on the UV test website:
- AP: 82.01
- AR: 16.15
- mAP: 56.91
- mAR: 14.77

The algorithm performed well on certain sign types but struggled with others, particularly traffic lights.

## Deep Learning Model

### YOLOv1 Implementation

- Modified YOLOv1 with adjusted parameters (S=12, C=7)
- Input size changed to 576x576
- Network layers decreased in size to reduce parameters

### Data Augmentation

Used albumentations framework for:
- Random rotations (+/- 15 degrees)
- Horizontal flips
- Random crops

### Training Process

- Batches of 30-100 epochs
- Dynamic learning rate adjustment
- Best model achieved at epoch 250

### Results

- mAP on test dataset: 0.4944

### Potential Improvements

1. Improved re-scaling of images
2. Addressing over-fitting
3. Further optimization of data augmentation

## Conclusion

Both methods achieved partial success:
- Classical approach: AP of 82% and mAP of 57%
- Deep learning approach: mAP of 49%

There is significant room for improvement in both approaches to increase accuracy.
