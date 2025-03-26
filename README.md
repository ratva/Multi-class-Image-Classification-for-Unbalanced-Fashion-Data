# Multi-class Image Classification for Unbalanced Fashion Data

This project investigates the impact of class imbalance on multi-class classification using an MLP classifier trained on a modified subset of the FashionMNIST dataset. The goal was to evaluate different strategies for improving classification performance across underrepresented classes, especially in scenarios with extreme imbalance.

## Overview

Fashion categories such as "top" and "trouser" had as few as one training sample, while others like "sneaker" and "sandal" had 800. This imbalance introduced significant challenges in model training and evaluation, particularly in terms of generalization across all classes.

## Project Goals

- Understand the effects of class imbalance on MLP classifier performance.
- Compare training strategies using raw, duplicated, and augmented datasets.
- Tune hyperparameters using balanced accuracy as the primary evaluation metric.
- Improve classification performance, especially on minority classes.

## Dataset

- Based on FashionMNIST with 6 selected categories.
- Artificially skewed to simulate extreme class imbalance.
- Custom validation and test splits maintained proportional class distributions to better reflect population-level performance.
<img width=50% src=".projB_fashion6.png" alt="Grid of greyscale images of clothes">

## Methodology

### 1. MLP Classifier on Imbalanced Data

- Implemented a one-hidden-layer MLP classifier using scikit-learn.
- Used L-BFGS solver (chosen for smaller datasets).
- Performed multiple rounds of grid search on:
  - Hidden layer size
  - Maximum iterations (for early stopping)
  - L2 regularization term (`alpha`)
  - Activation functions (`relu`, `identity`)
- Employed custom cross-validation splitters to avoid data leakage while using the validation set for hyperparameter tuning.

### 2. Class Balancing via Duplication

- Uniform training set created by duplicating images to match the largest class (800 samples per class).
- Significant improvement in underrepresented class accuracy, but overfitting on duplicated images (especially for classes with only one original image) was observed.

### 3. Data Augmentation Approach

- Used the `skimage` library for image augmentations:
  - Random noise (e.g., Gaussian, salt-and-pepper)
  - Rotations (±10 degrees)
  - Gamma adjustments (brightening/darkening)
  - Flipping and combinations of transformations
- Generated 800 varied training samples for each class via augmentation.
- Normalized input images for this stage, enabling better use of image transformation functions.
- Grid search and final training repeated using the augmented dataset.

## Results

| Model | Balanced Accuracy (Test Set) |
|-------|-------------------------------|
| Imbalanced Data | ~0.733 |
| Duplicated Data | ~0.732 |
| Augmented Data  | **0.906** |

- The final model trained on the augmented dataset achieved the highest test performance.
- Balanced accuracy significantly improved for previously underperforming classes like "top" and "trouser."
- Slight performance degradation observed for the "sandal" class, likely due to overfitting on the "sneaker" class.

## Key Takeaways

- Class imbalance leads to model bias and misclassification of minority classes.
- Duplication helps but risks overfitting when original data is too limited.
- Augmentation is an effective way to increase diversity and improve performance for low-sample classes.

## Technologies Used

- Python
- Scikit-learn
- NumPy
- Scikit-image
- Jupyter Notebook

---

This project was completed as part of Tufts University's Fall 2023 Machine Learning course (COMP 135).
Group members - Avtar and Leigh
https://www.cs.tufts.edu/cs/135/2023f/projectB.html


