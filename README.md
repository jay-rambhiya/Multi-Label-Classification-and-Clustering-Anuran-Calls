# Multi-Label Classification and Clustering on Anuran Calls Dataset

This project focuses on multi-class and multi-label classification using Support Vector Machines (SVMs) and unsupervised clustering using k-means. The Anuran Calls dataset is used, featuring data from frog species with multiple labels: Family, Genus, and Species. This project is part of Homework 7 for the DSCI 552 course.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Multi-Label Classification with SVM](#multi-label-classification-with-svm)
- [K-Means Clustering](#k-means-clustering)
- [Requirements](#requirements)

## Project Overview
The objectives of this project are:
1. To perform multi-label classification on the Anuran Calls dataset using SVM with various kernels and regularization methods.
2. To apply k-means clustering to the data and analyze the cluster assignments with respect to the multi-label structure.
3. To evaluate models using Hamming distance, Hamming score, and loss, and analyze the clustering results with respect to label accuracy.

## Dataset
The dataset is the **Anuran Calls (MFCCs) Data Set** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29). It includes:
- Instances labeled by **Family**, **Genus**, and **Species**.
- Each instance consists of multiple features derived from MFCC (Mel Frequency Cepstral Coefficients) analysis.

## Multi-Label Classification with SVM
1. **Binary Relevance Approach**:
   - Trained a separate SVM for each label (Family, Genus, and Species).
   - Evaluated models with exact match and Hamming score/loss metrics.
2. **Gaussian Kernel and L1-Penalized SVM**:
   - Performed SVM classification with Gaussian kernels and L1-penalized SVMs.
   - Used 10-fold cross-validation to optimize the SVM penalty weight and kernel width.
3. **Class Imbalance Handling**:
   - Applied SMOTE to address class imbalance and repeated the classification tasks.
4. **Classifier Chain (Extra Practice)**:
   - Explored the Classifier Chain method for multi-label classification.

## K-Means Clustering
1. **Cluster Determination**:
   - Applied k-means clustering with an automatic selection of `k` (number of clusters) based on methods such as Silhouettes, Gap Statistics, or scree plots.
2. **Cluster Analysis**:
   - Determined the majority label in each cluster for Family, Genus, and Species.
3. **Hamming Metrics**:
   - Calculated Hamming distance, Hamming score, and Hamming loss for cluster assignments compared to true labels.

## Requirements
The project requires:
- Python
- Libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`, `imblearn`
