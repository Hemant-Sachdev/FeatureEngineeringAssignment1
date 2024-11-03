# Feature Engineering Assignment 1 - Anomaly Detection

## Overview
This repository contains the implementation of various anomaly detection methods for identifying outliers in a thyroid disease dataset. The methods used include Z-Score, Mahalanobis Distance, Local Outlier Factor (LOF), Isolation Forest, and One-Class SVM. The assignment demonstrates the use of feature scaling, anomaly detection, and evaluation of the results using performance metrics and visualization.

## Setup and Imports
The project uses the following Python libraries:
- `pandas` and `numpy` for data manipulation
- `scikit-learn` for preprocessing and anomaly detection models
- `scipy.stats` for statistical calculations
- `matplotlib` and `seaborn` for data visualization

## Dataset
The dataset used is the **Thyroid Disease Dataset** from the UCI Machine Learning Repository. It consists of 21 features and a target variable indicating whether a data point is normal (1) or not (0). The data is loaded from a URL and preprocessed for anomaly detection.

- **URL**: [Thyroid Disease Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data)
- **Columns**: The dataset includes a `target` column and 21 feature columns (`feature_1` to `feature_21`).

## Data Preprocessing
The feature values are standardized using `StandardScaler` from `scikit-learn` to ensure they have a mean of 0 and a standard deviation of 1.

## Anomaly Detection Methods
The following methods are implemented to detect anomalies:

### a) Z-Score Based Anomaly Detection
- Outliers are identified if their z-score exceeds 3 in absolute value.
- **Results**: Detected 1195 outliers.

### b) Mahalanobis Distance-Based Anomaly Detection
- Anomalies are detected using the `EllipticEnvelope` model with a contamination level of 5%.
- **Results**: Detected 189 outliers.

### c) Local Outlier Factor (LOF)
- Uses the `LocalOutlierFactor` method with 20 neighbors and 5% contamination.
- **Results**: Detected 189 outliers.

### d) Isolation Forest
- Implements an `IsolationForest` with 5% contamination and a random seed for reproducibility.
- **Results**: Detected 189 outliers.

### e) One-Class SVM
- Uses `OneClassSVM` with an RBF kernel and 5% contamination.
- **Results**: Detected 186 outliers.

## Evaluation and Comparison
Each method's performance is evaluated using:
- **Classification Report**: Displays precision, recall, and F1-score.
- **Confusion Matrix**: Summarizes true and false positives and negatives.

The comparison reveals that Z-Score detection results in a higher number of outliers, while the other methods (Mahalanobis, LOF, Isolation Forest, One-Class SVM) are more consistent.

## Visualization
The dataset is visualized using PCA for dimensionality reduction, and anomalies are plotted with scatter plots. The visualization helps to understand how each method identifies anomalies in the feature space.

### Visualization Setup
- **PCA**: Reduces features to two components for easy plotting.
- **Seaborn**: Used to create scatter plots for each anomaly detection method.

## Results
The scatter plots illustrate the differences in how anomalies are identified. Each subplot shows the distribution of data points classified as normal or anomalous.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/feature-engineering-assignment1.git
   ```
2. Navigate to the project directory and install dependencies (if necessary):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python feature_engineering_assignment1.py
   ```

## Dependencies
- Python 3.6+
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn
