"""
Standalone runner script for Student Performance Prediction.
Runs the entire ML pipeline and saves evaluation figures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=" * 60)
    print("[*] Student Performance Prediction - ML Pipeline")
    print("=" * 60)

    # 1. Load Data
    data_path = os.path.join("data", "student-mat.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path, sep=";")
    print(f"Loaded dataset with {df.shape[0]} students and {df.shape[1]} features.")

    # 2. Check Missing Values
    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")

    # 3. Create Target
    df['Passed'] = (df['G3'] >= 10).astype(int)
    print(f"Pass/Fail distribution:\n{df['Passed'].value_counts()}\n")

    # 4. Features & Encoding
    X = df.drop(columns=['G3', 'Passed'])
    y = df['Passed']
    X = pd.get_dummies(X, drop_first=True)

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Model 1: Decision Tree
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)

    # 8. Model 2: KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_preds = knn.predict(X_test_scaled)
    knn_acc = accuracy_score(y_test, knn_preds)

    # Results
    print("-" * 60)
    print(f"Decision Tree Accuracy: {dt_acc * 100:.2f}%")
    print(f"K-Nearest Neighbors Accuracy: {knn_acc * 100:.2f}%")
    print("-" * 60)

    print("\nDecision Tree Classification Report:")
    print(classification_report(y_test, dt_preds, target_names=['Fail', 'Pass']))

    print("K-Nearest Neighbors Classification Report:")
    print(classification_report(y_test, knn_preds, target_names=['Fail', 'Pass']))

    # 9. Plot and Save Confusion Matrices
    os.makedirs("images", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(confusion_matrix(y_test, dt_preds), annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predicted Fail', 'Predicted Pass'],
                yticklabels=['Actual Fail', 'Actual Pass'])
    axes[0].set_title(f"Decision Tree Confusion Matrix\n(Accuracy: {dt_acc*100:.1f}%)", fontweight='bold')
    axes[0].set_ylabel("Actual Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(confusion_matrix(y_test, knn_preds), annot=True, fmt='d', cmap='Purples', ax=axes[1],
                xticklabels=['Predicted Fail', 'Predicted Pass'],
                yticklabels=['Actual Fail', 'Actual Pass'])
    axes[1].set_title(f"KNN Confusion Matrix\n(Accuracy: {knn_acc*100:.1f}%)", fontweight='bold')
    axes[1].set_ylabel("Actual Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plot_path = os.path.join("images", "confusion_matrix_comparison.png")
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved confusion matrix plot to: {plot_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
