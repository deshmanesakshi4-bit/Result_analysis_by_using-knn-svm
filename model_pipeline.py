"""
model_pipeline.py
=================
Core Machine Learning Pipeline for Student Pass/Fail Prediction.
Implements:
  - Missing value imputation (Mean, Median, Mode)
  - Feature Scaling
  - K-Nearest Neighbors (KNN) & Support Vector Machine (SVM) training
  - Comprehensive model evaluation and side-by-side comparison
  - Visualizations (Bar charts, Confusion Matrices)
  - Single/Batch student predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def load_dataset(csv_path="student_data.csv"):
    """Load the raw dataset from a CSV file."""
    return pd.read_csv(csv_path)


def get_missing_summary(df):
    """Summarize missing values per column."""
    features = ["Previous_Marks", "Study_Hours", "Attendance"]
    missing_info = {}
    for col in features:
        count = df[col].isnull().sum()
        pct = (count / len(df)) * 100
        missing_info[col] = {"missing_count": int(count), "missing_pct": pct}
    return missing_info


def impute_data(df, strategy="mean"):
    """
    Impute missing values using Mean, Median, or Mode.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with potential missing values.
    strategy : str, default='mean'
        One of 'mean', 'median', 'mode'.
        
    Returns:
    --------
    imputed_df : pd.DataFrame
        Dataframe with imputed missing values.
    imputed_values : dict
        Mapping of feature -> value used for imputation.
    """
    strategy = strategy.lower().strip()
    imputed_df = df.copy()
    features = ["Previous_Marks", "Study_Hours", "Attendance"]
    imputed_values = {}

    for col in features:
        if strategy == "mean":
            val = round(imputed_df[col].mean(), 2)
        elif strategy == "median":
            val = round(imputed_df[col].median(), 2)
        elif strategy == "mode":
            mode_series = imputed_df[col].mode()
            val = round(mode_series.iloc[0], 2) if not mode_series.empty else 0.0
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Supported: 'mean', 'median', 'mode'.")

        imputed_values[col] = val
        imputed_df[col] = imputed_df[col].fillna(val)

    return imputed_df, imputed_values


def prepare_features(df):
    """
    Extract features X and binary target y (1 for Pass, 0 for Fail).
    """
    feature_cols = ["Previous_Marks", "Study_Hours", "Attendance"]
    X = df[feature_cols].copy()
    
    # Reliably map target: Pass / '1' -> 1, Fail / '0' -> 0
    y = np.array([
        1 if str(val).strip().lower() in ["pass", "1"] else 0
        for val in df["Result"]
    ], dtype=int)

    return X, y, feature_cols


def train_and_compare_models(
    X,
    y,
    test_size=0.25,
    random_state=42,
    knn_neighbors=3,
    svm_kernel="linear",
    svm_C=1.0,
):
    """
    Train and evaluate both KNN and SVM models on the dataset.
    
    Returns:
    --------
    results : dict
        Contains models, scaler, train/test metrics, cross-validation scores,
        confusion matrices, and test splits.
    """
    # 1. Train-test split (Stratified to maintain Pass/Fail balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2. Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X)

    # 3. Train KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors, metric="euclidean")
    knn.fit(X_train_scaled, y_train)

    # 4. Train SVM Classifier
    svm = SVC(kernel=svm_kernel, C=svm_C, probability=True, random_state=random_state)
    svm.fit(X_train_scaled, y_train)

    # 5. Predictions
    knn_train_pred = knn.predict(X_train_scaled)
    knn_test_pred = knn.predict(X_test_scaled)

    svm_train_pred = svm.predict(X_train_scaled)
    svm_test_pred = svm.predict(X_test_scaled)

    # 6. Cross-Validation (Stratified 5-Fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    knn_cv_scores = cross_val_score(knn, X_full_scaled, y, cv=cv, scoring="accuracy")
    svm_cv_scores = cross_val_score(svm, X_full_scaled, y, cv=cv, scoring="accuracy")

    # 7. Calculate Metrics
    def compute_metrics(y_true, y_pred, y_train_true, y_train_pred, cv_scores):
        return {
            "train_accuracy": accuracy_score(y_train_true, y_train_pred),
            "test_accuracy": accuracy_score(y_true, y_pred),
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
            "classification_report": classification_report(
                y_true, y_pred, target_names=["Fail (0)", "Pass (1)"], zero_division=0
            ),
        }

    knn_metrics = compute_metrics(
        y_test, knn_test_pred, y_train, knn_train_pred, knn_cv_scores
    )
    svm_metrics = compute_metrics(
        y_test, svm_test_pred, y_train, svm_train_pred, svm_cv_scores
    )

    results = {
        "knn_model": knn,
        "svm_model": svm,
        "scaler": scaler,
        "knn_metrics": knn_metrics,
        "svm_metrics": svm_metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "knn_test_pred": knn_test_pred,
        "svm_test_pred": svm_test_pred,
        "params": {
            "knn_neighbors": knn_neighbors,
            "svm_kernel": svm_kernel,
            "test_size": test_size,
            "random_state": random_state,
        },
    }

    return results


def plot_comparison(results, output_path="model_comparison.png"):
    """
    Generate and save a visual comparison of KNN vs SVM.
    Includes:
      - Accuracy Comparison (Train, Test, 5-Fold CV)
      - Performance Metrics (Precision, Recall, F1-Score)
      - Side-by-side Confusion Matrices
    """
    knn_m = results["knn_metrics"]
    svm_m = results["svm_metrics"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Student Pass/Fail Prediction: KNN vs SVM Comparative Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # 1. Accuracy Comparison Bar Chart
    ax1 = axes[0, 0]
    categories = ["Training Accuracy", "Test Accuracy", "5-Fold CV Accuracy"]
    knn_acc_vals = [
        knn_m["train_accuracy"] * 100,
        knn_m["test_accuracy"] * 100,
        knn_m["cv_accuracy_mean"] * 100,
    ]
    svm_acc_vals = [
        svm_m["train_accuracy"] * 100,
        svm_m["test_accuracy"] * 100,
        svm_m["cv_accuracy_mean"] * 100,
    ]

    x = np.arange(len(categories))
    width = 0.32

    rects1 = ax1.bar(x - width / 2, knn_acc_vals, width, label="KNN (k=3)", color="#3498db")
    rects2 = ax1.bar(x + width / 2, svm_acc_vals, width, label="SVM (Linear)", color="#e74c3c")

    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_title("Accuracy Metrics Comparison", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontweight="medium")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper left")

    # Add data labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # 2. Classification Metrics (Precision, Recall, F1)
    ax2 = axes[0, 1]
    metric_names = ["Precision", "Recall", "F1-Score"]
    knn_class_vals = [
        knn_m["precision"] * 100,
        knn_m["recall"] * 100,
        knn_m["f1_score"] * 100,
    ]
    svm_class_vals = [
        svm_m["precision"] * 100,
        svm_m["recall"] * 100,
        svm_m["f1_score"] * 100,
    ]

    rects3 = ax2.bar(x - width / 2, knn_class_vals, width, label="KNN (k=3)", color="#2ecc71")
    rects4 = ax2.bar(x + width / 2, svm_class_vals, width, label="SVM (Linear)", color="#9b59b6")

    ax2.set_ylabel("Score (%)", fontweight="bold")
    ax2.set_title("Precision, Recall & F1-Score Comparison", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names, fontweight="medium")
    ax2.set_ylim(0, 115)
    ax2.legend(loc="upper left")

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # 3. KNN Confusion Matrix Heatmap
    ax3 = axes[1, 0]
    sns.heatmap(
        knn_m["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Fail (0)", "Pass (1)"],
        yticklabels=["Fail (0)", "Pass (1)"],
        ax=ax3,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax3.set_title(f"KNN Confusion Matrix (Test Acc: {knn_m['test_accuracy']*100:.1f}%)", fontweight="bold")
    ax3.set_xlabel("Predicted Label", fontweight="bold")
    ax3.set_ylabel("Actual Label", fontweight="bold")

    # 4. SVM Confusion Matrix Heatmap
    ax4 = axes[1, 1]
    sns.heatmap(
        svm_m["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Reds",
        cbar=False,
        xticklabels=["Fail (0)", "Pass (1)"],
        yticklabels=["Fail (0)", "Pass (1)"],
        ax=ax4,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax4.set_title(f"SVM Confusion Matrix (Test Acc: {svm_m['test_accuracy']*100:.1f}%)", fontweight="bold")
    ax4.set_xlabel("Predicted Label", fontweight="bold")
    ax4.set_ylabel("Actual Label", fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def predict_single_student(knn_model, svm_model, scaler, previous_marks, study_hours, attendance):
    """
    Predict Pass/Fail for a single student given raw input features.
    """
    feature_cols = ["Previous_Marks", "Study_Hours", "Attendance"]
    input_df = pd.DataFrame(
        [[previous_marks, study_hours, attendance]], columns=feature_cols
    )
    input_scaled = scaler.transform(input_df)

    knn_pred = knn_model.predict(input_scaled)[0]
    knn_proba = knn_model.predict_proba(input_scaled)[0]

    svm_pred = svm_model.predict(input_scaled)[0]
    svm_proba = svm_model.predict_proba(input_scaled)[0]

    label_map = {1: "Pass", 0: "Fail"}

    return {
        "input": {
            "Previous_Marks": previous_marks,
            "Study_Hours": study_hours,
            "Attendance": attendance,
        },
        "knn": {
            "prediction": label_map[knn_pred],
            "probability_pass": knn_proba[1],
            "probability_fail": knn_proba[0],
        },
        "svm": {
            "prediction": label_map[svm_pred],
            "probability_pass": svm_proba[1],
            "probability_fail": svm_proba[0],
        },
    }
