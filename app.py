"""
app.py
======
Flask Web Application for Student Pass/Fail Prediction.
Provides interactive web dashboard and REST API for:
  - Dataset inspection & missing value imputation (Mean/Median/Mode)
  - Side-by-side KNN & SVM model evaluation and metrics
  - Real-time student pass/fail predictions with confidence scores
  - Multi-imputation strategy comparison
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os

from model_pipeline import (
    load_dataset,
    get_missing_summary,
    impute_data,
    prepare_features,
    train_and_compare_models,
    predict_single_student,
)

app = Flask(__name__)
CSV_PATH = os.path.join(os.path.dirname(__file__), "student_data.csv")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data", methods=["GET"])
def get_data():
    """Return dataset overview, missing summary, and imputed data."""
    strategy = request.args.get("strategy", "mean").lower()
    if strategy not in ["mean", "median", "mode"]:
        strategy = "mean"

    raw_df = load_dataset(CSV_PATH)
    missing_summary = get_missing_summary(raw_df)
    imputed_df, imputed_values = impute_data(raw_df, strategy=strategy)

    # Convert to list of records with missing flags
    raw_records = []
    for idx, row in raw_df.iterrows():
        raw_records.append({
            "Student_ID": int(row["Student_ID"]),
            "Previous_Marks": None if pd.isna(row["Previous_Marks"]) else float(row["Previous_Marks"]),
            "Study_Hours": None if pd.isna(row["Study_Hours"]) else float(row["Study_Hours"]),
            "Attendance": None if pd.isna(row["Attendance"]) else float(row["Attendance"]),
            "Result": str(row["Result"]),
            "has_missing": bool(row.isna().any()),
        })

    imputed_records = imputed_df.to_dict(orient="records")

    return jsonify({
        "strategy": strategy,
        "missing_summary": missing_summary,
        "imputed_values": imputed_values,
        "raw_records": raw_records,
        "imputed_records": imputed_records,
        "total_students": len(raw_df),
    })


@app.route("/api/models", methods=["GET"])
def get_model_evaluation():
    """Train models and return comparative evaluation metrics."""
    strategy = request.args.get("strategy", "mean").lower()
    k_neighbors = int(request.args.get("k", 3))
    svm_kernel = request.args.get("kernel", "linear")

    raw_df = load_dataset(CSV_PATH)
    imputed_df, imputed_values = impute_data(raw_df, strategy=strategy)
    X, y, _ = prepare_features(imputed_df)

    results = train_and_compare_models(
        X,
        y,
        test_size=0.25,
        random_state=42,
        knn_neighbors=k_neighbors,
        svm_kernel=svm_kernel,
    )

    knn_m = results["knn_metrics"]
    svm_m = results["svm_metrics"]

    def format_metrics(m):
        return {
            "train_accuracy": round(float(m["train_accuracy"]) * 100, 2),
            "test_accuracy": round(float(m["test_accuracy"]) * 100, 2),
            "cv_accuracy_mean": round(float(m["cv_accuracy_mean"]) * 100, 2),
            "cv_accuracy_std": round(float(m["cv_accuracy_std"]) * 100, 2),
            "precision": round(float(m["precision"]) * 100, 2),
            "recall": round(float(m["recall"]) * 100, 2),
            "f1_score": round(float(m["f1_score"]) * 100, 2),
            "confusion_matrix": m["confusion_matrix"].tolist(),
        }

    return jsonify({
        "strategy": strategy,
        "k_neighbors": k_neighbors,
        "svm_kernel": svm_kernel,
        "imputed_values": imputed_values,
        "knn": format_metrics(knn_m),
        "svm": format_metrics(svm_m),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict Pass/Fail for custom student inputs using both KNN & SVM."""
    data = request.get_json(force=True)
    try:
        marks = float(data.get("marks", 0))
        hours = float(data.get("hours", 0))
        attendance = float(data.get("attendance", 0))
        strategy = data.get("strategy", "mean").lower()
        k_neighbors = int(data.get("k", 3))
        svm_kernel = data.get("kernel", "linear")

        raw_df = load_dataset(CSV_PATH)
        imputed_df, _ = impute_data(raw_df, strategy=strategy)
        X, y, _ = prepare_features(imputed_df)

        results = train_and_compare_models(
            X,
            y,
            test_size=0.25,
            random_state=42,
            knn_neighbors=k_neighbors,
            svm_kernel=svm_kernel,
        )

        pred_res = predict_single_student(
            results["knn_model"],
            results["svm_model"],
            results["scaler"],
            marks,
            hours,
            attendance,
        )

        return jsonify({
            "success": True,
            "inputs": {
                "marks": marks,
                "hours": hours,
                "attendance": attendance,
            },
            "knn": {
                "prediction": pred_res["knn"]["prediction"],
                "pass_probability": round(float(pred_res["knn"]["probability_pass"]) * 100, 1),
                "fail_probability": round(float(pred_res["knn"]["probability_fail"]) * 100, 1),
            },
            "svm": {
                "prediction": pred_res["svm"]["prediction"],
                "pass_probability": round(float(pred_res["svm"]["probability_pass"]) * 100, 1),
                "fail_probability": round(float(pred_res["svm"]["probability_fail"]) * 100, 1),
            },
            "agreement": pred_res["knn"]["prediction"] == pred_res["svm"]["prediction"],
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/compare-imputations", methods=["GET"])
def compare_imputations():
    """Return comparative metrics across Mean, Median, and Mode strategies."""
    raw_df = load_dataset(CSV_PATH)
    strategies = ["mean", "median", "mode"]
    summary = []

    for strat in strategies:
        imputed_df, imp_vals = impute_data(raw_df, strategy=strat)
        X, y, _ = prepare_features(imputed_df)
        res = train_and_compare_models(X, y, test_size=0.25, random_state=42, knn_neighbors=3)

        knn_m = res["knn_metrics"]
        svm_m = res["svm_metrics"]

        summary.append({
            "strategy": strat.capitalize(),
            "imputed_marks": imp_vals["Previous_Marks"],
            "imputed_hours": imp_vals["Study_Hours"],
            "imputed_attendance": imp_vals["Attendance"],
            "knn_test_acc": round(float(knn_m["test_accuracy"]) * 100, 1),
            "knn_cv_acc": round(float(knn_m["cv_accuracy_mean"]) * 100, 1),
            "svm_test_acc": round(float(svm_m["test_accuracy"]) * 100, 1),
            "svm_cv_acc": round(float(svm_m["cv_accuracy_mean"]) * 100, 1),
        })

    return jsonify({"imputation_comparison": summary})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Student Pass/Fail Prediction Web Application on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
