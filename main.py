"""
main.py
=======
Main entry point for Student Pass/Fail Prediction.
Demonstrates:
  1. Loading 20 students' dataset with missing values
  2. Imputation using Mean / Median / Mode
  3. Training KNN and SVM models separately
  4. Side-by-side comparison of results and performance metrics
  5. Generating visualization charts
  6. Testing and predicting outcomes for new student data
"""

import sys
import argparse
import pandas as pd
from model_pipeline import (
    load_dataset,
    get_missing_summary,
    impute_data,
    prepare_features,
    train_and_compare_models,
    plot_comparison,
    predict_single_student,
)


def print_banner():
    print("=" * 80)
    print("      STUDENT PASS / FAIL PREDICTION SYSTEM USING KNN & SVM")
    print("=" * 80)


def display_dataset_overview(raw_df, missing_summary):
    print("\n[STEP 1] DATASET OVERVIEW (20 Students Past Data):")
    print("-" * 80)
    print(raw_df.to_string(index=False))
    print("-" * 80)
    print("Missing Values Summary:")
    for feature, info in missing_summary.items():
        print(f"  - {feature:<16}: {info['missing_count']} missing ({info['missing_pct']:.1f}%)")


def display_imputation_summary(imputed_df, imputed_values, strategy):
    print(f"\n[STEP 2] MISSING DATA IMPUTATION (Strategy: '{strategy.upper()}'):")
    print("-" * 80)
    print("Imputed Values Used:")
    for col, val in imputed_values.items():
        print(f"  - {col:<16}: {val}")
    print("\nDataset after imputation (First 10 records):")
    print(imputed_df.head(10).to_string(index=False))
    print(f"Total remaining missing values: {imputed_df.isnull().sum().sum()}")


def display_comparative_results(results):
    knn_m = results["knn_metrics"]
    svm_m = results["svm_metrics"]

    print("\n" + "=" * 80)
    print("         MODEL PERFORMANCE COMPARISON (KNN vs SVM)")
    print("=" * 80)

    header = f"{'Metric':<28} | {'KNN (k=3)':<18} | {'SVM (Linear)':<18} | {'Status/Leader':<14}"
    divider = "-" * len(header)

    print(header)
    print(divider)

    def row(label, k_val, s_val, is_pct=True):
        if is_pct:
            k_str = f"{k_val * 100:.2f}%"
            s_str = f"{s_val * 100:.2f}%"
            diff = (s_val - k_val) * 100
            if abs(diff) < 0.01:
                leader = "Tied"
            elif diff > 0:
                leader = f"SVM (+{diff:.1f}%)"
            else:
                leader = f"KNN (+{-diff:.1f}%)"
        else:
            k_str = f"{k_val}"
            s_str = f"{s_val}"
            leader = "-"
        return f"{label:<28} | {k_str:<18} | {s_str:<18} | {leader:<14}"

    print(row("Training Accuracy", knn_m["train_accuracy"], svm_m["train_accuracy"]))
    print(row("Test Accuracy", knn_m["test_accuracy"], svm_m["test_accuracy"]))
    
    cv_k_str = f"{knn_m['cv_accuracy_mean']*100:.1f}% (+/- {knn_m['cv_accuracy_std']*100:.1f}%)"
    cv_s_str = f"{svm_m['cv_accuracy_mean']*100:.1f}% (+/- {svm_m['cv_accuracy_std']*100:.1f}%)"
    cv_diff = (svm_m['cv_accuracy_mean'] - knn_m['cv_accuracy_mean']) * 100
    cv_leader = "Tied" if abs(cv_diff) < 0.01 else (f"SVM (+{cv_diff:.1f}%)" if cv_diff > 0 else f"KNN (+{-cv_diff:.1f}%)")
    print(f"{'5-Fold Cross-Val Accuracy':<28} | {cv_k_str:<18} | {cv_s_str:<18} | {cv_leader:<14}")

    print(row("Precision (Pass)", knn_m["precision"], svm_m["precision"]))
    print(row("Recall (Pass)", knn_m["recall"], svm_m["recall"]))
    print(row("F1-Score (Pass)", knn_m["f1_score"], svm_m["f1_score"]))
    print(divider)

    print("\nCONFUSION MATRIX COMPARISON:")
    print("-" * 80)
    print("KNN Confusion Matrix:")
    print(f"  [[TN={knn_m['confusion_matrix'][0,0]}, FP={knn_m['confusion_matrix'][0,1]}],")
    print(f"   [FN={knn_m['confusion_matrix'][1,0]}, TP={knn_m['confusion_matrix'][1,1]}]]")
    print("\nSVM Confusion Matrix:")
    print(f"  [[TN={svm_m['confusion_matrix'][0,0]}, FP={svm_m['confusion_matrix'][0,1]}],")
    print(f"   [FN={svm_m['confusion_matrix'][1,0]}, TP={svm_m['confusion_matrix'][1,1]}]]")
    print("-" * 80)


def display_sample_predictions(results):
    print("\n[STEP 4] SAMPLE TEST PREDICTIONS ON NEW STUDENTS:")
    print("-" * 80)

    sample_students = [
        {"name": "Student A (High Achiever)", "marks": 85, "hours": 14, "attendance": 90},
        {"name": "Student B (Borderline)", "marks": 50, "hours": 5, "attendance": 65},
        {"name": "Student C (At Risk)", "marks": 35, "hours": 2, "attendance": 45},
    ]

    for s in sample_students:
        pred = predict_single_student(
            results["knn_model"],
            results["svm_model"],
            results["scaler"],
            s["marks"],
            s["hours"],
            s["attendance"],
        )
        print(f"\nProfile: {s['name']}")
        print(f"  Inputs: Marks={s['marks']}, Study Hours={s['hours']}h/wk, Attendance={s['attendance']}%")
        print(f"  -> KNN Prediction: {pred['knn']['prediction']} (Pass Prob: {pred['knn']['probability_pass']*100:.1f}%, Fail Prob: {pred['knn']['probability_fail']*100:.1f}%)")
        print(f"  -> SVM Prediction: {pred['svm']['prediction']} (Pass Prob: {pred['svm']['probability_pass']*100:.1f}%, Fail Prob: {pred['svm']['probability_fail']*100:.1f}%)")


def run_pipeline(csv_path="student_data.csv", strategy="mean", generate_plots=True):
    print_banner()

    # 1. Load Data
    raw_df = load_dataset(csv_path)
    missing_summary = get_missing_summary(raw_df)
    display_dataset_overview(raw_df, missing_summary)

    # 2. Impute Data
    imputed_df, imputed_values = impute_data(raw_df, strategy=strategy)
    display_imputation_summary(imputed_df, imputed_values, strategy)

    # 3. Train & Evaluate Models
    X, y, _ = prepare_features(imputed_df)
    results = train_and_compare_models(X, y, test_size=0.25, random_state=42, knn_neighbors=3)

    # 4. Display Comparison
    display_comparative_results(results)

    # 5. Visualizations
    if generate_plots:
        img_file = plot_comparison(results, output_path="model_comparison.png")
        print(f"\n[STEP 3] Visualization charts generated & saved to: {img_file}")

    # 6. Sample Predictions
    display_sample_predictions(results)

    return results


def interactive_mode(results):
    print("\n" + "=" * 80)
    print("                 INTERACTIVE STUDENT PREDICTION MODE")
    print("=" * 80)
    print("Enter student details to predict Pass/Fail (or type 'exit' to quit):")

    while True:
        try:
            val = input("\nEnter Previous Marks (0-100) [or 'exit']: ").strip()
            if val.lower() == "exit":
                break
            marks = float(val)

            val = input("Enter Weekly Study Hours: ").strip()
            if val.lower() == "exit":
                break
            hours = float(val)

            val = input("Enter Attendance Percentage (0-100): ").strip()
            if val.lower() == "exit":
                break
            attendance = float(val)

            pred = predict_single_student(
                results["knn_model"],
                results["svm_model"],
                results["scaler"],
                marks,
                hours,
                attendance,
            )

            print("\n" + "-" * 50)
            print(" PREDICTION RESULT:")
            print("-" * 50)
            print(f"  KNN Prediction : {pred['knn']['prediction'].upper():<4} (Confidence: {max(pred['knn']['probability_pass'], pred['knn']['probability_fail'])*100:.1f}%)")
            print(f"  SVM Prediction : {pred['svm']['prediction'].upper():<4} (Confidence: {max(pred['svm']['probability_pass'], pred['svm']['probability_fail'])*100:.1f}%)")
            print("-" * 50)

        except ValueError:
            print("Invalid input! Please enter numerical values.")
        except (KeyboardInterrupt, EOFError):
            break

def compare_all_imputations(csv_path="student_data.csv"):
    """
    Run pipeline across Mean, Median, and Mode imputation strategies
    and print a comparison table.
    """
    print_banner()
    print("\n" + "=" * 80)
    print("      COMPARING IMPUTATION STRATEGIES: MEAN vs MEDIAN vs MODE")
    print("=" * 80)

    raw_df = load_dataset(csv_path)
    strategies = ["mean", "median", "mode"]
    summary_rows = []

    for strat in strategies:
        imputed_df, imp_vals = impute_data(raw_df, strategy=strat)
        X, y, _ = prepare_features(imputed_df)
        res = train_and_compare_models(X, y, test_size=0.25, random_state=42, knn_neighbors=3)

        knn_m = res["knn_metrics"]
        svm_m = res["svm_metrics"]

        summary_rows.append({
            "Strategy": strat.capitalize(),
            "Imputed Values (Marks/Hours/Att)": f"{imp_vals['Previous_Marks']:.1f} / {imp_vals['Study_Hours']:.1f} / {imp_vals['Attendance']:.1f}",
            "KNN Test Acc": f"{knn_m['test_accuracy']*100:.1f}%",
            "KNN 5-Fold CV": f"{knn_m['cv_accuracy_mean']*100:.1f}%",
            "SVM Test Acc": f"{svm_m['test_accuracy']*100:.1f}%",
            "SVM 5-Fold CV": f"{svm_m['cv_accuracy_mean']*100:.1f}%",
        })

    comp_df = pd.DataFrame(summary_rows)
    print("\n" + comp_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Pass/Fail Prediction (KNN & SVM)")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["mean", "median", "mode"],
        default="mean",
        help="Imputation strategy for missing values (mean, median, mode)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive mode to test custom student inputs",
    )
    parser.add_argument(
        "--compare-all-imputations",
        action="store_true",
        help="Run and compare Mean, Median, and Mode imputation strategies side by side",
    )
    args = parser.parse_args()

    if args.compare_all_imputations:
        compare_all_imputations()
    else:
        results = run_pipeline(strategy=args.strategy)

        if args.interactive:
            interactive_mode(results)
