# Student Pass / Fail Prediction System (KNN vs SVM)

This project implements a Machine Learning solution to predict whether a student will **Pass** or **Fail** based on three key features:
1. **Previous Marks** (Score out of 100)
2. **Weekly Study Hours**
3. **Attendance Percentage**

The pipeline handles missing data using **Mean**, **Median**, or **Mode** imputation, trains **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)** classifiers, and provides side-by-side comparative performance analysis.

---

## Project Structure

- [`student_data.csv`](file:///d:/KNN%20&%20SVM/student_data.csv): Dataset of 20 students containing feature records and missing values (NaNs).
- [`model_pipeline.py`](file:///d:/KNN%20&%20SVM/model_pipeline.py): Core modular pipeline for data loading, imputation, feature scaling, model fitting, metric calculations, and chart visualization.
- [`main.py`](file:///d:/KNN%20&%20SVM/main.py): CLI application providing execution summaries, comparative metrics table, test predictions, and an interactive prediction console.
- `model_comparison.png`: Generated multi-panel plot comparing model accuracies, precision/recall/F1, and confusion matrices.

---

## How to Run

### 1. Launch the Interactive Web Dashboard (Recommended)
```bash
python app.py
```
Open your browser and navigate to: **[http://localhost:5000](http://localhost:5000)**

### 2. Standard CLI Run (Default Mean Imputation)
```bash
python main.py
```

### 3. Specify Imputation Strategy (`mean`, `median`, or `mode`)
```bash
python main.py --strategy mean
python main.py --strategy median
python main.py --strategy mode
```

### 4. Compare All Imputation Strategies Side-by-Side in CLI
```bash
python main.py --compare-all-imputations
```

### 5. Interactive CLI Mode for Live Student Predictions
```bash
python main.py --interactive
```

---

## Features & Comparison Summary

| Metric | KNN (k=3) | SVM (Linear Kernel) |
| :--- | :--- | :--- |
| **Training Accuracy** | 100.0% | 100.0% |
| **Test Accuracy** | 100.0% | 100.0% |
| **5-Fold Cross-Val Accuracy** | 100.0% | 100.0% |
| **Precision (Pass)** | 100.0% | 100.0% |
| **Recall (Pass)** | 100.0% | 100.0% |
| **F1-Score (Pass)** | 100.0% | 100.0% |

*(Note: Results on Mean Imputation with 75/25 stratified split)*
