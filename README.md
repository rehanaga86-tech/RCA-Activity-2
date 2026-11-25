# Activity 2: Binary Classification with Logistic Regression (Pass / Fail)

**Course:** Regression and Classification Algorithms (CIE 2 — Activity 2)  
**Student:** Rehan Altaf Aga (`AD3101`)  

---

## Executive Summary

This repository presents a **Logistic Regression** binary classification model trained on `student_exam_performance.csv` to predict whether a student **Passes (`1`)** or **Fails (`0`)**. Instead of fitting a continuous regression line to a discrete outcome, the model computes a linear log-odds score and applies the sigmoid function to estimate $P(\text{Pass} = 1)$, converting probabilities into class predictions through an explicit decision threshold.

---

## Dataset & Problem Framing

* **Dataset File:** `student_exam_performance.csv` (`100,000` rows, `44` columns)
* **Dataset Source:** `[Add your dataset source link or citation here]`
* **Target Variable (`pass_status`):**
  * **Positive Class (`1` — `Pass`):** `77,389` records (`77.39%`)
  * **Negative Class (`0` — `Fail`):** `22,611` records (`22.61%`)
* **Feature Selection & Leakage Control:** Removed `student_id` (identifier) as well as direct exam-outcome and post-exam columns (`exam_score`, `performance_grade`, `performance_level`, `questions_attempted`, and `questions_correct`) so the model learns from pre-exam study habits, attendance, preparation, and background features rather than leaking the exam result.

---

## Modeling & Validation Pipeline

* **Data Hygiene & Split:** Removed duplicate records and applied an `80:20` stratified train-test split (`random_state=42`) so both partitions preserve the original `Pass`/`Fail` class distribution.
* **Leak-Free Preprocessing:** Used a `ColumnTransformer` with median imputation and `StandardScaler` for numerical features, and most-frequent imputation with `OneHotEncoder(drop="first", handle_unknown="ignore")` for categorical features—fitted strictly on the training set.
* **Baseline & Classifier:** Trained an L2-regularized `LogisticRegression` classifier (`max_iter=1000`) and benchmarked it against a `DummyClassifier(strategy="most_frequent")` majority-class baseline (`0.7739` test accuracy).
* **Diagnostics & Thresholds:** Evaluated test performance via confusion matrix counts (`TN`, `FP`, `FN`, `TP`), accuracy, precision, recall (sensitivity), specificity, F1-score, ROC curve / ROC-AUC, precision-recall curve, and threshold analysis at `0.3`, `0.5`, and `0.7`.

---

## Key Results (Test Set @ Threshold `0.5`)

At the `0.5` threshold, the Logistic Regression model reaches a test accuracy of `0.8473`, precision of `0.8758`, recall of `0.9354`, F1-score of `0.9046`, and ROC-AUC of `0.8922`, beating the `0.7739` majority-class baseline. I recommend the `0.5` threshold for the strongest overall precision-recall balance (highest F1-score and accuracy), using `0.7` when higher precision (`0.9213`) is needed or `0.3` when higher recall (`0.9792`) is preferred.

| Metric | Formula | Test Score (@ `0.5`) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $(TP + TN) / (TP + TN + FP + FN)$ | `0.8473` | Share of all test predictions that are correct; beats the `0.7739` majority-class baseline |
| **Precision** | $TP / (TP + FP)$ | `0.8758` | Share of students predicted to `Pass` who actually `Pass` |
| **Recall (Sensitivity)** | $TP / (TP + FN)$ | `0.9354` | Share of actual `Pass` students correctly identified by the model |
| **Specificity** | $TN / (TN + FP)$ | `0.5458` | Share of actual `Fail` students correctly flagged as `Fail` |
| **F1-Score** | $2 \times (\text{Precision} \times \text{Recall}) / (\text{Precision} + \text{Recall})$ | `0.9046` | Harmonic balance between precision and recall |
| **ROC-AUC** | Area under ROC curve | `0.8922` | Model's ability to rank `Pass` students above `Fail` students across thresholds |

### Threshold Sensitivity (`0.3`, `0.5`, `0.7`)

| Threshold | Predicted Positive (`Pass`) | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0.3` | `18,129` | `0.8353` | `0.8360` | `0.9792` | `0.9020` |
| `0.5` | `16,532` | `0.8473` | `0.8758` | `0.9354` | `0.9046` |
| `0.7` | `14,182` | `0.8236` | `0.9213` | `0.8442` | `0.8811` |

---

## Decision Log, Limitations & Ethics

* **Leakage Prevention Decision:** Excluded `exam_score`, `performance_grade`, `performance_level`, `questions_attempted`, and `questions_correct` (along with `student_id`) prior to splitting so predictions rely on study behaviors, attendance, and preparation rather than exam outcomes.
* **Threshold Trade-Off:** Moving from `0.3` to `0.7` raises precision (`0.8360` to `0.9213`) while reducing recall (`0.9792` to `0.8442`); `0.5` is recommended as the balanced default (`F1 = 0.9046`), with `0.3` or `0.7` selected according to whether recall or precision carries higher operational priority.
* **Model Limitations:** Logistic Regression assumes a linear decision boundary in log-odds space and may miss non-linear feature interactions. A predicted probability is a statistical estimate—not a guaranteed outcome—and should not be used alone for high-stakes academic decisions.

---

## Repository Structure

* `activity_2_logistic_regression_pass_fail.ipynb` — Executed Google Colab notebook containing data inspection, preprocessing, training, evaluation metrics, plots, and a new-student prediction
* `README.md` — Project documentation, methodology, evaluation summary, and decision log
