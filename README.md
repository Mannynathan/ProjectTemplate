![](UTA-DataScience-Logo.png)

# Project Title

**Customer Churn Prediction Using Tabular Data**

## One Sentence Summary

This repository contains an end-to-end machine learning project that predicts bank customer churn using demographic and financial features from a public Kaggle dataset (similar to [Churn Modelling Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling)).

---

## Overview

**Definition of the Task**  
The goal is to predict whether a customer will exit (churn) based on personal and account-related features such as Age, Geography, Credit Score, and Account Activity. The task is a binary classification problem, where the output is either 0 (no churn) or 1 (churn).

**Approach**  
We formulated this as a binary classification task using a Random Forest Classifier. Numerical features were scaled, and categorical features were one-hot encoded. We built a preprocessing and modeling pipeline using Scikit-learn to streamline the workflow.

**Summary of Performance**  
Our model achieved a validation **ROC AUC score of 0.8717**.  
The overall validation **accuracy was 86%**. Precision, recall, and F1 scores indicated strong predictive performance, especially for the majority class.

---

## Summary of Work Done

---

### Data

**Type:**  
- Input: Tabular CSV file with 10,000+ customer records.
- Features: Mix of numerical (e.g., CreditScore, Balance) and categorical (e.g., Gender, Geography) data.

**Size:**  
- ~10,000 rows  
- 13 feature columns + 1 target column (`Exited`)

**Instances:**  
- 80% for training (8,000 customers)  
- 20% for validation (2,000 customers)

---

### Preprocessing / Cleanup

- Dropped ID columns (`CustomerId`, `Surname`, `id`) that are irrelevant for prediction.
- Handled categorical variables (`Gender`, `Geography`) using One-Hot Encoding.
- Scaled numerical features (`CreditScore`, `Age`, `Balance`, etc.) using StandardScaler.

---

### Data Visualization

- Histograms plotted for numerical features showing churn vs non-churn distribution.
- Bar plots plotted for categorical features showing differences in churn rates by Geography and Gender.
- Identified important trends: older age and higher balance correlated with churn.

---

### Problem Formulation

**Input:**  
- Processed numerical and categorical customer data.

**Output:**  
- Binary label: `Exited = 1` if customer churned, `Exited = 0` otherwise.

**Model:**  
- Random Forest Classifier (100 trees)

**Loss, Optimizer, Hyperparameters:**  
- Default scikit-learn loss (Gini impurity for splits)
- 100 estimators (trees)
- Random state fixed for reproducibility

---

### Training

- **Hardware:** MacBook / PC CPU
- **Software:** Python 3.10+, Jupyter Notebook
- **Training Time:** ~1-2 minutes
- **Training Details:**
  - Split 80/20 for train/validation.
  - Trained Random Forest with pipeline including preprocessing steps.
  - Evaluated using validation ROC AUC, accuracy, and F1-score.

- **Difficulties Encountered:**
  - Categorical encoding needed careful setup.
  - Slight imbalance in churn classes (customers leaving were fewer than staying).

---

### Performance Comparison

| Metric        | Score    |
|---------------|----------|
| Accuracy      | 86%      |
| ROC AUC Score | 0.8717   |
| Macro F1-Score| 0.76     |

**Classification Report:**









