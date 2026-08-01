# Tourism Package Purchase Prediction using MLOps

## Project Overview

This project implements an end-to-end MLOps pipeline to predict whether a customer will purchase the newly introduced **Wellness Tourism Package** offered by **Visit with Us**, a travel company.

The complete workflow automates:

- Data validation
- Data preprocessing
- Feature engineering
- Model training
- Hyperparameter tuning
- Experiment tracking using MLflow
- Model deployment using Streamlit
- Continuous Integration and Continuous Deployment (CI/CD) using GitHub Actions

---

# Business Problem

"Visit with Us" wants to identify customers who are most likely to purchase a newly launched Wellness Tourism Package before contacting them.

Instead of manually selecting customers, the company requires an automated machine learning solution that predicts potential buyers using historical customer and interaction data.

The objective is to improve:

- Marketing efficiency
- Customer targeting
- Conversion rate
- Business revenue

---

# Dataset

The dataset contains customer demographic information and sales interaction details.

Target Variable

- **ProdTaken**
    - 1 → Customer purchased the package
    - 0 → Customer did not purchase the package

---

# Project Structure

```
tourism_project/
│
├── data/
│   └── tourism.csv
│
├── data_registration/
│   ├── validate_dataset.py
│
├── data_preparation/
│   └── prep.py
│
├── model_building/
│   └── train.py
│
├── deployment/
│   ├── app.py
│   ├── requirements.txt
│   └── best_model_tourism_package_prediction_v1.joblib
│
├── .github/
│   └── workflows/
│       └── pipeline.yml
│
└── README.md
```

---

# MLOps Pipeline

The pipeline consists of the following stages.

## 1. Data Registration

- Loads the dataset
- Validates expected columns
- Prints dataset summary
- Prevents invalid datasets from entering the pipeline

---

## 2. Data Preparation

Performed preprocessing includes:

- Missing value treatment
- Feature engineering
- Removing unnecessary columns
- Train / Validation / Test split
- Saving processed datasets as workflow artifacts

---

## 3. Model Building

The model pipeline includes:

- StandardScaler
- OneHotEncoder
- OrdinalEncoder
- XGBoost Classifier

Hyperparameter tuning is performed using

- RandomizedSearchCV
- 5-Fold Cross Validation

Experiment tracking is implemented using

- MLflow

The best model and the optimized classification threshold are stored together using Joblib.

---

## 4. Deployment

The trained model is deployed using

- Streamlit Community Cloud

Users can enter customer details through the UI and receive:

- Purchase probability
- Predicted class
- Decision based on optimized threshold

---

## 5. CI/CD

GitHub Actions automatically executes:

- Dataset validation
- Data preprocessing
- Model training
- Model saving
- Workflow execution

whenever code is pushed to the repository.

---

# Feature Engineering

Additional features created include:

- Age Bucket
- Life Stage
- Interaction Efficiency
- Total Group Size
- Pitch Alignment
- Young Executive Flag

These engineered features improve predictive performance.

---

# Model Performance

Best Hyperparameters

| Parameter | Value |
|-----------|--------|
| n_estimators | 300 |
| learning_rate | 0.05 |
| max_depth | 8 |
| min_child_weight | 1 |
| gamma | 0.3 |
| subsample | 1.0 |
| colsample_bytree | 0.6 |
| reg_alpha | 1 |
| reg_lambda | 0.5 |

Optimized Classification Threshold

```
0.51
```

Performance Metrics

| Metric | Validation | Test |
|---------|-----------:|------:|
| Accuracy | 91.74% | 90.66% |
| Precision | 81.42% | 77.03% |
| Recall | 74.19% | 73.55% |
| F1 Score | 77.64% | 75.25% |

The model demonstrates strong generalization, as validation and test performance are very similar.

---

# Model Evaluation Metrics

## Confusion Matrix

| Actual | Predicted | Meaning |
|---------|-----------|---------|
| Purchased | Purchased | True Positive (TP) |
| Not Purchased | Purchased | False Positive (FP) |
| Purchased | Not Purchased | False Negative (FN) |
| Not Purchased | Not Purchased | True Negative (TN) |

---

## True Positive (TP)

The customer actually purchased the Wellness Tourism Package, and the model correctly predicted that the customer would purchase it.

Example:

- Customer purchased the package ✅
- Model predicted purchase ✅

Correct prediction.

---

## True Negative (TN)

The customer did not purchase the package, and the model correctly predicted that the customer would not purchase it.

Example:

- Customer did not purchase the package ✅
- Model predicted no purchase ✅

Correct prediction.

---

## False Positive (FP)

The customer did not purchase the package, but the model predicted that the customer would purchase it.

Example:

- Customer did not purchase ❌
- Marketing team contacted the customer because the model predicted purchase.

This leads to unnecessary marketing effort.

---

## False Negative (FN)

The customer actually purchased the package, but the model predicted that the customer would not purchase it.

Example:

- Customer purchased the package ✅
- Model predicted no purchase ❌

This represents a missed business opportunity because a valuable customer could have been ignored.

---

# Precision

Precision answers the question:

> Among all customers predicted to purchase, how many actually purchased?

Formula

```
Precision = TP / (TP + FP)
```

Higher precision means fewer false positives, reducing unnecessary marketing efforts.

Test Precision

```
77.03%
```

Meaning:

Among all customers predicted as likely buyers, **77.03% actually purchased the package.**

---

# Recall

Recall answers the question:

> Among all customers who actually purchased, how many did the model identify correctly?

Formula

```
Recall = TP / (TP + FN)
```

Higher recall means fewer missed potential customers.

Test Recall

```
73.55%
```

Meaning:

The model successfully identified **73.55% of all actual package purchasers.**

---

# F1 Score

The F1 Score balances Precision and Recall.

Formula

```
F1 Score =
2 × (Precision × Recall)
-------------------------
 Precision + Recall
```

A higher F1 Score indicates a better balance between identifying genuine buyers and avoiding unnecessary marketing efforts.

Test F1 Score

```
75.25%
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- MLflow
- Streamlit
- Joblib
- GitHub Actions

---
