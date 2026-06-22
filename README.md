# bKash Customer Churn Prediction

## Overview

This project was developed for the bKash Presents NSUCEC Datathon.

The objective is to predict customer churn using transaction history, account balance information, and customer profile data. The solution uses large-scale feature engineering, LightGBM, Optuna hyperparameter tuning, and SHAP explainability.

---

## Dataset

Data Sources:

- KYC Data
- Transaction Data
- Day End Balance Data

Target Variable:

- CHURN (0 = Active Customer, 1 = Churned Customer)

---

## Feature Engineering

Engineered features include:

### Transaction Features

- TOTAL_TRX_COUNT
- TOTAL_TRX_AMOUNT
- AVG_TRX_AMOUNT
- MAX_TRX_AMOUNT
- STD_TRX_AMOUNT
- RECENCY_DAYS

### Transaction Type Features

- BILLPAY_COUNT
- CASHIN_COUNT
- CASHOUT_COUNT
- MERCHANTPAY_COUNT
- P2P_COUNT

### Ratio Features

- BILLPAY_RATIO
- CASHIN_RATIO
- CASHOUT_RATIO
- MERCHANT_RATIO
- P2P_RATIO

### Balance Features

- AVG_BALANCE
- MAX_BALANCE
- MIN_BALANCE
- BALANCE_STD
- LAST_BALANCE

---

## Model

Algorithm:

- LightGBM Classifier

Hyperparameter Tuning:

- Optuna

Model Explainability:

- SHAP

---

## Results

### Validation Performance

| Metric | Score |
|----------|----------|
| AUC-ROC | 0.9832 |

---

## Explainability

SHAP was used to interpret model predictions.

Top Features:

1. trx_count_30d
2. recency_days
3. trx_count_90d
4. velocity_freq_30_90
5. total_amt_90d

Included:

- SHAP Summary Plot
- SHAP Dependence Plot (recency_days)
- SHAP Dependence Plot (trx_count_30d)

---

## Repository Structure

```text
README.md
features.md
features.py
notebook.ipynb
predictions.csv
report.pdf
presentation.pdf

shap_summary.png
shap_dependence_recency_days.png
shap_dependence_trx_count_30d.png
```

## Technologies Used

- Python
- Pandas
- Dask
- LightGBM
- Optuna
- SHAP
- Scikit-Learn
- Kaggle

---

## Author

Rashed Ul Islam Chowdhury

BRAC University
