# Credit Risk Prediction Model

A machine learning project that predicts the probability of loan default using applicant financial and demographic data. Built with Python, scikit-learn, Logistic Regression, and Random Forest.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Results](#results)
- [Key Findings](#key-findings)
- [Business Impact](#business-impact)
- [Saved Artifacts](#saved-artifacts)

---

## Overview

Banks and lenders need reliable ways to assess whether a loan applicant is likely to default. Poor credit risk assessment leads to financial losses and increased non-performing loan ratios. This project builds two classification models — Logistic Regression and Random Forest — to predict loan default and estimate each applicant's risk probability.

---

## Dataset

**File:** `credit_risk_dataset.csv` (semicolon-delimited)

The dataset contains personal financial and loan-related information for a sample of loan applicants. Key features include:

| Feature | Description |
|---|---|
| `person_age` | Age of the applicant |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status (e.g., RENT, OWN, MORTGAGE) |
| `person_emp_length` | Years of employment |
| `loan_intent` | Purpose of the loan (e.g., EDUCATION, MEDICAL) |
| `loan_grade` | Loan grade assigned by the lender |
| `loan_amnt` | Loan amount requested |
| `loan_int_rate` | Interest rate on the loan |
| `loan_percent_income` | Loan amount as a percentage of income |
| `cb_person_default_on_file` | Whether the applicant has a prior default on record |
| `cb_person_cred_hist_length` | Length of the applicant's credit history (years) |
| **`loan_status`** | **Target variable — 1 = Default, 0 = No Default** |

### Missing Value Handling

- `person_emp_length`: Missing values filled with `0`, interpreted as currently unemployed or no employment history.
- `loan_int_rate`: Missing values filled with `0`, treated as no interest rate recorded.


---

## Installation

Install the required Python libraries before running the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then launch the notebook:

```bash
jupyter notebook credit_risk_model.ipynb
```

---

## How It Works

### 1. Data Loading & Exploration
The dataset is loaded and inspected using `.info()`, `.describe()`, and `.isnull().sum()` to understand the data shape, types, and missing values.

### 2. Preprocessing
- Missing values in `person_emp_length` and `loan_int_rate` are filled with `0`.
- Categorical columns (`person_home_ownership`, `loan_intent`, `loan_grade`) are converted to numeric format using one-hot encoding (`pd.get_dummies`), dropping the first category to avoid multicollinearity.

### 3. Train/Test Split
The data is split 80/20 into training and test sets using a fixed random seed (`random_state=42`) for reproducibility.

### 4. Model Training
Two models are trained on the same training data:
- **Logistic Regression** — a linear baseline model (`max_iter=1000`)
- **Random Forest** — an ensemble of decision trees (default hyperparameters)

### 5. Evaluation
Both models are evaluated on the held-out test set using:
- **Accuracy score**
- **Classification report** (Precision, Recall, F1-score)
- **Confusion matrix**

### 6. Risk Probability Estimation
The Logistic Regression model outputs a probability for each applicant using `.predict_proba()`. The second column (index `[:,1]`) gives the probability of default, which is stored as `Risk_Probability` alongside predictions in a results table.

### 7. Feature Importance
Random Forest's built-in feature importances are extracted and ranked to identify the strongest predictors of loan default.

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~85% |
| Random Forest | ~93% |

---

## Key Findings

- **Random Forest outperformed Logistic Regression**, achieving higher accuracy by capturing non-linear relationships in the data.
- **Loan interest rate (`loan_int_rate`)** and **loan-to-income ratio (`loan_percent_income`)** were the strongest predictors of default risk — applicants with high rates and high loan burdens relative to income were significantly more likely to default.
- **Prior defaults on file (`cb_person_default_on_file`)** was also a strong risk signal, confirming that past behaviour is a reliable predictor of future default.
- **Loan grade** provided additional predictive signal, as lower grades are already associated with higher perceived risk by lenders.

---

## Business Impact

This model gives lenders a data-driven tool to:

- **Screen applicants** — flag high-risk applicants before approving loans
- **Quantify risk** — use the `Risk_Probability` score to set appropriate interest rates or credit limits
- **Reduce defaults** — improve the quality of the loan book by rejecting or adjusting offers for high-risk profiles
- **Support compliance** — document objective, evidence-based lending decisions

---

## Saved Artifacts

The trained Logistic Regression model and feature column names are saved using `pickle` for deployment or future use:

```python
import pickle

# Load the model
with open("credit_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature columns
with open("feature_columns.pkl", "rb") as f:
    features = pickle.load(f)
```

These files can be integrated into a Streamlit dashboard or a FastAPI deployment endpoint for real-time predictions.

---

## Author

*[Your Name]*
MSc Physics | Data Science Portfolio Project
