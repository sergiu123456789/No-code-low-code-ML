# Loan Condition Prediction with DataRobot

This guide shows a reproducible program pipeline to train, evaluate, and deploy a **loan condition (good / bad)** classifier in **DataRobot** using the following input features:

* `emp_length_int` (integer)
* `home_ownership` (categorical)
* `income_category` (categorical)
* `annual_inc` (numeric)
* `loan_amount` (numeric)
* `term` (categorical or numeric)
* `application_type` (categorical)
* `purpose` (categorical)
* `interest_payments` (numeric)
* `loan_condition` (target: `good` / `bad`) *-- target column*
* `interest_rate` (numeric)
* `grade` (categorical)
* `dti` (numeric)
* `region` (categorical)

---

## 1. Data preparation

1. Create a CSV file named `loans.csv` with a header row that contains exactly the column names above. Example first lines:

```csv
emp_length_int,home_ownership,income_category,annual_inc,loan_amount,term,application_type,purpose,interest_payments,loan_condition,interest_rate,grade,dti,region
5,RENT,25k-50k,42000,12000,36,INDIVIDUAL,credit_card,1800,good,13.5,B,12.3,North
2,OWN,50k-100k,75000,8000,60,JNT,debt_consolidation,900,bad,17.2,C,22.1,South
```

**Notes:**

* Make sure `loan_condition` is a categorical column with values like `good` and `bad` (consistent strings, no mixed casing/spaces).
* Avoid nulls in the target. For features, DataRobot will handle many missing-value patterns automatically, but you can impute or collapse rare categories beforehand for reproducibility.

---

## 2. Quick UI workflow (no code)

1. Login to DataRobot → **Projects** → **New Project** → Upload `loans.csv`.
2. When prompted, set **Target** to `loan_condition` and confirm the problem type (Binary Classification).
3. Run **Autopilot** (Default). DataRobot will explore blueprints and hyperparameters automatically.
4. Inspect the **Leaderboard** → pick a candidate model.
5. Use **Explain** (Feature Impact, Partial Dependence) to validate model behavior.
6. Deploy the chosen model via **Predict → Deploy Model** to get an API endpoint.

---

## 3. Validation & Evaluation

* Use the **Leaderboard** to inspect cross-validated metrics (AUC, LogLoss, recall, precision). For imbalanced classes consider using **F1** or a custom metric.
* Run the following checks in CI before deploying:

  * Unit tests for preprocessing logic (if any).
  * Small evaluation run on a holdout set (DataRobot supports backtesting and setting validation partitions).
* Consider using **Feature Impact** and \*\*Partial Dependence
