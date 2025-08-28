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

## 3. Reproducible programmatic workflow (Python)

This section uses the **DataRobot Python client** to perform the same steps programmatically: create a project, run training, choose a model, deploy, and predict.

### Install dependencies

```bash
pip install datarobot pandas
```

### Script: `train_and_deploy.py`

```python
import datarobot as dr
import pandas as pd
import time

# 1) Connect to DataRobot
API_TOKEN = "YOUR_API_TOKEN"
API_URL = "https://app.datarobot.com/api/v2"
dr.Client(token=API_TOKEN, endpoint=API_URL)

# 2) Create a project by uploading data
project = dr.Project.create(sourcedata='loans.csv', project_name='loan-condition-prediction')
project.set_target(target='loan_condition', mode=dr.AUTOPilotNode.BINARY)

# 3) Wait for Autopilot to complete (or poll leaderboard)
print('Waiting for Autopilot...')
while project.get_autopilot_status().state not in ('COMPLETED', 'ABORTED'):
    print('.', end='', flush=True)
    time.sleep(10)

print('\nAutopilot finished. Leaderboard:')
print(project.get_models()[:10])

# 4) Get best model (champion)
model = project.get_champion()
print('Champion model:', model.model_type, 'id=', model.id)

# 5) Deploy the model
deployment = dr.Deployment.create_from_learning_model(model.id, label='loan-condition-deployment')
print('Deployment created id:', deployment.id)

# 6) Make a prediction (example single-row predict)
sample = pd.DataFrame([{
    'emp_length_int': 5,
    'home_ownership': 'RENT',
    'income_category': '25k-50k',
    'annual_inc': 42000,
    'loan_amount': 12000,
    'term': 36,
    'application_type': 'INDIVIDUAL',
    'purpose': 'credit_card',
    'interest_payments': 1800,
    'interest_rate': 13.5,
    'grade': 'B',
    'dti': 12.3,
    'region': 'North'
}])

preds = deployment.predict(sample)
print(preds)
```

**Important:** Replace `YOUR_API_TOKEN` and `API_URL` with your credentials. If you use an on-prem or private instance, update `API_URL` accordingly.

---

## 4. Validation & Evaluation

* Use the **Leaderboard** to inspect cross-validated metrics (AUC, LogLoss, recall, precision). For imbalanced classes consider using **F1** or a custom metric.
* Run the following checks in CI before deploying:

  * Unit tests for preprocessing logic (if any).
  * Small evaluation run on a holdout set (DataRobot supports backtesting and setting validation partitions).
* Consider using **Feature Impact** and \*\*Partial Dependence
