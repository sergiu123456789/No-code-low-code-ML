# Vertex AI – Predicting Chance of Admission

A complete, end‑to‑end guide to build, evaluate, deploy, and use a **tabular ML model in Google Cloud Vertex AI** that predicts a student’s **Chance of Admission** from:

* **GRE Score** (integer)
* **TOEFL Score** (integer)
* **University Rating** (ordinal 1–5)
* **Statement of Purpose (SOP)** strength (continuous 1–5)
* **Letter of Recommendation (LOR)** strength (continuous 1–5)
* **CGPA** (continuous 0–10)
* **Research** experience (binary 0/1)

Outcome/label: **Chance of Admit** (continuous 0–1).

> You can implement this with **Vertex AI AutoML Tabular** (no-code/low-code) or **custom training** (e.g., XGBoost, scikit-learn). This README focuses on **AutoML Tabular** with Python/CLI equivalents.

---

## 1) Prerequisites

* **Google Cloud Project** with billing enabled
* Permissions: `Vertex AI Admin` (or a combo of `Storage Admin`, `BigQuery Data Editor`, `Vertex AI User`)
* **APIs enabled**: `Vertex AI API`, `Cloud Storage`, `BigQuery` (optional)
* Tools (choose one path):

  * **Console (UI)** – easiest
  * **CLI** – `gcloud` ≥ 455.0.0
  * **Python** – `python 3.9+`, `google-cloud-aiplatform`

```bash
# CLI quick setup
gcloud init
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>

# Install Python SDK (optional)
pip install -U google-cloud-aiplatform
```

---

## 2) Data & Schema

Create a CSV with one row per applicant. Example header and a few rows:

```csv
GRE,TOEFL,UniversityRating,SOP,LOR,CGPA,Research,ChanceOfAdmit
337,118,4,4.5,4.5,9.65,1,0.92
324,107,4,4.0,4.5,8.87,1,0.76
316,104,3,3.0,3.5,8.00,1,0.65
```

### Feature types

| Column           | Type          | Notes         |
| ---------------- | ------------- | ------------- |
| GRE              | Integer       | 260–340       |
| TOEFL            | Integer       | 0–120         |
| UniversityRating | Integer       | 1–5 (ordinal) |
| SOP              | Float         | 1.0–5.0       |
| LOR              | Float         | 1.0–5.0       |
| CGPA             | Float         | 0.0–10.0      |
| Research         | Integer       | 0 or 1        |
| ChanceOfAdmit    | Float (label) | 0.0–1.0       |

> Save your CSV(s) in **Cloud Storage**: `gs://<YOUR_BUCKET>/admissions/train.csv`

Optionally, load into **BigQuery** for data exploration and SQL transformations.

---

## 3) Training with Vertex AI AutoML Tabular (UI path)

1. Go to **Vertex AI → Datasets → Tabular → Create dataset**.
2. Import data from **Cloud Storage** (CSV) or **BigQuery** table.
3. **Target column**: `ChanceOfAdmit`.
4. **Problem type**: **Regression**.
5. **Optimization metric**: RMSE (default). Consider **R²** and **MAE** as supporting metrics.
6. **Data split**: Auto split or custom (e.g., 80/10/10).
7. (Optional) **Feature transformations**:

   * Treat `UniversityRating` as **categorical/ordinal**.
   * Keep `Research` as categorical **binary**.
   * Leave others numeric.
8. Click **Train new model**. Name it (e.g., `admission-regressor-automl`).
9. After training, open the **Evaluation** tab for metrics and **Feature importance**.

---

## 4) Training with Python SDK (programmatic)

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import TabularDataset
from google.cloud.aiplatform.training_jobs import AutoMLTabularTrainingJob

PROJECT_ID = "<YOUR_PROJECT_ID>"
REGION = "us-central1"  # or europe-west4, etc.
BUCKET = "<YOUR_BUCKET>"
GCS_SOURCE = f"gs://{BUCKET}/admissions/train.csv"
TARGET_COLUMN = "ChanceOfAdmit"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")

dataset = TabularDataset.create(
    display_name="admissions_dataset",
    gcs_source=[GCS_SOURCE],
)

job = AutoMLTabularTrainingJob(
    display_name="admission-automl-regressor",
    optimization_prediction_type="regression",
    optimization_objective="minimize-rmse",
)

model = job.run(
    dataset=dataset,
    target_column=TARGET_COLUMN,
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    sync=True,
)

print("Model resource name:", model.resource_name)
```

**Notes**

* Set `budget_milli_node_hours` on `run()` to control training budget, e.g. `1000` (≈ 1 node‑hour).
* To exclude columns, pass `excluded_columns=[...]`.

---

## 5) Model evaluation

Key metrics to examine:

* **RMSE / MAE**: lower is better
* **R²**: closer to 1 is better
* **Residual plots**: watch for heteroskedasticity and outliers
* \*\*Permutation feature
