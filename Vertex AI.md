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

---

## 1) Prerequisites

* **Google Cloud Project** with billing enable
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

---

## 3) Training with Vertex AI AutoML Tabular (UI path)

1. Go to **Vertex AI → Datasets → Tabular → Create dataset**.
2. Import data from **Cloud Storage** (CSV)
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

## 4) Model evaluation

Key metrics to examine:

* **RMSE / MAE**: lower is better
* **R²**: closer to 1 is better
* **Residual plots**: watch for heteroskedasticity and outliers

## 5) Deploy the model to an endpoint (for online predictions)

## 6) Make predictions

### Load new data 

## 7) Governance & Responsible AI notes

This is a statistical estimate, not an admissions decision engine.

Avoid using this in ways that could influence or gate real applications without human oversight.

Monitor data drift (e.g., changing score distributions across years) and model bias (e.g., towards certain universities).

Log predictions and evaluate periodically.

## 8) Cost & cleanup

Costs come from training, endpoint compute, storage, and optionally BigQuery.

For dev/test, set a small training budget and delete endpoints when not in use.

# Delete endpoint (and undeploy model)

# (Optional) Delete model

## 9) Troubleshooting tips

Schema mismatch: ensure column names/types in CSV match training schema.

Missing values: let AutoML handle, or impute before upload.

Low R²: add more data, try feature interactions, consider custom models.

Region mixups: use the same REGION everywhere (dataset, training, endpoint).



