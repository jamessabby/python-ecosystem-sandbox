Here are your notes on **Data Quality and Ingestion** based on your latest slides.

---

## 1. What is Data Quality?

In machine learning, **Data Quality** measures how well your data serves its intended purpose.

> **Core Rule:** "Garbage in, garbage out." The performance of any machine learning model is directly constrained by the quality of the data fed into it.

---

## 2. The 4 Dimensions of Data Quality

To evaluate if your dataset is ready for production, check these four dimensions:

| Dimension | Key Question | Real-World Failure Example |
| --- | --- | --- |
| **Accuracy** | Does the data correctly describe reality? | A customer's recorded age is $18$, but they are actually $32$. |
| **Completeness** | Is any required data missing? | $80\%$ of customer records are missing a last name or phone number. |
| **Consistency** | Is the data definition synchronized across all systems? | Customer status is listed as `"Active"` in the sales database but `"Inactive"` in billing. |
| **Timeliness** | Is the data available when the model needs it? | Orders are processed daily at midnight, but the model needs real-time data to flag live fraud. |

* **Good News:** Low data quality doesn't mean a project is ruined! You can fix issues early in the pipeline by collecting more data, imputing missing values, or cleaning up data feeds.

---

## 3. Data Ingestion & ETL Pipelines

During the **Design Phase**, you map out how raw data moves from its original sources into your machine learning infrastructure.

An **ETL Pipeline** handles this process automatically in three steps:

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│       1. EXTRACT        │ ──> │      2. TRANSFORM       │ ──> │         3. LOAD         │
│                         │     │                         │     │                         │
│ Pull raw data from      │     │ Combine datasets, clean │     │ Save processed data into│
│ APIs, databases, or     │     │ formats, and run        │     │ a central database or   │
│ external services.      │     │ automated validation.   │     │ feature store.          │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘

```

### Automated Data Quality Checks

Inside the **Transform** step, automated sanity checks ensure bad data never reaches the model:

* **Schema Validation:** Verifying that expected columns exist.
* **Type Checking:** Ensuring the `Temperature` column strictly contains numbers, not strings.
* **Range Checks:** Flagging impossible values (e.g., negative age or percentages over $100\%$).

Catching these issues inside the ingestion pipeline prevents silent model failures downstream in production!