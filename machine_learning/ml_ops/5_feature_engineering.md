Here are your structured notes covering **Feature Engineering, Selection, Feature Stores, and Data Version Control** based on your slides.

---

## 1. What is Feature Engineering? (`Screenshot 2026-07-23 103125.png`)

Feature engineering kicks off the **Development Phase** of the MLOps lifecycle. It is the process of selecting, manipulating, and transforming raw data into meaningful variables (features) that help models learn patterns more effectively.

### Example: Creating New Features (`Screenshot 2026-07-23 103147.png`)

If you have raw customer order data:

* **Raw Features:** `Number of orders` and `Total expenditure`
* **Engineered Feature:** `Average expenditure = Total expenditure / Number of orders`

Creating `Average expenditure` provides a single, high-signal variable ($495.50 vs $4,272.50) that directly highlights high-value customer behavior.

---

## 2. Feature Selection (`Screenshot 2026-07-23 103340.png`)

More features aren't always better. **Feature Selection** keeps only the most relevant variables to speed up training times, prevent overfitting, and keep models interpretable.

### Methods for Selecting Features:

* **Domain-Specific Knowledge:** Using industry expertise to know which inputs actually matter in real life.
* **Correlation Matrices / Heatmaps:** Finding and dropping redundant features that tell the exact same story (e.g., `years_in_company` vs `experience` having a $0.99$ correlation).
* **Feature Importance Scores:** Tree-based models or algorithms evaluating which variables directly drive predictive power.
* **Advanced Selection Methods:**
* **PCA (Principal Component Analysis):** Transforms correlated features into fewer, uncorrelated components.
* **RFE (Recursive Feature Elimination):** Iteratively fits models and removes the weakest features step-by-step.



---

## 3. The Feature Store (`Screenshot 2026-07-23 103408.png`)

A **Feature Store** is a central repository where engineered features are stored, documented, and shared across different teams and projects.

* **Key Functions:**
* **Transform & Store:** Ingests batch or streaming data and transforms them into standard features.
* **Serve:** Delivers features consistently to models during training (batch) and live inference (real-time).
* **Monitor & Register:** Tracks feature definitions so teams avoid re-inventing the wheel.


* **When to use it?** Mainly relevant for large engineering teams running multiple projects off shared datasets. For small or isolated projects, a full feature store adds unnecessary complexity.

---

## 4. Data Version Control (DVC) (`Screenshot 2026-07-23 103628.png`)

Traditional Git tracks changes in code (`.py` files), but it cannot handle massive datasets (`.csv`, `.parquet`, image folders).

**Data Version Control (DVC)** works alongside Git to track dataset changes:

* **Maintains Consistency:** Links specific code commits to exact dataset versions so training runs are 100% reproducible.
* **Easy Rollbacks:** If a new dataset update corrupts a model or introduces bias, you can instantly rollback to an earlier dataset version (`v1` $\rightarrow$ `v2`).