 Here are your structured notes on **Experiment Tracking & The Experiment Process** based on your latest slides.

---

## 1. What is Experiment Tracking?

In the **Development Phase**, building a machine learning model is inherently iterative. **Experiment Tracking** is the practice of systematically recording all the components and outputs of your training runs to find the best-performing model.

### Key Factors to Track in Every Experiment:

* **ML Model Types:** Linear Regression, Decision Trees, Neural Networks, etc.
* **Hyperparameters:** Learning rate, tree depth, number of hidden layers.
* **Data Versions:** Exact dataset snapshot used for training.
* **Execution Scripts:** Code files (`.py` or `.ipynb`) used to run the run.
* **Environment Configurations:** Python/R versions, library dependencies (`scikit-learn==1.3.0`, `torch==2.0`, etc.).

---

## 2. Why Tracking is Crucial

Proper tracking transforms isolated trial-and-error into a organized, reproducible pipeline:

* **Compare Results:** Benchmark performance metrics across dozens or hundreds of runs.
* **100% Reproducibility:** Recreate exact historic runs months later without guessing parameters.
* **Seamless Collaboration:** Let team members inspect each other's configurations and progress.
* **Clear Stakeholder Reporting:** Present trade-offs and metric progressions to business leaders clearly.

---

## 3. Tooling Comparison

| Tool Type | Pros | Cons |
| --- | --- | --- |
| **Spreadsheets (Excel/Google Sheets)** | Simple, zero setup, easy to use initially. | Heavy manual work, prone to human error, impossible to scale. |
| **Proprietary Platforms** | Custom-built for exact company workflows. | Expensive, requires high initial development time/effort. |
| **Dedicated Tracking Tools** *(e.g., MLflow, Weights & Biases)* | Built specifically for ML, automated logging, rich UI visualizations. | Requires a short initial learning curve. |

---

## 4. The 8-Step Experiment Process

```
  1. Formulate Hypothesis ("We expect adding 1,000 images improves accuracy...")
                          │
  2. Gather Images & Labels (Collect & validate dataset)
                          │
  3. Define Experiment (Select architecture, dataset, & initial hyperparameters)
                          │
  4. Setup Experiment Tracking (Initialize logger / tracking tool)
                          │
  5. Train the Machine Learning Model(s)
                          │
  6. Test on Hold-Out Set (Evaluate predictions against unseen test data)
                          │
  7. Register Suitable Model (Save top performing model weights & metadata)
                          │
  8. Visualize & Report (Share performance dashboards with team & plan next iteration)

```

> **Key Takeaway:** An experiment isn't finished when `model.fit()` completes—it's finished when the run is tracked, evaluated, logged, and ready for model registration!