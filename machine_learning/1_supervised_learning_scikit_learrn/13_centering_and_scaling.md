Here are your structured study notes covering **Centering & Scaling Data** and using **GridSearchCV with Pipelines** based on your latest slides.

---

## 1. Why Do We Scale Data?

Many machine learning models—especially distance-based ones like **$K$-Nearest Neighbors (KNN)**, **Support Vector Machines (SVMs)**, or **Lasso/Ridge Regression**—calculate mathematical distances between data points to make predictions.

If your features have vastly different numerical ranges, the feature with the largest numbers will completely overpower the model's logic, making it ignore features with smaller ranges.

### The Dataset Example (`image_04a6c5.png`):

* `duration_ms` ranges up to **1,610,000**
* `loudness` ranges from **-38.7 to -0.88**
* `speechiness` ranges from **0.02 to 0.71**

Because `duration_ms` numbers are so massive, an unscaled KNN model thinks duration is millions of times more important than `speechiness` or `loudness`, simply because of the units used!

---

## 2. Scaling Methods (`image_04f976.png`)

To put features on an equal playing field, we bring them to a similar scale:

* **Standardization (Centering and Scaling):**
Subtract the mean ($\mu$) and divide by the standard deviation ($\sigma$). This centers the feature around $0$ with a variance (and standard deviation) of $1$.

$$z = \frac{x - \mu}{\sigma}$$


* **Min-Max Normalization:**
Subtract the minimum value and divide by the range ($\text{max} - \text{min}$). This squashes all values into a range strictly between $0$ and $1$.

---

## 3. Scaling Mechanics & Proof (`image_04fd3d.png`)

In scikit-learn, we use `StandardScaler`. Just like imputers, scalers are **transformers**, so you must `fit_transform` the training data and only `transform` the test data to prevent **data leakage**.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()

# Fit parameters (mean, std) on train data, then scale train data
X_train_scaled = scaler.fit_transform(X_train)

# Scale test data using the training parameters
X_test_scaled = scaler.transform(X_test)

```

### The Proof of Impact (`image_0500f9.png` vs `image_0500f9.png`):

* **Accuracy WITHOUT Scaling:** **$53\%$** (`0.53`)
* **Accuracy WITH Scaling:** **$81\%$** (`0.81`)

> *Simply scaling the features boosted the KNN model's accuracy by nearly $30\%$!*

---

## 4. Putting Scalers into a `Pipeline` (`image_0500f9.png`)

Instead of scaling manually every time, you package the scaler and classifier inside a scikit-learn `Pipeline`.

```python
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

# Define sequence of steps as (name, object) tuples
steps = [
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=6))
]

pipeline = Pipeline(steps)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Fit pipeline on train set and evaluate on test set
knn_scaled = pipeline.fit(X_train, y_train)
print(knn_scaled.score(X_test, y_test))  # Output: 0.81

```

---

## 5. Hyperparameter Tuning on a Pipeline (`image_05043c.png` & `image_05045b.png`)

When tuning hyperparameters (like `n_neighbors` in KNN) using `GridSearchCV` on a `Pipeline`, you must use a special syntax to tell GridSearch which pipeline step the parameter belongs to:

### Syntax Rule: `<step_name>__<parameter_name>`

Use the exact name assigned in your `steps` tuple, followed by a **double underscore (`__`)**, then the parameter name.

```python
from sklearn.model_selection import GridSearchCV
import numpy as np

# Define pipeline steps
steps = [
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
]
pipeline = Pipeline(steps)

# Set grid parameters using double underscore syntax: stepname__paramname
parameters = {"knn__n_neighbors": np.arange(1, 50)}

# Run GridSearch cross-validation on the pipeline
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)

# Make predictions and check results
y_pred = cv.predict(X_test)

print(cv.best_score_)   # Output: ~0.82
print(cv.best_params_)  # Output: {'knn__n_neighbors': 12}

```

### Key Takeaways:

1. **Never skip scaling** when using distance-based algorithms like KNN or SVMs.
2. **Pipelines automatically scale folds separately** during cross-validation, protecting you completely from data leakage.
3. **Use double underscores (`__`)** in `GridSearchCV` parameter dictionaries to access parameters deep inside a pipeline step.