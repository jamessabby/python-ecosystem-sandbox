# Hyperparameter Tuning & Cross-Validation

* **Hyperparameters**: Settings or knobs you choose manually **before** fitting the model.
* Unlike model weights that the algorithm learns automatically, you have to pick these yourself.
* *Examples:* Choosing `n_neighbors` in KNN or `alpha` in Ridge/Lasso regression.


* **Hyperparameter Tuning**: The process of systematically trying out different hyperparameter values, fitting them separately, and checking their performance to pick the best one.
* **Crucial Rule**: You must use **Cross-Validation (CV)** on your training set when tuning hyperparameters. If you use your raw test set to pick the best knobs, your model will overfit the test set and lose its ability to generalize to new, unseen data.



### 1. Grid Search Cross-Validation

* **Grid Search**: Setting up a grid of combinations for your hyperparameters, trying every single option, and picking the winner.
* *Example:* If you check 4 values for `n_neighbors` and 2 options for distance `metric` (Euclidean vs. Manhattan), Grid Search tests all 8 combinations.


* **How it runs**: It evaluates every combination by running k-fold cross-validation on the training set, calculating a mean score for each setting to find the optimal choice.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold

# 1. Setup the Cross-Validation splitting rule
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 2. Define the grid of options you want to test
# np.arange creates an array of choices from 0.0001 to 1.0
param_grid = {
    "alpha": np.arange(0.0001, 1.0, 10),
    "solver": ["sag", "lsqr"]
}

# 3. Instantiate Grid Search with your model, grid, and CV settings
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)

# 4. Run the search on the training data
ridge_cv.fit(X_train, y_train)

# .best_params_ gives the winning settings; .best_score_ gives the best validation score
print(ridge_cv.best_params_, ridge_cv.best_score_)

```

### 2. The Computational Limit & An Alternative

* **The Problem with Grid Search**: It scales terribly. The number of model fits equals:

$$\text{Total Fits} = (\text{Number of combinations}) \times (\text{Number of CV splits})$$


* *Example:* Testing 3 distinct hyperparameters with 10 values each over a 10-fold cross-validation requires **900 total fits**, which dramatically slows down your pipeline.


* **Randomized Search (`RandomizedSearchCV`)**: Instead of brute-forcing every single intersection on the grid, it randomly samples a fixed number of combinations.
* **Parameter `n_iter**`: Controls exactly how many random combinations to pick and test.
* *Why it's great:* It saves huge amounts of computation time while often finding a hyperparameter combination that performs just as well as (or sometimes better than) a full grid search.



```python
from sklearn.model_selection import RandomizedSearchCV

# n_iter=2 forces the model to pick and test only 2 random combinations from the grid
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2)
ridge_cv.fit(X_train, y_train)

print(ridge_cv.best_params_, ridge_cv.best_score_)

```

### 3. Final Evaluation

* **Testing the Winner**: Once your search finds the absolute best settings on the training/validation data, you run a final score evaluation using your untouched test set.

```python
# Evaluate the final performance of your optimized model on the test data
test_score = ridge_cv.score(X_test, y_test)
print(test_score)

```