# Linear Regression mechanics

### formula
y = ax + b

- y = target 
- x = single feature
- a, b = parameters/coefficients of the model (slope, interecept)

### How do we choose a and b?
- define an error function for any given line
- Choose the line that minimizes the error function

- Error function = loss function = cost function

### Residual
- vertical distance between an actual data point (the real value) and our regression line (the predicted value).
- It is literally "how much the model missed by" for that specific point.

* **Why can't we just add up all the errors?** 
- Some points are above the line (positive error) and some are below (negative error). If you simply add them up, they cancel each other out, making a bad line look perfect.

* **The Solution:** 
- We must define a loss function (also called an error or cost function) that measures total error without letting positive and negative values cancel out.

### Ordinary Least Squares (OLS)

* **What is RSS?** To get rid of negative values, we square every single residual (which makes them all positive) and add them together. This is called the Residual Sum of Squares (RSS):

  $$RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

* **What is OLS?** Ordinary Least Squares is the method used to find the absolute best line. It does this by testing different lines until it finds the one that makes the RSS as small (minimized) as possible.

## Linear Regression in Higher Dimension

y = a1x1 + a2x2 + b

where:
- x1, x2 = features
- y = target 

- To fit a linear regression:
    - need to specify 3 variables: a1, a2, b

- In higher dimension:
    - known as multiple regression
    - must specify coefficients for each feature and the variable b
    - y = a1x1 + a2x2 + a3x3 + ... anxn + ... b

- scikit learn works exactly the same way:
    - pass two arrays: features and target

### Linear Regression using all features

```python

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg_all = LinearRegression
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

```

## R-squared
- quantifies the variance in target values explained by the features
     - values can range from 0 to 1
        - 1 = feature completely explained the target's variance

```python

reg_all.score(X_test, y_test)

```

### Mean squared error (MSE) and root mean squared error

- MSE is measured in target units, squared

```python

from sklearn.metrics import root_mean_squared_error

root_mean_squared_error(y_test, y_pred) # returns the square root of the MSE
```
