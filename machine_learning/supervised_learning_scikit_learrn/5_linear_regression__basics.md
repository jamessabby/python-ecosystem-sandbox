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

* **Single Regression:** Predicts a target using only one feature (e.g., predicting glucose using only BMI).
  
  $$y = ax + b

* **Multiple Regression:** Predicts a target using multiple features at the same time (e.g., predicting glucose using BMI, age, and insulin levels). 

  $$y = a_1x_1 + a_2x_2 + ... + a_nx_n + b

* **How it works in Scikit-Learn:** The code works exactly the same way. You simply pass an array containing multiple feature columns instead of just one.

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
- a score from 0 to 1 that tells you how much of your target's behavior your features can explain
- measures how well your features explain the variation in your target variable.
     - values can range from 0 to 1
        - 1 = feature completely explained the target's variance
        
- If you are predicting Sales using Ad Spend, and your R^2 is 0.80:
    - 80% of the ups and downs in your sales numbers are directly explained by your advertising budget.
    - The remaining 20% is random noise or other factors your model doesn't know about (like the weather, holidays, or competitor actions).

* **The Scale:** It ranges from 0 to 1.
  * **High $R^2$ (Close to 1):** The data points hug the line tightly. Your features do a great job explaining the target.
  * **Low $R^2$ (Close to 0):** The data points are scattered randomly. Your features do not explain the target well.
* **Interpretation:** If your model gets an $R^2$ score of `0.356`, it means your features only explain about 35.6% of the variance in blood glucose levels. The other 64.4% is due to other factors not in your model.

```python

reg_all.score(X_test, y_test)

```

### Mean squared error (MSE) and root mean squared error


* **Mean Squared Error (MSE):** The average of the squared residuals. Because the errors are squared, the unit of measurement is also squared (e.g., blood glucose squared), which makes it hard to interpret in the real world.

  $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

* **Root Mean Squared Error (RMSE):** The square root of the MSE. Taking the square root converts the error back into the target variable's original unit of measurement.

  $$RMSE = \sqrt{MSE}$$

* **Interpretation:** If your model's RMSE is `24.02`, it means that on average, your model's predictions are off by approximately 24 milligrams per deciliter (mg/dL) of blood glucose.

```python

from sklearn.metrics import root_mean_squared_error

root_mean_squared_error(y_test, y_pred) # returns the square root of the MSE
```
