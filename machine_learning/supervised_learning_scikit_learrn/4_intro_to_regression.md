- target variable has continuous values
    - country's GDP
    - price of a house

###  Creating feautures and targets arrays

```python

# .values: Converts that single pandas column into a clean, raw NumPy array of numbers (like [148, 85, 183, ...]).

X = diabetes_df.drop("glucose", axis=1).values    # find the column glucose and remove it so we only input the features not the target (gluocose)

y = diabetes_df["glucose"].values # Grabs only the column named "glucose" from your DataFrame. This is target variable (the "correct answers" we want the model to learn to predict)

print(type(X), type(y)) # double checks if theyre numpy array already

```

```text
### Making predictions from a single feature

```python
X_bmi = X[:, 3]

```

* **Explanation:** Extracts the 4th column (index 3) of features, which represents BMI data.
* **Purpose:** Isolate a single variable to test its relationship with blood glucose.

```python
print(y.shape, X_bmi.shape)

```

* **Explanation:** Prints the dimensions of the target and feature arrays.
* **Purpose:** Confirms if the dimensions match. Both output as 1D arrays `(752,)`.

```python
X_bmi = X_bmi.reshape(-1, 1)

```

* **Explanation:** Converts the flat 1D feature array into a 2D array. The `-1` automatically calculates the row count, while `1` creates exactly one column.
* **Purpose:** Formats the feature array to `(752, 1)` to meet Scikit-Learn's strict requirement for 2D inputs.

```python
print(X_bmi.shape)

```

* **Explanation:** Prints the new dimensions of the BMI array.
* **Purpose:** Verifies the shape successfully changed to `(752, 1)`.

### Visualizing the Raw Data

```python
import matplotlib.pyplot as plt
plt.scatter(X_bmi, y)

```

* **Explanation:** Imports the plotting library and creates a scatter plot.
* **Purpose:** Plots patients as individual data points, with BMI on the X-axis and blood glucose on the Y-axis, to visually identify trends.

```python
plt.ylabel("Blood Glucose (mg/dL)")
plt.xlabel("Body Mass Index")
plt.show()

```

* **Explanation:** Applies labels to both axes and renders the plot window.
* **Purpose:** Makes the visual chart readable and displays it on screen.

### Fitting the Regression Model

```python
from sklearn.linear_model import LinearRegression

```

* **Explanation:** Imports the linear regression algorithm.
* **Purpose:** Provides the tool needed to calculate a line of best fit.

```python
reg = LinearRegression()

```

* **Explanation:** Instantiates an untrained linear regression model object.
* **Purpose:** Prepares a model container to hold the mathematically calculated trendline.

```python
reg.fit(X_bmi, y)

```

* **Explanation:** Fits the model using the BMI values and true blood glucose targets.
* **Purpose:** Mathematically determines the slope and intercept of the line that best aligns with the data points.

```python
predictions = reg.predict(X_bmi)

```

* **Explanation:** Generates estimated glucose levels for each BMI value using the trained trendline.
* **Purpose:** Produces coordinates to plot the linear path of our predictions.

```python
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)

```

* **Explanation:** Re-plots the scattered data points and overlay-plots a continuous line using the predictions.
* **Purpose:** Shows the calculated trendline directly on top of the original dataset.

```python
plt.ylabel("Blood Glucose (mg/dL)")
plt.xlabel("Body Mass Index")
plt.show()

```

* **Explanation:** Adds standard axis labels and renders the final combined chart.
* **Purpose:** Outputs the complete visual performance of the regression model.

```

```