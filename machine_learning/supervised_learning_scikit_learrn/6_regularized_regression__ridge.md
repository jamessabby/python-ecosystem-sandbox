# Notes from lesson

### Understanding Regularization

    In simple linear regression, the model tries to fit the data points as closely as possible by choosing coefficients ($a_1, a_2, \dots$) for each feature. However, if the model is allowed to make these coefficients extremely large, it starts matching the noise in the training data perfectly. This leads to overfitting.

    **Regularization** is a technique that keeps these coefficients in check by introducing a penalty (essentially a "tax" or a "fine") on large coefficients. This forces the model to stay simple and generalize better to new data.

---

### Ridge Regression

Ridge regression penalizes the model by adding the sum of the **squared values** of the coefficients to the loss function:

$$Loss = OLS\ Loss + \alpha \sum_{i=1}^{n} a_i^2$$

* **$\alpha$ (Alpha):** This is a hyperparameter you choose to control how strictly you penalize large coefficients.
* **If $\alpha = 0$:** There is no penalty. The model behaves exactly like standard linear regression (OLS), which can lead to overfitting.
* **If $\alpha$ is too high:** The penalty is too heavy, forcing coefficients close to zero. This makes the model too simple and can lead to underfitting.



#### Ridge Code Breakdown

```python
from sklearn.linear_model import Ridge
scores = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    scores.append(ridge.score(X_test, y_test))

```

* **`Ridge(alpha=alpha)`:** Instantiates a Ridge model using the current $\alpha$ value from the loop.
* **`scores.append(...)`:** Calculates the $R^2$ score on the test set for each $\alpha$ value and saves it. In the output, you can see that as $\alpha$ gets too large, the score starts dropping because the penalty becomes too aggressive.

---

### Lasso Regression

Lasso regression is similar to Ridge, but it penalizes the model using the sum of the **absolute values** of the coefficients:

$$Loss = OLS\ Loss + \alpha \sum_{i=1}^{n} \vert{}a_i\vert{}$$

#### Lasso Code Breakdown

```python
from sklearn.linear_model import Lasso
scores = []
for alpha in [0.01, 1.0, 10.0, 20.0, 50.0]:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    scores.append(lasso.score(X_test, y_test))

```

* Works identically to the Ridge loop, but uses absolute penalties. The output shows performance dropping off substantially once $\alpha$ goes over 20.

---

### Lasso for Feature Selection

Lasso has a unique and powerful superpower that Ridge does not have: **it can shrink unimportant coefficients all the way to exactly zero.**

* In Ridge regression, coefficients get very small, but they never quite hit zero.
* In Lasso regression, if a feature is not helping the model predict the target, Lasso completely zeroest it out, removing it from the equation.
* This makes Lasso excellent for **feature selection**—helping you identify which features in a massive dataset actually matter.

#### Feature Selection Code Breakdown

```python
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values
names = diabetes_df.drop("glucose", axis=1).columns

lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_

plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.show()

```

* **`names`**: Saves the actual names of the feature columns (like pregnancies, bmi, age) so we can label our graph later.
* **`lasso.fit(X, y).coef_`**: Fits the Lasso model to the entire dataset and immediately extracts the calculated coefficient values for each feature.
* **`plt.bar(names, lasso_coef)`**: Draws a bar chart where the X-axis shows the feature names and the Y-axis shows their coefficient weights.

#### Reading the Bar Chart Output

Looking at the final bar chart:

* Features like **insulin, triceps, and age** have bars sitting at or extremely close to **0**. Lasso decided these features are not very important for predicting glucose levels when other features are present.
* The feature **"diabetes"** (representing whether the patient's biological parent has diabetes) has a massive positive bar sitting near **25**.
* This visualizes that family history of diabetes is by far the most important predictor of blood glucose levels in this dataset.




# Questions and answers

### 1. Are coefficients numbers? How can they be extremely large?

Yes, coefficients are just regular numbers. In the line equation $y = ax + b$, the coefficient is $a$. It represents the slope of the line (how much $y$ changes when $x$ moves by 1).

* **Small coefficient ($a = 2$):** If $x$ increases by 1, $y$ increases by 2. This is a gentle, stable slope.
* **Large coefficient ($a = 10,000$):** If $x$ increases by 1, $y$ shoots up by 10,000.

**How they get large:** When a model is overcomplicating things, it starts making wild, aggressive guesses. To force the trendline to curve up and down to hit every single random outlier dot, the math behind standard linear regression will generate massive positive and negative coefficient numbers (like $+5000$ for one feature and $-4000$ for another).

---

### 2. What does "matching the noise... leads to overfitting" mean?

Real-world data is messy. It contains the true, underlying trend (the signal) and random, temporary flukes (the noise).

* **The Signal:** Generally, people with higher BMIs have higher blood glucose.
* **The Noise:** One specific patient with a very high BMI had a low glucose level on Tuesday because they skipped breakfast, drank five cups of water, and took medication right before the test. This is a one-time fluke.

**Overfitting:** A basic linear regression model is obsessive. It wants to get a perfect 100% score, so it draws a crazy, zigzagging line to make sure it passes directly through that one-time fluke point.

Because the model bent its shape to match that "noise," it will fail completely when you give it new patients who didn't happen to skip breakfast on Tuesday. It learned the flukes instead of the actual trend.

---

### 3. What does "penalizing" mean in this context?

Normally, the model's only math rule is: **"Make the prediction error as close to zero as possible."** The model doesn't care if it has to use crazy, giant coefficients to achieve that.

**Penalizing** changes the rules of the game. We change the formula so the model's goal becomes: **"Make the prediction error small, BUT you get penalized (fined) if you make your coefficients too big."**

It works like a tax:

* If the model wants to use a tiny coefficient like $1.2$, the tax is almost $0$.
* If the model tries to use a massive coefficient like $8,500$ to chase an outlier, the penalty is huge.

The model is forced to compromise. It decides: *"Chasing that weird outlier dot isn't worth the massive penalty, so I'll just draw a simpler, flatter, and more realistic line instead."*

---

### 4. In the Ridge code, what are those values: `[0.1, 1.0, 10.0, 100.0, 1000.0]`?

These numbers represent different values of **Alpha ($\alpha$)**, which is the **strictness of your tax rate**.

* **`0.1`:** A very loose, low tax. The model is barely penalized for large coefficients.
* **`1.0`:** A moderate tax.
* **`1000.0`:** An extremely strict, heavy tax. The model will be heavily punished for any coefficient that isn't almost zero.

Because we don't know which tax rate is the "sweet spot" for our specific dataset, we use a loop to test all of them. We look at the resulting accuracy scores to see which value of Alpha gives the highest performance on the test set.

---

### 5. Lasso Regression: The Easy Explanation

Think of building a machine learning model like packing a backpack for a weekend hiking trip. Your "features" (BMI, Age, Insulin, etc.) are the items you can pack.

* **Ridge Regression** is like saying: *"You must pack every single item in the house, but we will shrink their weights so they are all incredibly light."* (Ridge keeps all features, but makes their coefficients very small).
* **Lasso Regression** is a strict minimalist. It says: *"You have a strict weight limit. If an item is not 100% essential to your survival, throw it in the trash."*

Because Lasso uses absolute math values for its penalty, it is able to shrink the coefficient of unimportant features **all the way to exactly zero**.

If a feature's coefficient is 0, it means $0 \times \text{feature} = 0$. That feature is completely erased from the model. This leaves you with only the absolute most important features, making your model incredibly clean and easy to understand.

