

## 1. Logistic Regression: The Basics

Despite the word "regression," **Logistic Regression is used for Classification** (predicting labels, not continuous numbers).

* **How it works:** Instead of predicting a label directly, it calculates the **probability ($p$)** that a data point belongs to a certain class.
* **The Threshold:** By default, it uses 0.5 as the cutoff point:
* If $p \ge 0.5 \rightarrow$ Class 1 (e.g., Has Diabetes)
* If $p < 0.5 \rightarrow$ Class 0 (e.g., Healthy)


* **Linear Decision Boundary:** In a 2D plot, it draws a straight line to split the data. Everything on one side is guessed as class 1, and everything on the other is class 0.

---

## 2. Shift Your Perspective on the ROC Curve

When you change that $0.5$ threshold (say, to $0.2$ or $0.8$), your model's sensitivity changes. The **ROC Curve (Receiver Operating Characteristic)** maps out how the model performs at *every possible threshold*.

* **The Y-Axis (True Positive Rate / Recall):** Out of all the actual bad things, how many did we catch? (We want this high).
* **The X-Axis (False Positive Rate):** Out of all the actual clean things, how many did we accidentally flag as bad? (We want this low).
* **The Dotted Diagonal Line:** This represents a useless model that just guesses randomly (50/50 coin flip).
* **The Solid Curve:** This is your actual model. You want this line to bow upwards toward the top-left corner as much as possible—meaning you are catching positives *without* raising false alarms.

---

## 3. ROC AUC: The Final Score

**AUC** stands for **Area Under the Curve**. It compresses the whole ROC graph into a single score between $0.0$ and $1.0$.

* $\text{AUC} = 0.5$: Your model is as good as a random coin flip.
* $\text{AUC} = 1.0$: A perfect model.
* **Your Slide's Score (0.67):** This means the model is 34\% better than a random guesser (0.67 - 0.50 = 0.17, which represents the improvement scale). It's decent, but has room to grow.

---

## 4. Code Snippet for Your Notes

Here is the code structure adapted to your note-taking style to generate these metrics:

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# roc_curve calculates the TPR/FPR; roc_auc_score gives the final area metric
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# predict_proba() returns probabilities for [Class 0, Class 1]. 
# We slice [:, 1] to grab only the probability of being Class 1.
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

# 1. Generate ROC Curve coordinates
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Plotting the random guess line (diagonal)
plt.plot([0, 1], [0, 1], 'k--')             # point A to point B, then black dashed line
# Plotting the actual model's ROC curve
plt.plot(fpr, tpr)                      
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()

# 2. Calculate the AUC Score directly
print(roc_auc_score(y_test, y_pred_probs))

```
