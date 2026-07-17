# Class Imbalance & Classification Metrics

* **Class Imbalance**: Occurs when your dataset has an uneven frequency of classes (e.g., one class vastly outnumbers the other).
* If you evaluate an imbalanced dataset using standard accuracy, a model that simply predicts the majority class every single time will get a misleadingly high score while completely failing at its actual purpose.
* *Example:* In a dataset with 99% legitimate transactions and 1% fraudulent transactions, a lazy model that predicts "NONE of the transactions are fraudulent" will be **99% accurate**, yet it fails to catch a single fraudster.



### Classification Metrics

* **Accuracy**: Correct predictions / total number of samples.
* Not a reliable metric for datasets with class imbalance because it fails to highlight how well the model predicts minority classes.


* **Confusion Matrix**: A $2 \times 2$ grid used to assess the performance of a binary classifier by comparing actual labels against predicted labels.

|  | Predicted: Legitimate | Predicted: Fraudulent |
| --- | --- | --- |
| **Actual: Legitimate** | **True Negative (TN)** <br>

<br> *(Legitimate correctly predicted as legitimate)* | **False Positive (FP)** <br>

<br> *(Legitimate incorrectly predicted as fraudulent)* |
| **Actual: Fraudulent** | **False Negative (FN)** <br>

<br> *(Fraudulent incorrectly predicted as legitimate)* | **True Positive (TP)** <br>

<br> *(Fraudulent correctly predicted as fraudulent)* |

* **Precision**: Out of all the positive predictions the model made, how many were actually correct.
* High precision = lower false positive rate (fewer false alarms).
* Formula:

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$




* **Recall (Sensitivity)**: Out of all the actual positive cases in the dataset, how many did the model manage to find and catch.
* High recall = lower false negative rate (fewer missed cases).
* Formula:

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$




* **F1-Score**: The harmonic mean of precision and recall.
* It gives equal weight to both metrics, making it a great single-number metric for evaluating models trained on imbalanced datasets.
* Formula:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$





```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# returns 4 arrays: training data, test data, training label, test label
# splits the data into two categories (training data and test data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# confusion_matrix() returns a 2D array:
# [ [ True Negatives (top-left),      False Positives (top-right) ],
#   [ False Negatives (bottom-left),  True Positives (bottom-right) ] ]
print(confusion_matrix(y_test, y_pred))

# classification_report() outputs a text report showing:
# precision, recall, f1-score, and support for each individual class
print(classification_report(y_test, y_pred))

```