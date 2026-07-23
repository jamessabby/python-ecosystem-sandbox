### Classifying labels of unseen data
1. Build a model
2. Model learns from the labeled data we pass to it
3. Pass unlabeled data to the model as input
4. Model predicts the labels of unseen data

- Labeled data = training data

### k-Nearest neighbors
- predict the label of a data point by
    - looking at the k closest labeled data points 
    - taking a majority vote

```python
from sklearn.neighbors import KNeighborsClassifier

X = churn_df[["total_day_charge", "total_eve_charge"]].values()

y = churn_df["churn"].values

print(X.shape, y.shape)

knn = KneighborsClassifier(n_neighbors=15)

knn.fit(X, y)   # 

predictions = knn.predict(X_new)
print("Predictions: {}".format(y_pred)) 
```