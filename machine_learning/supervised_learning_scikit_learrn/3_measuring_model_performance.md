# K-Nearest Neighbors (KNN) 
- predict the label of any data point by looking at the k (ex: k = 3) closest labeled data points
- the label that will be put on the unlabelled data depends on the majority labels around it

### Accuracy 

- correct predictions / total number of predictions
- could compute the accuracy on the data used to fit the classifier  
    - not indicative on the ability to generalize (Just because your model gets a perfect 100% score on the data it already trained on, that does not guarantee it will actually work in the real world when it's given new data)
- Testing a model on the same data used to train it only tells you how well the model memorized that specific dataset.
- To prove your model has actually learned the general concept (generalized), you must test it on a completely fresh, separate batch of data (the test set) that it has never laid eyes on before.

```python
import matplotlib.pyplot as plt

# returns 4 arrays: training data, test data, training label, test label
from sklearn.model_selection import train_test_split

# train test split = splits the data into two categories (training data and test data)
# Parameters (both needs to be converted to numpy arrays):
# X = 2D array of features from the dataframe (factors the affect the target value)
# y = 1D array of values from the dataframe (the target value or the output based on the features)
#   = a single column with the same number of observations as the feature data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# X_train (Practice Questions): The math problems on the study guide.

# y_train (Practice Answers): The step-by-step solutions at the back of the study guide. The student uses these to learn how to solve the problems.

# X_test (Exam Questions): The brand-new questions on the actual exam.

# y_test (Exam Answer Key): The correct answers to the exam. You keep this hidden in your desk. The student hands in their guesses (y_pred), and you compare them to y_test to calculate their final grade.

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```
### Model Complexity

- larger k = less complex model = can cause underfittting
- smaller k = more complex model = can lead to overfitting

```python

train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1,26)
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neigbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

plt.figure(figsize(8, 6))
plt.title("KNN: Varying number of neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Tesing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()

```
