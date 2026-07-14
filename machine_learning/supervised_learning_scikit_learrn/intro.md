# What is machine learning?
- computer can make decisions from data
- without being explicitly programmed

## Unsupervised Learning
- uncovering hidden patterns from unlabeled data
    - groupng customers into distinct categories (Clustering) 
    
## Supervised Learning
- values to be predicted are already known
- Aim: Predict the values of unseen data, given the features

### Types of supervised learning

1. Classification
- used to predict the label, category, of an observation
    - predict whether a bank transaction is fraudulent or not   
- Binary classification:
    - fraudulent transcation 
    - non-fraudulent transaction

2. Regression
- used to predict continuous values
    - use the number of bedrooms and the size of the property to predict the target price of property


## Naming conventions
- Feature = predictor variable = independent variable
- Target variable = dependent variable = response variable

## Before you supervise learning:

- No missing values
- data in numeric format
- data stored in pandas dataframes or nmumpy array
- Perform Exploratory Data Analysis (EDA) first

## Scikit-learn syntax
```python
from sklearn.module import Model
model = Model()
model.fit(X, y)     # X = array of features, y = array of values
predictions = model.predict(X_new)
```