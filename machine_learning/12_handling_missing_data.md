# 1. Diagnosing missing data

- a row with no value for a feature
- typically happens due to unrecorded observations or corrupted records.


```python
print(music_df.isna().sum().sort_values())
```

# 2. Strategies for Handling Missing Data

## Strategy A: Dropping Rows (Listwise Deletion)
- If the missing values account for a tiny fraction of your entire dataset (e.g., less than 5%), it is often safe to simply drop those rows.

```python
music_df = music_df.dropna(subset=["genre", "popularity", "loudness", "liveness", "tempo"])
```

## Strategy B: Imputation (Filing the Gaps)

- Imputation means replacing missing values with calculated mathematical guesses so you don't throw away valuable data.
- Numerical Data: Commonly replaced using the mean or median value of that column.
- Categorical Data: Commonly replaced using the mode (the most frequent value) of that column.


### ⚠️ Critical Rule: Prevent Data Leakage
You must always split your data into Training and Test sets before imputing values. If you calculate the mean using the entire dataset, information from the test set "leaks" into your training phase, rendering your evaluation metrics artificially inflated and unrealistic.

# 3. Imputation Mechanics via SimpleImputer
- Scikit-learn provides the SimpleImputer class to handle this process programmatically. Objects that transform data like this are known as transformers.

### Step-by-Step Execution:
1. Split the columns into categorical and numerical structures.
2. Perform the train_test_split independently across them to safeguard the test partitions.
3. fit_transform the training data: This calculates the imputation metric (e.g., mode or mean) based only on the training split, then fills the gaps.
4. transform the test data: This fills gaps in the test set using the metric calculated from the training split. Do not re-fit on the test data.

```python
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# 1. Split targets: categorical and numerical
X_cat = music_df['genre'].values.reshape(-1, 1) # -1 is a placeholder it figures out rows and automatically becomes (1000,1)
X_num = music_df.drop(['genre', 'popularity'], axis=1).values   # genre dropped because it's in categorical feature, popularity = target
y = music_df['popularity'].values

# 2. Train/Test splits

X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_Cat, y, test_size=0.2, random_state=12)
X_train_num, X_test_num, y_train, y_test = train_test_split(X_Num, y, test_size=0.2, random_state=12)

# 3. Impute Categorical Values using the mode
imp_cat = SimpleImputer(strategy="most_freq")
X_train_cat = imp_cat.fit_transform(X_train_cat)    # counts and learns which genre is the most frequent then overwrites all NaN values
X_test_cat = imp_cat.fit(X_test_cat)  # uses the same mode learned from the training step & plugs it into any missing slots in X_test_cat

# 4. Impute the Numerical Values using the mean (default)
imp_num = SimpleImputer()
X_train_num = imp_cat.fit_transform(X_train_num)
X_test_num = imp_cat.fit(X_test_num)
```
# 4. Transitioning to Pipelines
 - Manually executing different split operations, scaling structures, and tracking matching indices across multiple arrays quickly becomes messy and error-prone.
 - To clean this up, scikit-learn uses a Pipeline object. A pipeline allows you to chain a series of data transformations together with a final model into a single workflow block.

 ```python 
 from sklearn.pipeline import Pipeline

 # Drop rows for columns with low missing rates
 music_df = music_df.dropna(["genre", "popularity", "loudness", "liveness", "tempo"])

 # Convert target category to Binary 1 or 0
 music_df["genre"] = np.where(music_df["genre"] == "Rock", 1, 0)

 X = music_df.drop("genre", axis=1).values
 y = music_df["genre"].values
 ```

 - To build a pipeline we construct a list of tuples with string specified as their names

 ```python

steps = [("imputation", SimpleImputer()),
            "logistic_regression", LogisticRegression()]
pipeline=pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)  # compute for accuracy
 ```