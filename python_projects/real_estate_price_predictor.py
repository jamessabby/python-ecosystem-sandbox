import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

df = pd.read_csv("datasets/usa_housing_kaggle.csv")

# print(df.head())
# print(df.info())

# PHASE 1: see if other features has correlation with Pricing column

price_corr = df.corr(numeric_only=True)
print(price_corr.loc["Price"])
sns.heatmap(price_corr, annot=True)
plt.show()                                    # features/columns have weak correlation to 'Pricing' 
print(price_corr.shape)               

# PHASE 2: find missing rows
# print(df.isnull().sum())      # no missing rows found

# PHASE 3: add a new column named House_Age

# Attempt 1 - from my stock knowledgeqq

# df["House_Age"] = 2026 - df["YearBuilt"]    
# print(df[["YearBuilt", "House_Age"]].head())    # KeyError: "['HouseAge'] not in index"; Fix: Fixed typo from HouseAge to House_Age


# Attempt 2 - from what i saw in stack overflow
# House_Age = 2026 - df["YearBuilt"]
# df = df.assign(House_Age.values) # Error: TypeError: DataFrame.assign() takes 1 positional argument but 2 were given
# print(df[["YearBuilt", "HouseAge"]].head())

# PHASE 4: Splitting and The Regularization Trap (Feature Scaling)

X = df.drop("Price", axis=1).values # training features, not including 'Price'
y = df["Price"].values  # isolated 'Price' as it is the target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)   # hide 30% of the houses so we can use them later for testing

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # looks at the training data, finds the average of each features and memorize those values
X_test_scaled = scaler.transform(X_test)    # scale the memorized training averages down to Z-scores

reg_all = LinearRegression()
ridge = Ridge(alpha=1)

reg_all.fit(X_train_scaled, y_train)
ridge.fit(X_train_scaled, y_train)

reg_pred = reg_all.predict(X_test_scaled)
ridge_pred = ridge.predict(X_test_scaled)

reg_scores = []
ridge_scores = []

reg_scores.append(reg_all.score(X_test_scaled, y_test)) # 
ridge_scores.append(ridge.score(X_test_scaled, y_test))


print(f"Linear Regression score: {reg_scores}")
print(f"Ridge score: {ridge_scores}")


