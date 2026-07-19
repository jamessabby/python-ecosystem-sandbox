import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# PHASE 1 - Ingestion (loading the data) and feature isolation

df = pd.read_csv("python_projects/datasets/github_bugs.csv")

# for label_num in [0, 1, 2]:
#     print(f"\n-----EXAMPLES FOR CLASS {label_num}-----")
#     # Get the first 5 titles belonging to this label 
#     sample_titles = df[df['label'] == label_num]['title'].head(5)
#     for title in sample_titles:
#         print(f" - {title}")

# Class 0 = Bugs; Class 1 = Features; Class = doc requests

# print(df.head())  
# print(df.info())

# This function counts the appearnce of a specific words from the dataset and turn it into 0's and 1's
# max_features - looks through the  top 1000 words from the dataset
# stop_words = avoids words that are too common like "is" and "are"
# max_df = ignore the words that appeared in more than 70% of issues (too common)
# min_df = ignore the words that appeared in less than 5 issues (too rare)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', max_df=0.70, min_df=5) 

count_vec = vectorizer.fit_transform(df['title'])   # actually trains and scales the data and applies the word counter algorithm

print(f"Dataset columns: ", df.columns)

X = count_vec           # treat as features
y = df['label'].values  # target value
 
# PHASE 2 - predict the label of the issue (bug / feature / doc request)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression(class_weight='balanced')

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)    # make predictions on the test test based on logistic regression

# Explicitly calculate the accuracy score
acc = accuracy_score(y_test, y_pred)

# Generate a breakdown report of the model's performance
print("-----Classification Report----")
print(classification_report(y_test, y_pred))

# PHASE 3 - Model serialization
# saves the trained pipeline so we don't need to train 450,000 rows every single time we use that to a separate script

joblib.dump(vectorizer, "tfid_vectorizer.joblib")
joblib.dump(log_reg, 'logistic_regression_model.joblib')
print('Pipeline commands successfully serialized!')