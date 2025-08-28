📌 Fraud Detection with Exploratory Data Analysis (EDA) & Machine Learning
📖 Project Overview

This project focuses on detecting fraudulent credit card transactions using EDA and Machine Learning models.
We use the Credit Card Fraud Detection dataset, perform exploratory analysis, check class imbalance, visualize distributions, and then apply two models:

Logistic Regression

Random Forest Classifier

We also compare their performance using confusion matrices and ROC curves.

⚙️ Step 1: Import Libraries

We import the required Python libraries for:

Data handling → pandas, numpy

Visualization → matplotlib, seaborn

Modeling → sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

📂 Step 2: Load Dataset
NOTE: DATA SET SIZE IS TOO LARGE SO YOU CAN DOWNLOAD IT FROM BELOW LINK
LINK: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
We load the dataset (creditcard.csv) and check its shape & first few rows.

df = pd.read_csv("creditcard.csv")
print("Shape of dataset:", df.shape)
display(df.head())

📊 Step 3: Class Balance Check

The dataset is highly imbalanced (fraud cases are very rare).
We check distribution & visualize fraud vs. legitimate transactions.

df["Class"].value_counts()
sns.countplot(data=df, x="Class", hue="Class", palette="Set2", legend=False)

💰 Step 4: Transaction Amount Distribution

We explore how transaction amounts differ between fraud and legitimate transactions using histograms & boxplots.

sns.histplot(df["Amount"], bins=50, kde=True, color="teal")
sns.boxplot(data=df, x="Class", y="Amount", hue="Class", palette="coolwarm", legend=False)

🔎 Step 5: Correlation Heatmap

We plot a correlation heatmap to understand feature relationships.

corr = df.corr()
sns.heatmap(corr, cmap="RdBu_r", center=0)

🎯 Step 6: Feature Engineering

Define X (features) and y (target).

Apply StandardScaler to normalize values.

Split into train and test sets.

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

🤖 Step 7: Logistic Regression Model

We train a Logistic Regression model with class balancing.
Then, we check classification report, ROC-AUC, and confusion matrix.

log_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
log_clf.fit(X_train, y_train)

🌳 Step 8: Random Forest Model

We train a Random Forest Classifier with balanced weights.
Then, we evaluate it using classification metrics & confusion matrix.

rf_clf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_clf.fit(X_train, y_train)

📈 Step 9: ROC Curve Comparison

We compare the models using ROC Curves.
This helps us visualize which model better separates fraud vs. non-fraud transactions.

fpr_log, tpr_log, _ = roc_curve(y_test, log_clf.predict_proba(X_test)[:,1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_clf.predict_proba(X_test)[:,1])

✅ Results & Insights

The dataset is highly imbalanced → fraud cases are less than 1%.

Random Forest generally performs better than Logistic Regression due to its ability to handle complex feature interactions.

ROC Curve & AUC score help us compare models effectively.

📌 Future Improvements

Use SMOTE / Oversampling to handle imbalance.

Try XGBoost / LightGBM for better accuracy.

Deploy as a real-time fraud detection API.
