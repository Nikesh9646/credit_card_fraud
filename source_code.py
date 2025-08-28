# =========================
# Fraud Detection Project with EDA
# =========================

# 1) Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Make sure plots show up
%matplotlib inline  
sns.set(style="whitegrid", palette="pastel")

# =========================
# 2) Load dataset
# =========================
df = pd.read_csv("creditcard.csv")
print("Shape of dataset:", df.shape)
display(df.head())

# =========================
# 3) Class balance check
# =========================
print("\nClass distribution:")
print(df["Class"].value_counts())
print("Fraud %:", round(df["Class"].mean()*100, 3), "%")

plt.figure(figsize=(5,4))
sns.countplot(data=df, x="Class", hue="Class", palette="Set2", legend=False)
plt.title("Fraud (1) vs Legit (0)")
plt.show()

# =========================
# 4) Transaction Amount Distribution
# =========================
plt.figure(figsize=(8,4))
sns.histplot(df["Amount"], bins=50, kde=True, color="teal")
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount ($)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(data=df, x="Class", y="Amount", hue="Class", palette="coolwarm", legend=False)
plt.title("Transaction Amount by Class (Fraud vs Legit)")
plt.show()

# =========================
# 5) Correlation Heatmap
# =========================
plt.figure(figsize=(12,8))
corr = df.corr()
sns.heatmap(corr, cmap="RdBu_r", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

# =========================
# 6) Features / Target
# =========================
X = df.drop("Class", axis=1)
y = df["Class"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# 7) Logistic Regression
# =========================
log_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
log_clf.fit(X_train, y_train)
y_pred_log = log_clf.predict(X_test)

print("\n=== Logistic Regression Report ===")
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, log_clf.predict_proba(X_test)[:,1]))

cm = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# =========================
# 8) Random Forest
# =========================
rf_clf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("\n=== Random Forest Report ===")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:,1]))

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest - Confusion Matrix")
plt.show()

# =========================
# 9) ROC Curve
# =========================
fpr_log, tpr_log, _ = roc_curve(y_test, log_clf.predict_proba(X_test)[:,1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_clf.predict_proba(X_test)[:,1])

plt.figure(figsize=(6,6))
plt.plot(fpr_log, tpr_log, label="Logistic Regression", color="blue")
plt.plot(fpr_rf, tpr_rf, label="Random Forest", color="green")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Fraud Detection")
plt.legend()
plt.show()
