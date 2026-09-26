# ============================================
# CUSTOMER CHURN ANALYSIS PROJECT
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================
# STEP 1: LOAD DATASET
# ============================================

print("Loading Dataset...")

data = pd.read_csv("customer_churn.csv")

print("\nDataset Loaded Successfully!")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(data.head())

# ============================================
# STEP 2: CHECK DATASET INFORMATION
# ============================================

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# ============================================
# STEP 3: DATA PREPROCESSING
# ============================================

# Convert TotalCharges column to numeric
data['TotalCharges'] = pd.to_numeric(
    data['TotalCharges'],
    errors='coerce'
)

# Fill missing values
data['TotalCharges'].fillna(
    data['TotalCharges'].median(),
    inplace=True
)

# Drop customerID column
data.drop('customerID', axis=1, inplace=True)

# ============================================
# STEP 4: ENCODE CATEGORICAL DATA
# ============================================

# Convert all categorical columns into numbers

categorical_columns = data.select_dtypes(include=['object']).columns

for column in categorical_columns:
    label_encoder = LabelEncoder()
    data[column] = label_encoder.fit_transform(data[column].astype(str))

print("\nCategorical Data Encoded Successfully!")

# ============================================
# STEP 5: DEFINE FEATURES AND TARGET
# ============================================

X = data.drop("Churn", axis=1)

y = data["Churn"]

# ============================================
# STEP 6: SPLIT TRAINING AND TESTING DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

# ============================================
# STEP 7: TRAIN MACHINE LEARNING MODEL
# ============================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# ============================================
# STEP 8: MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

# ============================================
# STEP 9: MODEL EVALUATION
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================
# STEP 10: CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ============================================
# STEP 11: FEATURE IMPORTANCE
# ============================================

importance = model.feature_importances_

features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)

# ============================================
# STEP 12: FEATURE IMPORTANCE GRAPH
# ============================================

plt.figure(figsize=(10, 6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df
)

plt.title("Customer Churn Feature Importance")

plt.show()

# ============================================
# PROJECT COMPLETED
# ============================================

print("\nCustomer Churn Analysis Completed Successfully!")
