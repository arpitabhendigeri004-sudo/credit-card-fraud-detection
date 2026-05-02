import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Load Dataset
# -------------------------------
print("Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

# -------------------------------
# 2. Features & Target
# -------------------------------
X = df.drop('Class', axis=1)
y = df['Class']

print("Dataset Shape:", df.shape)

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# -------------------------------
# 4. Scaling
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nScaling Done Successfully!")

# -------------------------------
# 5. Before SMOTE
# -------------------------------
print("\nBefore SMOTE:")
print("Class distribution:", np.bincount(y_train))

# -------------------------------
# 6. Apply SMOTE
# -------------------------------
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print("Class distribution:", np.bincount(y_train_resampled))

# -------------------------------
# 7. Train Model
# -------------------------------
print("\nTraining model... please wait ⏳")

model = RandomForestClassifier(
    n_estimators=30,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_resampled, y_train_resampled)

print("Model Training Completed!")

# -------------------------------
# 8. Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 9. Evaluation
# -------------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# -------------------------------
# 10. Confusion Matrix Visualization
# -------------------------------
plt.figure(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")

# -------------------------------
# SAVE MODEL BEFORE SHOW
# -------------------------------
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and scaler saved successfully!")

plt.show()

print("\nConfusion matrix saved in outputs folder!")
import joblib

# -------------------------------
# 11. Save Model
# -------------------------------
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and scaler saved successfully!")
import joblib
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save model and scaler
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and scaler saved successfully in 'models/' folder!")