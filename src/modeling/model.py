import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# ví dụ SQLite
import sqlite3
conn = sqlite3.connect("data/clean/data_traffic_clean.db")
df = pd.read_sql("SELECT * FROM traffic_data_clean", conn)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
le_location = LabelEncoder()
df['location'] = le_location.fit_transform(df['location'])
le_target = LabelEncoder()
df['traffic_level'] = le_target.fit_transform(df['traffic_level'])
y = df['traffic_level']
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 7<=x<=9 or 17<=x<=19 else 0)
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x>=5 else 0)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
# THOANG = 0, DONG = 1 (thường là vậy)
X = df[[
    'location',
    'hour_sin',
    'hour_cos',
    'is_peak_hour', 
    'day_of_week'
    # 'is_weekend',
    # 'current_speed_kmh',
    # 'free_flow_speed_kmh',r
    # 'speed_ratio',
    # 'confidence'
]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Logistic Regression
log_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))
# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
acc_log = accuracy_score(y_test, y_pred_log)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\n=== Comparison ===")
print("Logistic:", acc_log)
print("Random Forest:", acc_rf)
print("\nConfusion Matrix (Logistic):")
print(confusion_matrix(y_test, y_pred_log))

print("\nConfusion Matrix (RF):")
print(confusion_matrix(y_test, y_pred_rf))
import matplotlib.pyplot as plt

importances = rf_model.feature_importances_
features = X.columns

indices = np.argsort(importances)

plt.barh(np.array(features)[indices], importances[indices])
plt.title("Feature Importance")
plt.show()