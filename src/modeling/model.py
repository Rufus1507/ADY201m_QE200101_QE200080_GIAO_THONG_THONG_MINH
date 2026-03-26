import pandas as pd
import numpy as np
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# ================== LOAD DATA ==================
conn = sqlite3.connect(r"C:\Users\Admin\Desktop\3\ADY201m\data\clean\data_traffic_clean.db")
df = pd.read_sql("SELECT * FROM traffic_data_clean", conn)

# ================== FEATURE ENGINEERING ==================
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

le_location = LabelEncoder()
df['location'] = le_location.fit_transform(df['location'])

le_target = LabelEncoder()
df['traffic_level'] = le_target.fit_transform(df['traffic_level'])

df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 7<=x<=9 or 17<=x<=19 else 0)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# ================== FEATURES ==================
X = df[['location','hour_sin','hour_cos','is_peak_hour','day_of_week']]
y = df['traffic_level']

# ================== SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================== BALANCE TRAIN ==================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Train after SMOTE:", np.bincount(y_train))
print("Test original:", np.bincount(y_test))

# ================== BALANCE TEST ==================
test_data = pd.concat([X_test, y_test], axis=1)

df_0 = test_data[test_data['traffic_level'] == 0]
df_1 = test_data[test_data['traffic_level'] == 1]
df_2 = test_data[test_data['traffic_level'] == 2]

max_size = max(len(df_0), len(df_1), len(df_2))

df_0_up = resample(df_0, replace=True, n_samples=max_size, random_state=42)
df_1_up = resample(df_1, replace=True, n_samples=max_size, random_state=42)
df_2_up = resample(df_2, replace=True, n_samples=max_size, random_state=42)

test_balanced = pd.concat([df_0_up, df_1_up, df_2_up]).sample(frac=1, random_state=42)

X_test_bal = test_balanced.drop('traffic_level', axis=1)
y_test_bal = test_balanced['traffic_level']

print("Test balanced:", np.bincount(y_test_bal))

# ================== MODELS ==================
log_model = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ================== PREDICT ==================
y_pred_log = log_model.predict(X_test_bal)
y_pred_rf = rf_model.predict(X_test_bal)

# ================== RESULTS ==================
print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test_bal, y_pred_log))
print(classification_report(y_test_bal, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test_bal, y_pred_log))

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test_bal, y_pred_rf))
print(classification_report(y_test_bal, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test_bal, y_pred_rf))

# ================== SO SÁNH TEST GỐC ==================
print("\n=== TEST GỐC ===")
print("RF Accuracy:", accuracy_score(y_test, rf_model.predict(X_test)))

print("\n=== TEST BALANCED ===")
print("RF Accuracy:", accuracy_score(y_test_bal, y_pred_rf))