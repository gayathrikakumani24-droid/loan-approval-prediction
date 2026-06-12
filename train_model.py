import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
# Load dataset
df = pd.read_csv("E:\\full_stack\\Loan_approval\\loan_approval_dataset (1).csv")
df.columns = df.columns.str.strip()

# Remove extra spaces
df['education'] = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()
df['loan_status'] = df['loan_status'].str.strip()

# Encode
edu_encoder = LabelEncoder()
emp_encoder = LabelEncoder()
loan_encoder = LabelEncoder()

df['education'] = edu_encoder.fit_transform(df['education'])
df['self_employed'] = emp_encoder.fit_transform(df['self_employed'])
df['loan_status'] = loan_encoder.fit_transform(df['loan_status'])

# Feature engineering
df['total_assets'] = (
    df['residential_assets_value']
    + df['commercial_assets_value']
    + df['luxury_assets_value']
    + df['bank_asset_value']
)

# Balance dataset
approved = df[df['loan_status'] == 0]
rejected = df[df['loan_status'] == 1]

approved_upsampled = resample(
    approved,
    replace=True,
    n_samples=len(rejected),
    random_state=42
)

df_balanced = pd.concat([rejected, approved_upsampled])

# Features
X = df_balanced.drop(
    ['loan_status', 'loan_id'],
    axis=1
)

y = df_balanced['loan_status']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train


dt = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)

dt.fit(X_train, y_train)

model = CalibratedClassifierCV(
    estimator=dt,
    method='sigmoid'
)

model.fit(X_train, y_train)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dt.feature_importances_
})

print(
    importance_df.sort_values(
        by="Importance",
        ascending=False
    )
)

# Save model
joblib.dump(model, "trained_model.pkl")

# Save feature columns
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Model saved successfully!")