import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

def train_model():

    df = pd.read_csv("E:\\full_stack\\Loan_approval\\loan_approval_dataset (1).csv")
    df.columns = df.columns.str.strip()

    le = LabelEncoder()

    df['education'] = le.fit_transform(df['education'])
    df['self_employed'] = le.fit_transform(df['self_employed'])
    df['loan_status'] = le.fit_transform(df['loan_status'])

    df['total_assets'] = (
        df['residential_assets_value']
        + df['commercial_assets_value']
        + df['luxury_assets_value']
        + df['bank_asset_value']
    )

    approved = df[df['loan_status'] == 0]
    rejected = df[df['loan_status'] == 1]

    approved_upsampled = resample(
        approved,
        replace=True,
        n_samples=len(rejected),
        random_state=42
    )

    df_balanced = pd.concat(
        [rejected, approved_upsampled]
    )

    X = df_balanced.drop(
        ['loan_status', 'loan_id'],
        axis=1
    )

    y = df_balanced['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    dt = DecisionTreeClassifier(
        criterion='entropy',
        class_weight='balanced',
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )

    dt.fit(X_train, y_train)

    model = CalibratedClassifierCV(
        estimator=dt,
        method='sigmoid'
    )

    model.fit(X_train, y_train)
    print(le.classes_)
    return model, X.columns, df