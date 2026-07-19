import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# 1. Load the dataset
print("Loading SBA dataset...")
# Make sure SBAnational.csv is in the same directory as this script
df = pd.read_csv('./data/SBAnational.csv', low_memory=False)

print(f"Dataset loaded. Shape: {df.shape}")

# 2. Define the Target Variable
# The 'MIS_Status' column tells us if the loan defaulted.
# 'P I F' = Paid In Full (0)
# 'CHGOFF' = Charged Off / Defaulted (1)
print("Cleaning target variable...")
df = df.dropna(subset=['MIS_Status']) # Drop rows where target is missing
df['target'] = df['MIS_Status'].apply(lambda x: 1 if x == 'CHGOFF' else 0)

# 3. Drop Data Leakage Columns
# These columns contain information only known AFTER the loan was approved.
# Using them is a critical error that hiring managers will instantly spot.
leakage_columns = [
    'ChgOffDate',       # Date it was charged off
    'ChgOffPrinGr',     # Amount charged off
    'BalanceGross',     # Current balance
    'MIS_Status',       # The original target string
    'DisbursementDate', # When money was sent (often after approval)
    'DisbursementGross' # Amount sent
]
df = df.drop(columns=leakage_columns)

print(f"Baseline default rate: {df['target'].mean():.2%}")

# ... [Assume previous code loaded df and created 'target' column] ...

print("Cleaning raw currency strings...")
# The SBA dataset has currency stored as strings (e.g., "$150,000.00").
# In production, our API will expect strict floats, so we clean the training data to match.
currency_cols = ['GrAppv', 'SBA_Appv']
for col in currency_cols:
    if df[col].dtype == 'str':
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

print("Engineering business risk features...")
# 1. NAICS Sector: Extract the first two digits to get the macro-industry
# We fill missing NAICS with '00' (Unknown)
df['NAICS'] = df['NAICS'].fillna(0).astype(str)
df['NAICS_Sector'] = df['NAICS'].apply(lambda x: x[:2] if len(x) >= 2 else '00')

# 2. Franchise Flag: 0 and 1 mean no franchise. Anything else is a franchise code.
df['IsFranchise'] = df['FranchiseCode'].apply(lambda x: 0 if x in [0, 1] else 1)

# 3. Real Estate Backing: SBA loans >= 240 months are backed by real estate (lower risk)
df['RealEstate'] = df['Term'].apply(lambda x: 1 if x >= 240 else 0)

# 4. SBA Guarantee Ratio: What percentage of the loan is the government backing?
df['Guarantee_Ratio'] = df['SBA_Appv'] / df['GrAppv']
df['Guarantee_Ratio'] = df['Guarantee_Ratio'].fillna(0)

# ---------------------------------------------------------
# DEFINE THE PIPELINE ARCHITECTURE
# ---------------------------------------------------------
# We explicitly define which columns the model is allowed to look at.
# This ensures we don't accidentally leak data.

numeric_features = [
    'Term', 'NoEmp', 'CreateJob', 'RetainedJob', 
    'GrAppv', 'Guarantee_Ratio'
]

categorical_features = [
    'NAICS_Sector', 'NewExist', 'UrbanRural', 'IsFranchise', 'RealEstate',
    'RevLineCr', 'LowDoc'
]

# Clean up messy categorical text before pipelining
for col in categorical_features:
    df[col] = df[col].astype(str)

print("Splitting data...")
X = df[numeric_features + categorical_features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------
# BUILD THE COLUMN TRANSFORMER
# ---------------------------------------------------------
# This handles missing values and encoding automatically. 
# CRITICAL: handle_unknown='ignore' ensures the API doesn't crash if a user 
# inputs an industry code (NAICS) that wasn't in the training data.

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ---------------------------------------------------------
# BUILD THE FULL MLOPS PIPELINE
# ---------------------------------------------------------
# Scale_pos_weight helps handle class imbalance (more approvals than defaults)
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    ))
])

print("Training the pipeline (this takes a moment)...")
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------
# EVALUATION & SERIALIZATION
# ---------------------------------------------------------
print("Evaluating model...")
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Serializing pipeline for production API...")
# We save the ENTIRE pipeline, not just the XGBoost model.
joblib.dump(pipeline, 'sba_pipeline.joblib')
print("Saved as 'sba_pipeline.joblib'. Ready for deployment.")