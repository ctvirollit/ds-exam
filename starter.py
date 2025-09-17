# Random Forest Entity Matching (1-hour exam) — Starter
# Instructions:
# 1) pip install pandas scikit-learn (and optionally rapidfuzz or jellyfish)
# 2) Implement featurize() to generate 4–8 features from the raw columns.
# 3) Train a RandomForestClassifier and report precision, recall, and F1 on a holdout set.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Load data
df = pd.read_csv("pairs_small.csv")

# === TODO: Feature engineering ===
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame()
    # 1) Exact matches (booleans -> ints)
    X["last_name_exact"] = (df["a_last_name"].str.lower()==df["b_last_name"].str.lower()).astype(int)
    X["dob_exact"] = (df["a_dob"]==df["b_dob"]).astype(int)
    # 2) First name similarity (simple normalized overlap as placeholder; replace with rapidfuzz if available)
    def simple_sim(a, b):
        a = str(a).lower().strip()
        b = str(b).lower().strip()
        if not a and not b: return 1.0
        s = set(a)
        t = set(b)
        if not s or not t: return 0.0
        return len(s & t) / len(s | t)
    X["first_name_sim"] = [simple_sim(a,b) for a,b in zip(df["a_first_name"], df["b_first_name"])]
    # 3) City token overlap
    X["city_exact"] = (df["a_city"].str.lower()==df["b_city"].str.lower()).astype(int)
    # 4) Mobile last-4 match
    X["mobile_last4"] = (df["a_mobile"].str[-4:]==df["b_mobile"].str[-4:]).astype(int)
    return X

X = featurize(df)
y = df["is_match"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Model
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

# Evaluate
y_prob = clf.predict_proba(X_test)[:,1]
# Simple threshold at 0.5 (candidate may tune)
y_pred = (y_prob >= 0.5).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
print("F1:", f1_score(y_test, y_pred))
