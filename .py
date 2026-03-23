import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
sample_sub = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
y = train["SalePrice"]
X = train.drop(columns=["SalePrice", "Id"], errors="ignore")
X_test = test.drop(columns=["Id"], errors="ignore")
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])
model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ))
])
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42
)
model.fit(X_train, y_train)
print("Train R2:", model.score(X_train, y_train))
print("Validation R2:", model.score(X_val, y_val))
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

print("CV RMSE:", -cv_scores.mean(), "±", cv_scores.std())
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

print("CV RMSE:", -cv_scores.mean(), "±", cv_scores.std())
preds = model.predict(X_test)
preds = np.maximum(preds, 0)
preds = model.predict(X_test)
preds = np.maximum(preds, 0)
submission = sample_sub.copy()

if "SalePrice" in submission.columns:
    submission["SalePrice"] = preds
else:
    submission.iloc[:, 1] = preds

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created")
submission = sample_sub.copy()

if "SalePrice" in submission.columns:
    submission["SalePrice"] = preds
else:
    submission.iloc[:, 1] = preds

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created")