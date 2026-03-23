import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import joblib

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

y = np.log1p(train["SalePrice"])
X = train.drop(columns=["SalePrice", "Id"], errors="ignore")
X_test = test.drop(columns=["Id"], errors="ignore")

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)),
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
model.fit(X_train, y_train)

print("Train R2:", model.score(X_train, y_train))
print("Val R2:", model.score(X_val, y_val))

cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
print("CV RMSE (log scale):", -cv_scores.mean(), "±", cv_scores.std())

# Save the model
joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")

preds = model.predict(X_test)
preds = np.expm1(preds)
preds = np.maximum(preds, 0)

submission = sample_sub.copy()
if "SalePrice" in submission.columns:
    submission["SalePrice"] = preds
elif submission.shape[1] == 2:
    submission.iloc[:, 1] = preds
else:
    raise ValueError("Unexpected sample_submission format")

submission.to_csv("submission.csv", index=False)
print("submission.csv created")
