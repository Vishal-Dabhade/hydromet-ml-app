"""
Gold Cyanidation — Extraction Efficiency Predictor
====================================================
Trains a regression model on real gold cyanidation process data.
Features: Temperature, pH, Reaction Time, NaCN Concentration,
          Particle Size, Dissolved O2
Target:   Au Extraction Efficiency (%)
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ── Load dataset ─────────────────────────────────────────────────────────────
print("Loading gold cyanidation dataset...")
df = pd.read_csv("model/gold_cyanidation_data.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(df.describe().round(2))

FEATURES = [
    "temperature_C",
    "pH",
    "reaction_time_hr",
    "nacn_conc_gL",
    "particle_size_um",
    "dissolved_o2_mgL",
]
TARGET = "Au_extraction_pct"

X = df[FEATURES]
y = df[TARGET]

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train)} rows  |  Test: {len(X_test)} rows")

# ── Model definitions ─────────────────────────────────────────────────────────
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_split=5,
        n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    "Ridge Regression": Ridge(alpha=1.0),
}

# ── Train & evaluate ──────────────────────────────────────────────────────────
print("\nTraining models...\n")
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_r2   = -np.inf
best_key  = None
best_pipe = None

for name, reg in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", reg)])
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {
        "R2":         round(r2,   4),
        "MAE":        round(mae,  4),
        "RMSE":       round(rmse, 4),
        "CV_R2_mean": round(cv_r2.mean(), 4),
        "CV_R2_std":  round(cv_r2.std(),  4),
    }
    print(f"{name:22s}  R²={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}  CV={cv_r2.mean():.4f}±{cv_r2.std():.4f}")

    if r2 > best_r2:
        best_r2   = r2
        best_key  = name
        best_pipe = pipe

print(f"\n✓ Best model: {best_key} (R²={best_r2:.4f})")

# ── Feature importances ───────────────────────────────────────────────────────
reg_model = best_pipe.named_steps["model"]
if hasattr(reg_model, "feature_importances_"):
    importances = dict(zip(FEATURES, reg_model.feature_importances_.round(4)))
else:
    coefs = np.abs(reg_model.coef_)
    importances = dict(zip(FEATURES, (coefs / coefs.sum()).round(4)))

print("\nFeature importances:")
for k, v in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {k:25s}  {v:.4f}")

# ── Save model & metadata ─────────────────────────────────────────────────────
joblib.dump(best_pipe, "model/hydromet_model.pkl")

metadata = {
    "best_model":  best_key,
    "metal":       "Gold (Au)",
    "process":     "Cyanidation (NaCN)",
    "features":    FEATURES,
    "metrics":     results[best_key],
    "all_metrics": results,
    "importances": importances,
    "data_stats": {
        f: {
            "min":  round(float(X[f].min()),  3),
            "max":  round(float(X[f].max()),  3),
            "mean": round(float(X[f].mean()), 3),
        }
        for f in FEATURES
    },
}

with open("model/metadata.json", "w") as fh:
    json.dump(metadata, fh, indent=2)

print("\n✓ Model saved  →  model/hydromet_model.pkl")
print("✓ Metadata saved  →  model/metadata.json")
print("\nFinal metrics:")
print(json.dumps(results[best_key], indent=2))