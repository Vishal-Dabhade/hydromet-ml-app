"""
HydroMet ML Backend — Gold Cyanidation
=======================================
Flask REST API for Au extraction efficiency prediction.
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib, json, sqlite3, os, traceback
from datetime import datetime
import numpy as np

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "hydromet_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "model", "metadata.json")
DB_PATH    = os.path.join(BASE_DIR, "database", "predictions.db")
STATIC_DIR = os.path.join(BASE_DIR, "static")

pipeline = joblib.load(MODEL_PATH)
with open(META_PATH) as fh:
    metadata = json.load(fh)
FEATURES = metadata["features"]

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at       TEXT NOT NULL,
                temperature_C    REAL,
                pH               REAL,
                reaction_time_hr REAL,
                nacn_conc_gL     REAL,
                particle_size_um REAL,
                dissolved_o2_mgL REAL,
                efficiency       REAL,
                confidence_low   REAL,
                confidence_high  REAL
            )
        """)
        conn.commit()

init_db()

app = Flask(__name__, static_folder=STATIC_DIR)

@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        body = request.get_json(force=True)

        mapped = {
            "temperature_C": body["temperature"],
            "pH": body["pH"],
            "reaction_time_hr": body["reaction_time"] / 60,
            "nacn_conc_gL": body["acid_conc"],
            "particle_size_um": body["particle_size"],
            "dissolved_o2_mgL": 8.0
        }

        vals = [float(mapped[f]) for f in FEATURES]

        import pandas as pd
        X = pd.DataFrame([vals], columns=FEATURES)
        import pandas as pd
        X = pd.DataFrame([vals], columns=FEATURES)

        pred = float(pipeline.predict(X)[0])
        pred = round(np.clip(pred, 0, 100), 2)

        try:
            estimators = pipeline.named_steps["model"].estimators_
            scaler     = pipeline.named_steps["scaler"]
            Xs         = scaler.transform(X)
            tree_preds = np.array([t[0].predict(Xs)[0] for t in estimators])
            lo = round(float(np.percentile(tree_preds, 10)), 2)
            hi = round(float(np.percentile(tree_preds, 90)), 2)
        except Exception:
            lo, hi = round(pred - 2.5, 2), round(pred + 2.5, 2)

        lo = round(np.clip(lo, 0, 100), 2)
        hi = round(np.clip(hi, 0, 100), 2)

        with get_db() as conn:
            conn.execute("""
                INSERT INTO predictions
                  (created_at, temperature_C, pH, reaction_time_hr,
                   nacn_conc_gL, particle_size_um, dissolved_o2_mgL,
                   efficiency, confidence_low, confidence_high)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.utcnow().isoformat(),
                vals[0], vals[1], vals[2], vals[3], vals[4], vals[5],
                pred, lo, hi,
            ))
            conn.commit()

        return jsonify({"efficiency": pred, "confidence_low": lo, "confidence_high": hi, "unit": "%"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/api/history", methods=["GET"])
def history():
    limit = min(int(request.args.get("limit", 50)), 200)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    result = []

    for r in rows:
        result.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "temperature": r["temperature_C"],
            "pH": r["pH"],
            "reaction_time": r["reaction_time_hr"] * 60,
            "acid_conc": r["nacn_conc_gL"],
            "particle_size": r["particle_size_um"],
            "efficiency": r["efficiency"],
            "confidence_low": r["confidence_low"],
            "confidence_high": r["confidence_high"]
        })

    return jsonify(result)

@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify(metadata)

@app.route("/api/stats", methods=["GET"])
def stats():
    with get_db() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as total,
                   ROUND(AVG(efficiency),2) as avg_eff,
                   ROUND(MIN(efficiency),2) as min_eff,
                   ROUND(MAX(efficiency),2) as max_eff
            FROM predictions
        """).fetchone()
    return jsonify(dict(row))

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/<path:p>")
def assets(p):
    return send_from_directory(STATIC_DIR, p)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"HydroMet API running on port {port}")
    app.run(host="0.0.0.0", port=port)