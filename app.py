import pickle
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ─── Chemins vers les fichiers du modèle ───────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "football_potentiel_model.pkl"
META_PATH  = BASE_DIR / "football_potentiel_metadata.json"

# ─── Chargement du modèle au démarrage ─────────────────────────────────────────
print("[INFO] Chargement du modele...")
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

FEATURE_COLS = META["feature_columns"]
CLASSES      = META["classes"]
MEDIANS      = META["medians_for_imputation"]
POS_COLS     = META["pos_columns"]
FOOT_COLS    = META["foot_columns"]

print(f"[OK] Modele pret — classes: {CLASSES}")


# ─── Helper : construire le vecteur de features ─────────────────────────────────
def build_feature_vector(data: dict) -> pd.DataFrame:
    """
    Accepte un dict avec les champs bruts du formulaire et retourne
    un DataFrame avec les 25 features dans le bon ordre.
    """
    def to_float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    age               = to_float(data.get("age"),               MEDIANS.get("age", 25))
    height            = to_float(data.get("height"),            MEDIANS.get("height", 180))
    total_goals       = to_float(data.get("total_goals"))
    total_assists     = to_float(data.get("total_assists"))
    total_minutes     = to_float(data.get("total_minutes"))
    goal_per_90       = to_float(data.get("goal_assist_per_90"))
    matches_count     = to_float(data.get("matches_count"))
    top_league_ratio  = to_float(data.get("top_league_ratio"))
    market_value      = to_float(data.get("market_value_t"))
    growth_rate       = to_float(data.get("growth_rate"))

    # Indicateurs de manque → 0 (toutes les valeurs sont fournies via le formulaire)
    missing_indicators = {
        "total_goals_missing":         0,
        "total_assists_missing":       0,
        "total_minutes_missing":       0,
        "matches_count_missing":       0,
        "goal_assist_per_90_missing":  0,
        "top_league_ratio_missing":    0,
        "market_value_t_missing":      0,
        "growth_rate_missing":         0,
        "foot_missing":                0,
    }

    # Position (one-hot, 4 colonnes)
    position = data.get("main_position", "Attack")
    all_positions = ["Attack", "Defender", "Goalkeeper", "Midfield"]
    pos_ohe = {f"pos_{p}": (1 if p == position else 0) for p in all_positions}

    # Pied (one-hot, drop_first=True → "both" = référence)
    foot = data.get("foot", "right")
    foot_ohe = {
        "foot_left":  1 if foot == "left"  else 0,
        "foot_right": 1 if foot == "right" else 0,
    }

    row = {
        "age":                age,
        "height":             height,
        "total_goals":        total_goals,
        "total_assists":      total_assists,
        "total_minutes":      total_minutes,
        "goal_assist_per_90": goal_per_90,
        "matches_count":      matches_count,
        "top_league_ratio":   top_league_ratio,
        "market_value_t":     market_value,
        "growth_rate":        growth_rate,
        **missing_indicators,
        **pos_ohe,
        **foot_ohe,
    }

    return pd.DataFrame([row])[FEATURE_COLS]


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        df_row = build_feature_vector(data)

        prediction = MODEL.predict(df_row)[0]
        probas     = MODEL.predict_proba(df_row)[0]
        proba_dict = {c: round(float(p), 4) for c, p in zip(CLASSES, probas)}

        # Score de confiance
        confidence = round(float(max(probas)) * 100, 1)

        return jsonify({
            "prediction":  prediction,
            "probabilities": proba_dict,
            "confidence":  confidence,
            "classes":     CLASSES,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify({
        "classes":      CLASSES,
        "feature_cols": FEATURE_COLS,
        "performance":  META.get("performance", {}),
    })


if __name__ == "__main__":
    print("[START] Football Potential Predictor — http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
