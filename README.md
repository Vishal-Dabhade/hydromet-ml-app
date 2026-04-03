# HydroMet ML — Extraction Efficiency Predictor

## Quick Start

```bash
# 1. Install dependencies
pip install flask scikit-learn pandas numpy joblib

# 2. Train the model
cd hydromet
python model/train_model.py

# 3. Run the web app
python app.py
# → Open http://127.0.0.1:5000
```

## Architecture

```
hydromet/
├── model/
│   ├── train_model.py       # ML training pipeline
│   ├── hydromet_model.pkl   # Saved best model (auto-generated)
│   └── metadata.json        # Metrics & feature importances
├── database/
│   └── predictions.db       # SQLite (auto-created)
├── static/
│   └── index.html           # Single-page frontend
├── app.py                   # Flask REST API
└── README.md
```

## API Endpoints

| Method | Path             | Description                        |
|--------|------------------|------------------------------------|
| POST   | /api/predict     | Predict efficiency from parameters |
| GET    | /api/history     | Retrieve past predictions          |
| GET    | /api/model-info  | Model metadata & feature importance|
| GET    | /api/stats       | Aggregate stats from DB            |

## Input Parameters

| Parameter     | Range       | Unit   |
|---------------|-------------|--------|
| temperature   | 20 – 90     | °C     |
| pH            | 0.5 – 6.0   | —      |
| reaction_time | 10 – 240    | min    |
| acid_conc     | 0.1 – 5.0   | mol/L  |
| particle_size | 10 – 500    | µm     |

## Model Performance (best model)

- R² Score:  0.9584
- MAE:       2.49 %
- RMSE:      3.16 %
- CV R²:     0.9584 ± 0.006
