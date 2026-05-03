# Predictive Maintenance & Remaining Useful Life (RUL) Estimation

Predicting the Remaining Useful Life of turbofan engines using the NASA C-MAPSS dataset.

## Problem Statement
In industrial settings, unexpected engine failure causes costly downtime. This project builds a machine learning pipeline that predicts how many cycles an engine has left before failure — enabling proactive maintenance scheduling.

## Dataset
- **Source**: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
- **Size**: 20,631 records across 100 engines
- **Features**: 21 sensor readings per cycle

## Approach
1. Computed RUL as target variable (max_cycle - current_cycle)
2. Dropped zero-variance sensors (sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19)
3. Engineered rolling mean and rolling std features (window=5) to capture degradation trends
4. Applied time-aware train-test split (80/20) to prevent data leakage
5. Trained and compared Linear Regression vs Random Forest
6. Applied SHAP TreeExplainer for model explainability

## Results

| Model | RMSE | R2 |
|---|---|---|
| Linear Regression | 53.71 | 0.5339 |
| Random Forest | 52.29 | 0.5581 |

**Best Model**: Random Forest  
**Top Degradation Sensor**: sensor_4_roll_mean (identified via SHAP)

## Key Visualisations
- Sensor degradation curves over engine lifetime
- SHAP feature importance (bar + beeswarm)
- Engine 1 — Actual vs Predicted RUL over full lifetime

## Project Structure
predictive-maintenance-rul/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── models/
│   ├── rf_rul_model.pkl
│   └── scaler.pkl
├── predictive_maintenance.ipynb
├── requirements.txt
└── README.md
## Tech Stack

- **Language**: Python
- **ML**: Scikit-learn (Linear Regression, Random Forest)
- **Explainability**: SHAP
- **Data**: Pandas, NumPy
- **Visualisation**: Matplotlib, Seaborn
- **Model Persistence**: Joblib

## How to Run

```bash
git clone https://github.com/PragnyaGuduri/predictive-maintenance-rul.git
cd predictive-maintenance-rul
pip install -r requirements.txt
jupyter notebook predictive_maintenance.ipynb
```