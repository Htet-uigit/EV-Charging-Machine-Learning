# EV Charging & Grid Optimization Dashboard

A Streamlit dashboard for forecasting EV charging demand using ARIMA, SARIMA, and LSTM models.

## Project Structure

```
ev_dashboard/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit theme & server settings
├── preprocessed_ev_data.csv   # (Your dataset — place here)
└── README.md
```

## Setup & Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a **GitHub repository** (public or private).
2. Place your `preprocessed_ev_data.csv` in the same folder as `app.py`.
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
4. Click **New app** → select your repo, branch, and set `app.py` as the main file.
5. Click **Deploy** — done!

> **Note:** If `preprocessed_ev_data.csv` is not present, the app automatically
> generates a synthetic dataset so it will always run without errors.

## Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPIs, demand trend chart, project pipeline |
| 📊 Model Results | Per-model metrics, code snippets, forecast charts |
| 🧪 Live Simulator | Adjust inputs and see real-time demand prediction |

## Models

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| ARIMA (2,1,2) | 23.4992 | 26.2170 | 0.0109 |
| **SARIMA (1,1,1)×(1,1,1)₂₄** | **3.7551** | **4.7835** | **0.9671** |
| LSTM (64-unit, 24h lookback) | 4.0824 | 5.2351 | 0.9608 |
