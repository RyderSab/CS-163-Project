from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Configuration
RF_MODEL_PATH = 'rf_model.joblib'
PROPHET_DICT_PATH = 'prophet_models_dict.joblib'
FUTURE_METRICS_PATH = 'df_metrics_future.csv'


def _canonical_route_key(route) -> str:
    """Match CSV/query route strings to dict keys built from pandas (often np.int64)."""
    s = str(route).strip()
    try:
        x = float(s)
        if x == int(x):
            return str(int(x))
    except (ValueError, OverflowError):
        pass
    return s


# Load assets once on startup
print("Loading models and data...")
rf_model = joblib.load(RF_MODEL_PATH)
_prophet_raw = joblib.load(PROPHET_DICT_PATH)
# Prophet dict keys are often numpy int64 from groupby('Route'); `'23' in d` is False for those keys.
prophet_models = {_canonical_route_key(k): v for k, v in _prophet_raw.items()}
df_future = pd.read_csv(FUTURE_METRICS_PATH)
df_future['Route'] = df_future['Route'].map(_canonical_route_key)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['GET'])
def predict():
    route_id = request.args.get('route_id')
    if not route_id:
        return jsonify({"error": "Missing route_id parameter"}), 400

    route_id_str = _canonical_route_key(route_id)

    # 1. Get Future Metrics
    route_data = df_future[df_future['Route'] == route_id_str]
    if route_data.empty:
        return jsonify({"error": f"Route {route_id_str} not found"}), 404

    # 2. Get Random Forest Prediction
    features = ['Avg_Boardings', 'Total_Scheduled_Trips', 'Avg_Percent_Low_Income', 'Avg_Percent_Zero_Vehicle']
    X_input = route_data[features]
    is_underserved = int(rf_model.predict(X_input)[0])

    # 3. Get Specific Prophet Forecast
    forecast_val = None
    if route_id_str in prophet_models:
        m = prophet_models[route_id_str]
        future = m.make_future_dataframe(periods=1, freq='MS')
        forecast = m.predict(future)
        forecast_val = float(forecast.iloc[-1]['yhat'])

    return jsonify({
        "route_id": route_id_str,
        "is_underserved_prediction": is_underserved,
        "underserved_score": float(route_data['Underserved_Score'].iloc[0]),
        "next_month_forecast": forecast_val,
        "metrics": route_data[features].to_dict(orient='records')[0]
    })

if __name__ == '__main__':
    # Port 8080 for Cloud Run compatibility
    app.run(host='0.0.0.0', port=8080)
