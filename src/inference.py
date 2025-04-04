# inference.py
import joblib
import numpy as np
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from scipy.stats import boxcox

# Initialize Flask App
app = Flask(__name__)

# Setup Logging
log_filename = f"logs/inference_{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    filename=log_filename, 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Also log to console 
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(console_handler)

# Load preprocessing artifacts
scaler = joblib.load("src/scaler.pkl")
lambda_bc = joblib.load("src/lambda_bc.pkl")

# Load models from local files
models = {
    "linear_regression": joblib.load("models/linear_regression.pkl"),
    "decision_tree": joblib.load("models/decision_tree.pkl"),
    "random_forest": joblib.load("models/random_forest.pkl")
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    model_name = data.get("model", "linear_regression")

    if model_name not in models:
        logging.warning(f"Invalid model requested: {model_name}")
        return jsonify({"error": "Invalid model name"}), 400

    try: 
        input_value = data["features"][0] + 1e-6
        input_boxcox = boxcox(input_value, lmbda=lambda_bc)
        input_scaled = scaler.transform([[input_boxcox]])

        log_salary = models[model_name].predict(input_scaled)
        predicted_salary = np.expm1(log_salary)

        logging.info(f"Model: {model_name}, Features: {data['features']}, Predicted Salary: {predicted_salary}")

        return jsonify({
            "model": model_name,
            "predicted_salary": float(predicted_salary)
        })
    
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)

@app.route("/models", methods=["GET"])
def get_models():
    logging.info("Model list requested.")
    return jsonify({"available_models": list(models.keys())})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)