from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# Căi către model și scaler
MODEL_PATH = 'bilstm_model.keras'
SCALER_PATH = 'scaler.save'

# Încarcă modelul și scalerul
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "sequence" not in data or len(data["sequence"]) != 14:
        return jsonify({"error": "Trebuie exact 14 zile de date"}), 400

    sequence = np.array(data["sequence"])
    sequence_scaled = scaler.transform(sequence)

    future_predictions_scaled = []

    for _ in range(7):
        X_input = np.expand_dims(sequence_scaled[-14:], axis=0)
        pred = model.predict(X_input, verbose=0)[0][0]
        future_predictions_scaled.append(pred)
        new_row = np.array([pred, 0.5, 0.5])  # calorii și pași estimate
        sequence_scaled = np.vstack((sequence_scaled, new_row))

    weight_min = scaler.data_min_[0]
    weight_max = scaler.data_max_[0]
    predicted_weights = [
        w * (weight_max - weight_min) + weight_min for w in future_predictions_scaled
    ]

    return jsonify({"predicted_weights": [round(w, 2) for w in predicted_weights]})

# Pentru rulare locală
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)