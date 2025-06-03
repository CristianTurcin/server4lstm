from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "✅ App is running"

# Căi către model și scaler
MODEL_PATH = 'bilstm_model.keras'
SCALER_PATH = 'scaler.save'

print("🔄 Încarc modelul și scalerul...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model încărcat cu succes.")
except Exception as e:
    print("❌ Eroare la încărcarea modelului:", e)

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler încărcat cu succes.")
except Exception as e:
    print("❌ Eroare la încărcarea scalerului:", e)


@app.route("/predict", methods=["POST"])
def predict():
    print("📩 Request primit pe /predict")
    try:
        data = request.get_json()
        print("📦 Date primite:", data)

        if "sequence" not in data or len(data["sequence"]) != 14:
            print("⚠️ Date invalide sau lipsă.")
            return jsonify({"error": "Trebuie exact 14 zile de date"}), 400

        sequence = np.array(data["sequence"])
        print("✅ Secvență convertită în numpy:", sequence.shape)

        sequence_scaled = scaler.transform(sequence)
        print("✅ Secvență scalată.")

        future_predictions_scaled = []

        for i in range(7):
            X_input = np.expand_dims(sequence_scaled[-14:], axis=0)
            pred = model.predict(X_input, verbose=0)[0][0]
            print(f"🔮 Predicție {i+1}: {pred}")
            future_predictions_scaled.append(pred)

            new_row = np.array([pred, 0.5, 0.5])  # calorii și pași estimate
            sequence_scaled = np.vstack((sequence_scaled, new_row))

        weight_min = scaler.data_min_[0]
        weight_max = scaler.data_max_[0]
        predicted_weights = [
            w * (weight_max - weight_min) + weight_min for w in future_predictions_scaled
        ]
        print("✅ Predicții denormalizate:", predicted_weights)

        return jsonify({"predicted_weights": [round(w, 2) for w in predicted_weights]})

    except Exception as e:
        print("❌ Eroare în /predict:", e)
        return jsonify({"error": "Eroare internă", "details": str(e)}), 500


# Rulare locală (opțional)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
