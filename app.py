from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Define model directory
save_dir = r"C:\Users\gagan\OneDrive\Desktop\IBM_PROJECT"

# Load models
kmeans_model_path = os.path.join(save_dir, "kmeans_model.pkl")
scaler_path = os.path.join(save_dir, "scaler.pkl")

# Ensure models exist before loading
if not os.path.exists(kmeans_model_path):
    raise FileNotFoundError(f"K-Means model not found at {kmeans_model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler not found at {scaler_path}")

# Load K-Means model
with open(kmeans_model_path, "rb") as f:
    kmeans = pickle.load(f)

# Load Scaler
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Ensure the scaler is valid
if not isinstance(scaler, StandardScaler):
    raise TypeError("Loaded scaler is not a valid StandardScaler object. Check the saved model.")

# Cluster descriptions
cluster_descriptions = {
    0: "High spenders who use credit mainly for purchases.",
    1: "Low spenders with minimal credit card usage.",
    2: "Customers who rely heavily on cash advances instead of purchases."
}

# Define API route for K-Means prediction
@app.route("/predict/kmeans", methods=["POST"])
def predict_kmeans():
    try:
        data = request.json.get("data", None)

        if not data:
            return jsonify({"error": "No data received"}), 400

        print("Received Data:", data)  # Debugging log

        data = np.array(data).reshape(1, -1)
        print("Data Shape:", data.shape)  # Debugging log

        data_scaled = scaler.transform(data)  # Apply StandardScaler

        cluster = kmeans.predict(data_scaled)[0]
        cluster_description = cluster_descriptions.get(cluster, "Unknown Cluster")

        # Return response with cluster number and description
        response = {
            "cluster": int(cluster),
            "description": cluster_description
        }

        return jsonify(response)
    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging log
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
