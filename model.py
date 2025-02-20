#creating , training, saving the models(k-menas,autoencoder)
#source code
import pickle
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the directory to save models
save_dir = r"/kaggle/input/ccdata/CC GENERAL.csv"
os.makedirs(save_dir, exist_ok=True)

# Load dataset
csv_path = os.path.join(save_dir, "CC_GENERAL.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}")

data = pd.read_csv(csv_path)

# Drop missing values
data = data.dropna()

# Preprocess the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))

# Save the scaler for later use in API
scaler_path = os.path.join(save_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)

# Split data
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Train K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(data_scaled)

# Save K-Means model
kmeans_model_path = os.path.join(save_dir, 'kmeans_model.pkl')
with open(kmeans_model_path, 'wb') as f:
    pickle.dump(kmeans, f)

print("✅ K-Means model saved successfully!")

# Train Autoencoder
autoencoder = Sequential([
    Dense(64, activation='relu', input_shape=(data_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(data_scaled.shape[1], activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train
autoencoder.fit(X_train, X_train, epochs=5, batch_size=256, validation_data=(X_test, X_test), verbose=1)

# Save Autoencoder Model
autoencoder_model_path = os.path.join(save_dir, 'autoencoder_model.keras')
autoencoder.save(autoencoder_model_path)

print("✅ Autoencoder model saved successfully!")

# Check if models exist
print("Checking if models exist...")
print("K-Means Model Exists:", os.path.exists(kmeans_model_path))
print("Autoencoder Model Exists:", os.path.exists(autoencoder_model_path))
print("Scaler Exists:", os.path.exists(scaler_path))
