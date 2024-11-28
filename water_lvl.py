import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

file_path = "goa_data.csv"

df = pd.read_csv(file_path, delimiter=',', encoding='utf-8', on_bad_lines='skip')
df.rename(columns=lambda x: x.strip(), inplace=True)
df.rename(columns={'WATER LEVEL (mbgl)': 'WaterLevel'}, inplace=True)

water_levels = df["WaterLevel"].values.reshape(-1, 1)

scaler = MinMaxScaler()
normalized_levels = scaler.fit_transform(water_levels)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_dim=1),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.fit(normalized_levels, normalized_levels, epochs=50, batch_size=8, shuffle=True, verbose=1)

predicted_levels = model.predict(normalized_levels)
reconstruction_error = np.abs(predicted_levels - normalized_levels)
threshold = 1.5 * np.mean(reconstruction_error)
anomalies = reconstruction_error > threshold

df["Anomaly"] = anomalies

plt.figure(figsize=(10, 6))
plt.plot(water_levels, label="Water Levels", color='blue', marker='o')
plt.scatter(np.where(anomalies)[0], water_levels[anomalies], label="Anomalies", color='red', zorder=5)
plt.xlabel("Wells")
plt.ylabel("Water Level (mbgl)")
plt.title("Water Level Anomalies")
plt.legend()
plt.show()

output_file = "water_level_anomalies.csv"
df.to_csv(output_file, index=False)
print(f"Anomalies saved to {output_file}")

reconstruction_error = np.abs(predicted_levels - normalized_levels)

threshold = 1.5 * np.mean(reconstruction_error)

predicted_labels = (reconstruction_error <= threshold).flatten()

true_labels = np.ones_like(predicted_labels, dtype=bool)

accuracy = np.mean(predicted_labels == true_labels) * 100
print(f"Reconstruction Accuracy: {accuracy:.2f}%")

print(f"Anomaly Threshold: {threshold}")
print(f"Mean Reconstruction Error: {np.mean(reconstruction_error)}")
print(f"Max Reconstruction Error: {np.max(reconstruction_error)}")
