from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, f1_score, recall_score

app = Flask(__name__, static_folder='static')
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global state (for demo; use a better state mgmt in prod)
model = None
scaler_X = None
scaler_y = None
label_encoders = {}
seq_length = 10
features = ["Machine_Type", "Operation_Type", "Cycle_Time"]
df = None
X_scaled = None
y_scaled = None
X = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None
y_pred_inv = None
y_test_inv = None
future_preds_inv = None
future_class = None
optimization_suggestions = []
avg_energy_machine = None
avg_energy_operation = None
anomaly_labels = None
iso_forest = None
anomaly_metrics = {}

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/upload-and-train', methods=['POST'])
def upload_and_train():
    global model, scaler_X, scaler_y, label_encoders, df, X_scaled, y_scaled, X, y, X_train, X_test, y_train, y_test, y_pred_inv, y_test_inv, future_preds_inv, future_class, optimization_suggestions, avg_energy_machine, avg_energy_operation, iso_forest, anomaly_labels, anomaly_metrics

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    df = pd.read_csv(filepath)

    # Encode categorical columns
    label_encoders = {}
    for col in ["Machine_Type", "Operation_Type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Select features and target
    X_data = df[features].values
    y_data = df["Energy_Consumed"].values.reshape(-1, 1)

    # Scale features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data)

    # Create sequences
    def create_sequences(X, y, seq_length):
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:i+seq_length])
            ys.append(y[i+seq_length])
        return np.array(Xs), np.array(ys)

    X, y = create_sequences(X_scaled, y_scaled, seq_length)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(seq_length, len(features))))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, callbacks=[early_stop], verbose=0)

    # Predictions
    y_pred = model.predict(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    # Future forecast
    n_future = 5
    last_seq = X_scaled[-seq_length:]
    future_preds = []
    current_seq = last_seq.reshape(1, seq_length, len(features))
    for _ in range(n_future):
        next_pred = model.predict(current_seq)[0]
        future_preds.append(next_pred)
        new_step = current_seq[:, -1, :].copy()
        new_step = np.expand_dims(new_step, axis=1)
        current_seq = np.append(current_seq[:, 1:, :], new_step, axis=1)
    future_preds_inv = scaler_y.inverse_transform(np.array(future_preds).reshape(-1,1))

    # Classification threshold
    threshold = np.median(y_test_inv)
    y_test_class = (y_test_inv.flatten() > threshold).astype(int)
    y_pred_class = (y_pred_inv.flatten() > threshold).astype(int)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class)

    # Future class
    future_class = (future_preds_inv.flatten() > threshold).astype(int)

    # Optimization suggestions
    optimization_suggestions = []
    for i, energy in enumerate(future_preds_inv.flatten()):
        if energy > threshold:
            suggestion = f"step {i+1}: High energy predicted ({energy:.2f}). Suggest reducing cycle time or rescheduling operation."
        else:
            suggestion = f"step {i+1}: Energy normal ({energy:.2f}). No action needed."
        optimization_suggestions.append(suggestion)

    # Average energy per Machine/Operation
    avg_energy_machine = df.groupby('Machine_Type')['Energy_Consumed'].mean()
    avg_energy_operation = df.groupby('Operation_Type')['Energy_Consumed'].mean()

    # Anomaly detection
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    X_features = df[features].values
    iso_forest.fit(X_features)
    anomaly_labels = iso_forest.predict(X_features)
    df["Anomaly"] = np.where(anomaly_labels == -1, 1, 0)
    anomaly_threshold = df["Energy_Consumed"].quantile(0.95)
    y_true = (df["Energy_Consumed"] > anomaly_threshold).astype(int)
    y_pred_anom = df["Anomaly"]
    anomaly_metrics = {
        "precision": float(precision_score(y_true, y_pred_anom)),
        "recall": float(recall_score(y_true, y_pred_anom)),
        "f1_score": float(f1_score(y_true, y_pred_anom))
    }

    return jsonify({"message": "Model trained successfully!"})

@app.route('/data/performance_metrics')
def performance_metrics():
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    return jsonify({
        "regression": {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2_score": float(r2)
        },
        "classification": {
            "accuracy": float(accuracy_score((y_test_inv.flatten() > np.median(y_test_inv)).astype(int), (y_pred_inv.flatten() > np.median(y_test_inv)).astype(int))),
            "precision": float(precision_score((y_test_inv.flatten() > np.median(y_test_inv)).astype(int), (y_pred_inv.flatten() > np.median(y_test_inv)).astype(int))),
            "f1_score": float(f1_score((y_test_inv.flatten() > np.median(y_test_inv)).astype(int), (y_pred_inv.flatten() > np.median(y_test_inv)).astype(int)))
        }
    })

@app.route('/data/prediction_graph')
def prediction_graph():
    labels = list(range(len(y_test_inv)))
    return jsonify({
        "labels": labels,
        "actual": y_test_inv.flatten().tolist(),
        "predicted": y_pred_inv.flatten().tolist()
    })

@app.route('/data/future_forecast')
def future_forecast():
    labels = [f"Step {i+1}" for i in range(len(future_preds_inv))]
    return jsonify({
        "labels": labels,
        "consumption": future_preds_inv.flatten().tolist(),
        "classification": future_class.tolist()
    })

@app.route('/data/optimization')
def optimization():
    return jsonify({
        "suggestions": optimization_suggestions,
        "avg_by_machine": {
            "labels": [str(x) for x in avg_energy_machine.index],
            "data": avg_energy_machine.values.tolist()
        },
        "avg_by_operation": {
            "labels": [str(x) for x in avg_energy_operation.index],
            "data": avg_energy_operation.values.tolist()
        }
    })

@app.route('/data/anomaly_detection')
def anomaly_detection():
    chart_data = {
        "labels": list(range(len(df))),
        "consumption": df["Energy_Consumed"].tolist(),
        "anomaly_indices": df.index[df["Anomaly"]==1].tolist()
    }
    return jsonify({
        "metrics": anomaly_metrics,
        "chart_data": chart_data
    })

@app.route('/data/dashboard')
def dashboard():
    # Combine metrics for dashboard
    return jsonify({
        "regression": {
            "mae": float(mean_absolute_error(y_test_inv, y_pred_inv)),
            "r2_score": float(r2_score(y_test_inv, y_pred_inv))
        },
        "classification": {
            "accuracy": float(accuracy_score((y_test_inv.flatten() > np.median(y_test_inv)).astype(int), (y_pred_inv.flatten() > np.median(y_test_inv)).astype(int)))
        },
        "anomaly_metrics": anomaly_metrics
    })

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
