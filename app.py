import os
import uuid
import json
import sqlite3
import logging
import threading

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_FOLDER = BASE_DIR / "models"
DATABASE_PATH = BASE_DIR / "jobs.db"

UPLOAD_FOLDER.mkdir(exist_ok=True)
MODEL_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}

MAX_ROWS = 100000
MIN_ROWS = 20

SEQ_LENGTH = 10
FEATURES = ["Machine_Type", "Operation_Type", "Cycle_Time"]

TARGET = "Energy_Consumed"

N_FUTURE = 5

app = Flask(__name__, static_folder="static")

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# CORS is disabled by default.
# Set CORS_ORIGINS to a comma-separated list if frontend
# and backend are hosted on different origins.

cors_origins = os.getenv("CORS_ORIGINS", "").strip()

if cors_origins:
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": [
                    origin.strip()
                    for origin in cors_origins.split(",")
                    if origin.strip()
                ]
            }
        }
    )

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Background executor.
# This is suitable for a single-process deployment.
# For multi-worker production deployments, use Celery + Redis.

executor = ThreadPoolExecutor(max_workers=2)

# Protects concurrent access to SQLite initialization and job updates.
db_lock = threading.Lock()


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    with db_lock:

        connection = get_db_connection()

        connection.execute("""
            CREATE TABLE IF NOT EXISTS jobs (

                job_id TEXT PRIMARY KEY,

                status TEXT NOT NULL,

                filename TEXT,

                created_at TEXT,

                updated_at TEXT,

                error TEXT,

                results TEXT

            )
        """)

        connection.commit()
        connection.close()


def update_job(job_id, **fields):

    if not fields:
        return

    fields["updated_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    columns = ", ".join(
        f"{key} = ?" for key in fields
    )

    values = list(fields.values())
    values.append(job_id)

    with db_lock:

        connection = get_db_connection()

        connection.execute(
            f"UPDATE jobs SET {columns} WHERE job_id = ?",
            values
        )

        connection.commit()
        connection.close()


def get_job(job_id):

    connection = get_db_connection()

    job = connection.execute(
        "SELECT * FROM jobs WHERE job_id = ?",
        (job_id,)
    ).fetchone()

    connection.close()

    return job


def create_job(job_id, filename):

    now = datetime.now(
        timezone.utc
    ).isoformat()

    with db_lock:

        connection = get_db_connection()

        connection.execute("""
            INSERT INTO jobs (
                job_id,
                status,
                filename,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            job_id,
            "queued",
            filename,
            now,
            now
        ))

        connection.commit()
        connection.close()


initialize_database()


# --------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def job_model_directory(job_id):

    directory = MODEL_FOLDER / job_id

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    return directory


def get_job_results(job_id):

    job = get_job(job_id)

    if job is None:

        return None, (
            jsonify({"error": "Job not found"}),
            404
        )

    if job["status"] != "completed":

        return None, (
            jsonify({
                "error": "Training is not completed",
                "status": job["status"]
            }),
            409
        )

    if not job["results"]:

        return None, (
            jsonify({"error": "Results are unavailable"}),
            500
        )

    try:

        results = json.loads(job["results"])

    except json.JSONDecodeError:

        logger.exception(
            "Invalid results JSON for job %s",
            job_id
        )

        return None, (
            jsonify({"error": "Stored results are corrupted"}),
            500
        )

    return results, None


def safe_float(value):

    value = float(value)

    if not np.isfinite(value):
        return 0.0

    return value


# --------------------------------------------------
# CSV VALIDATION
# --------------------------------------------------

def validate_dataset(filepath):

    try:

        data = pd.read_csv(filepath)

    except pd.errors.EmptyDataError:

        raise ValueError("The uploaded CSV is empty.")

    except pd.errors.ParserError:

        raise ValueError("The uploaded file is not a valid CSV.")

    except UnicodeDecodeError:

        raise ValueError("The CSV encoding is not supported.")

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(data) < MIN_ROWS:

        raise ValueError(
            f"Dataset must contain at least {MIN_ROWS} rows."
        )

    if len(data) > MAX_ROWS:

        raise ValueError(
            f"Dataset cannot contain more than {MAX_ROWS} rows."
        )

    data = data[required_columns].copy()

    # Clean categorical values.

    for column in ["Machine_Type", "Operation_Type"]:

        data[column] = (
            data[column]
            .astype("string")
            .str.strip()
        )

        if data[column].isna().any():

            raise ValueError(
                f"{column} contains missing values."
            )

        if (data[column] == "").any():

            raise ValueError(
                f"{column} contains empty values."
            )

    # Validate numerical columns.

    for column in ["Cycle_Time", TARGET]:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    if data.isnull().any().any():

        raise ValueError(
            "Dataset contains missing or invalid values."
        )

    numeric_values = data[
        ["Cycle_Time", TARGET]
    ].to_numpy()

    if not np.isfinite(numeric_values).all():

        raise ValueError(
            "Dataset contains infinite values."
        )

    if (data["Cycle_Time"] < 0).any():

        raise ValueError(
            "Cycle_Time cannot contain negative values."
        )

    if (data[TARGET] < 0).any():

        raise ValueError(
            "Energy_Consumed cannot contain negative values."
        )

    return data


# --------------------------------------------------
# SEQUENCE CREATION
# --------------------------------------------------

def create_sequences(X, y, sequence_length):

    X_sequences = []
    y_sequences = []

    for index in range(
        len(X) - sequence_length
    ):

        X_sequences.append(
            X[index:index + sequence_length]
        )

        y_sequences.append(
            y[index + sequence_length]
        )

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

def train_model(job_id, filepath):

    try:

        update_job(
            job_id,
            status="running",
            error=None
        )

        logger.info(
            "Starting training for job %s",
            job_id
        )

        # ------------------------------------------
        # Load and validate dataset
        # ------------------------------------------

        data = validate_dataset(filepath)

        # Preserve original category labels for charts.

        original_data = data.copy()

        # ------------------------------------------
        # Chronological train/test split
        # ------------------------------------------

        split_index = int(len(data) * 0.8)

        train_data = data.iloc[:split_index].copy()

        test_data = data.iloc[split_index:].copy()

        if len(train_data) <= SEQ_LENGTH:

            raise ValueError(
                "Not enough training rows to create sequences."
            )

        if len(test_data) == 0:

            raise ValueError(
                "Not enough test data."
            )

        # ------------------------------------------
        # Encode categorical columns
        # ------------------------------------------

        label_encoders = {}

        encoded_data = data.copy()

        for column in [
            "Machine_Type",
            "Operation_Type"
        ]:

            encoder = LabelEncoder()

            encoder.fit(
                train_data[column]
            )

            unknown_categories = (
                set(test_data[column])
                - set(encoder.classes_)
            )

            if unknown_categories:

                raise ValueError(
                    f"Test data contains unseen categories "
                    f"in {column}: {unknown_categories}"
                )

            encoded_data[column] = (
                encoder.transform(data[column])
            )

            label_encoders[column] = encoder

        # ------------------------------------------
        # Prepare features and target
        # ------------------------------------------

        X_raw = encoded_data[
            FEATURES
        ].to_numpy(dtype=float)

        y_raw = encoded_data[
            TARGET
        ].to_numpy(dtype=float).reshape(-1, 1)

        # ------------------------------------------
        # Fit scalers using training data only
        # ------------------------------------------

        scaler_X = MinMaxScaler()

        scaler_y = MinMaxScaler()

        scaler_X.fit(
            X_raw[:split_index]
        )

        scaler_y.fit(
            y_raw[:split_index]
        )

        X_scaled = scaler_X.transform(X_raw)

        y_scaled = scaler_y.transform(y_raw)

        # ------------------------------------------
        # Create sequences
        # ------------------------------------------

        X, y = create_sequences(
            X_scaled,
            y_scaled,
            SEQ_LENGTH
        )

        # Each target corresponds to the original
        # row at index SEQ_LENGTH + sequence index.

        target_indices = np.arange(
            SEQ_LENGTH,
            len(data)
        )

        train_mask = (
            target_indices < split_index
        )

        test_mask = (
            target_indices >= split_index
        )

        X_train = X[train_mask]

        y_train = y[train_mask]

        X_test = X[test_mask]

        y_test = y[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:

            raise ValueError(
                "Unable to create valid training and test sequences."
            )

        # ------------------------------------------
        # Build LSTM model
        # ------------------------------------------

        model = Sequential()

        model.add(
            LSTM(
                64,
                activation="relu",
                input_shape=(
                    SEQ_LENGTH,
                    len(FEATURES)
                )
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            Dense(1)
        )

        model.compile(
            optimizer="adam",
            loss="mse"
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(
                X_test,
                y_test
            ),
            epochs=30,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        # ------------------------------------------
        # Predictions
        # ------------------------------------------

        y_pred = model.predict(
            X_test,
            verbose=0
        )

        y_test_inv = scaler_y.inverse_transform(
            y_test
        )

        y_pred_inv = scaler_y.inverse_transform(
            y_pred
        )

        # ------------------------------------------
        # Regression metrics
        # ------------------------------------------

        mse = mean_squared_error(
            y_test_inv,
            y_pred_inv
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            y_test_inv,
            y_pred_inv
        )

        r2 = r2_score(
            y_test_inv,
            y_pred_inv
        )

        # ------------------------------------------
        # Classification metrics
        # ------------------------------------------

        threshold = float(
            np.median(y_test_inv)
        )

        y_test_class = (
            y_test_inv.flatten() > threshold
        ).astype(int)

        y_pred_class = (
            y_pred_inv.flatten() > threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_test_class,
            y_pred_class
        )

        precision = precision_score(
            y_test_class,
            y_pred_class,
            zero_division=0
        )

        recall = recall_score(
            y_test_class,
            y_pred_class,
            zero_division=0
        )

        f1 = f1_score(
            y_test_class,
            y_pred_class,
            zero_division=0
        )

        # ------------------------------------------
        # Future forecast
        # ------------------------------------------

        last_sequence = X_scaled[-SEQ_LENGTH:]

        current_sequence = last_sequence.reshape(
            1,
            SEQ_LENGTH,
            len(FEATURES)
        )

        future_predictions = []

        for _ in range(N_FUTURE):

            next_prediction = model.predict(
                current_sequence,
                verbose=0
            )[0]

            future_predictions.append(
                next_prediction
            )

            # Keep the latest feature vector.
            # This is a baseline forecast assumption:
            # future input features remain unchanged.

            new_step = current_sequence[:, -1:, :].copy()

            current_sequence = np.concatenate(
                [
                    current_sequence[:, 1:, :],
                    new_step
                ],
                axis=1
            )

        future_predictions_inv = (
            scaler_y.inverse_transform(
                np.array(
                    future_predictions
                ).reshape(-1, 1)
            )
        )

        future_class = (
            future_predictions_inv.flatten() > threshold
        ).astype(int)

        # ------------------------------------------
        # Optimization suggestions
        # ------------------------------------------

        optimization_suggestions = []

        for index, energy in enumerate(
            future_predictions_inv.flatten()
        ):

            if energy > threshold:

                suggestion = (
                    f"Step {index + 1}: High energy predicted "
                    f"({energy:.2f}). Suggest reducing cycle "
                    f"time or rescheduling operation."
                )

            else:

                suggestion = (
                    f"Step {index + 1}: Energy normal "
                    f"({energy:.2f}). No action needed."
                )

            optimization_suggestions.append(
                suggestion
            )

        # ------------------------------------------
        # Average energy by machine and operation
        # ------------------------------------------

        avg_energy_machine = (
            original_data
            .groupby("Machine_Type")[TARGET]
            .mean()
        )

        avg_energy_operation = (
            original_data
            .groupby("Operation_Type")[TARGET]
            .mean()
        )

        # ------------------------------------------
        # Anomaly detection
        # ------------------------------------------

        # Use encoded feature values for Isolation Forest.

        X_features = encoded_data[
            FEATURES
        ].to_numpy(dtype=float)

        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )

        anomaly_labels = iso_forest.fit_predict(
            X_features
        )

        anomaly_flags = (
            anomaly_labels == -1
        ).astype(int)

        anomaly_threshold = (
            original_data[TARGET].quantile(0.95)
        )

        y_true_anomaly = (
            original_data[TARGET] > anomaly_threshold
        ).astype(int)

        anomaly_precision = precision_score(
            y_true_anomaly,
            anomaly_flags,
            zero_division=0
        )

        anomaly_recall = recall_score(
            y_true_anomaly,
            anomaly_flags,
            zero_division=0
        )

        anomaly_f1 = f1_score(
            y_true_anomaly,
            anomaly_flags,
            zero_division=0
        )

        anomaly_metrics = {
            "precision": safe_float(anomaly_precision),
            "recall": safe_float(anomaly_recall),
            "f1_score": safe_float(anomaly_f1)
        }

        # ------------------------------------------
        # Save model and preprocessing artifacts
        # ------------------------------------------

        model_directory = job_model_directory(
            job_id
        )

        model.save(
            model_directory / "energy_model.keras"
        )

        joblib.dump(
            {
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "label_encoders": label_encoders,
                "features": FEATURES,
                "target": TARGET,
                "sequence_length": SEQ_LENGTH,
                "threshold": threshold
            },
            model_directory / "preprocessing.joblib"
        )

        joblib.dump(
            iso_forest,
            model_directory / "isolation_forest.joblib"
        )

        # ------------------------------------------
        # Prepare JSON results
        # ------------------------------------------

        results = {

            "regression": {
                "mae": safe_float(mae),
                "mse": safe_float(mse),
                "rmse": safe_float(rmse),
                "r2_score": safe_float(r2)
            },

            "classification": {
                "accuracy": safe_float(accuracy),
                "precision": safe_float(precision),
                "recall": safe_float(recall),
                "f1_score": safe_float(f1)
            },

            "predictions": {
                "labels": list(
                    range(len(y_test_inv))
                ),
                "actual": [
                    safe_float(value)
                    for value in y_test_inv.flatten()
                ],
                "predicted": [
                    safe_float(value)
                    for value in y_pred_inv.flatten()
                ]
            },

            "future_forecast": {
                "labels": [
                    f"Step {i + 1}"
                    for i in range(N_FUTURE)
                ],
                "consumption": [
                    safe_float(value)
                    for value in future_predictions_inv.flatten()
                ],
                "classification": future_class.tolist()
            },

            "optimization": {
                "suggestions": optimization_suggestions,

                "avg_by_machine": {
                    "labels": [
                        str(value)
                        for value in avg_energy_machine.index
                    ],
                    "data": [
                        safe_float(value)
                        for value in avg_energy_machine.values
                    ]
                },

                "avg_by_operation": {
                    "labels": [
                        str(value)
                        for value in avg_energy_operation.index
                    ],
                    "data": [
                        safe_float(value)
                        for value in avg_energy_operation.values
                    ]
                }
            },

            "anomaly_metrics": anomaly_metrics,

            "anomaly_chart": {
                "labels": list(
                    range(len(original_data))
                ),

                "consumption": [
                    safe_float(value)
                    for value in original_data[TARGET].tolist()
                ],

                "anomaly_indices": np.where(
                    anomaly_flags == 1
                )[0].tolist()
            }

        }

        # ------------------------------------------
        # Persist results in SQLite
        # ------------------------------------------

        update_job(
            job_id,
            status="completed",
            results=json.dumps(results),
            error=None
        )

        logger.info(
            "Training completed for job %s",
            job_id
        )

    except Exception as error:

        logger.exception(
            "Training failed for job %s",
            job_id
        )

        update_job(
            job_id,
            status="failed",
            error=str(error)
        )

    finally:

        # Remove uploaded CSV after training.
        try:

            if os.path.exists(filepath):
                os.remove(filepath)

        except OSError:

            logger.warning(
                "Could not delete uploaded file: %s",
                filepath
            )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "application": "Energy Optimization Platform"
    })


# --------------------------------------------------
# UPLOAD AND TRAIN
# --------------------------------------------------

@app.route("/upload-and-train", methods=["POST"])
def upload_and_train():

    file = request.files.get("file")

    if file is None:

        return jsonify({
            "error": "No file uploaded."
        }), 400

    if not file.filename:

        return jsonify({
            "error": "Filename is empty."
        }), 400

    if not allowed_file(file.filename):

        return jsonify({
            "error": "Only CSV files are allowed."
        }), 400

    safe_name = secure_filename(
        file.filename
    )

    if not safe_name:

        return jsonify({
            "error": "Invalid filename."
        }), 400

    job_id = uuid.uuid4().hex

    unique_name = (
        f"{job_id}_{safe_name}"
    )

    filepath = UPLOAD_FOLDER / unique_name

    try:

        file.save(filepath)

        # Validate before starting background work.

        validate_dataset(filepath)

        create_job(
            job_id,
            unique_name
        )

        executor.submit(
            train_model,
            job_id,
            str(filepath)
        )

        return jsonify({
            "message": "Training job submitted successfully.",
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/jobs/{job_id}"
        }), 202

    except ValueError as error:

        if filepath.exists():
            filepath.unlink()

        return jsonify({
            "error": str(error)
        }), 400

    except Exception:

        logger.exception(
            "Could not submit training job."
        )

        if filepath.exists():
            filepath.unlink()

        return jsonify({
            "error": "Unable to submit training job."
        }), 500


# --------------------------------------------------
# JOB STATUS
# --------------------------------------------------

@app.route("/jobs/<job_id>", methods=["GET"])
def job_status(job_id):

    job = get_job(job_id)

    if job is None:

        return jsonify({
            "error": "Job not found."
        }), 404

    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "filename": job["filename"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"]
    }

    if job["status"] == "failed":

        response["error"] = job["error"]

    return jsonify(response)


# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

@app.route("/data/performance_metrics", methods=["GET"])
def performance_metrics():

    job_id = request.args.get("job_id")

    if not job_id:

        return jsonify({
            "error": "job_id is required."
        }), 400

    results, error = get_job_results(job_id)

    if error:
        return error

    return jsonify({
        "regression": results["regression"],
        "classification": results["classification"]
    })


# --------------------------------------------------
# PREDICTION GRAPH
# --------------------------------------------------

@app.route("/data/prediction_graph", methods=["GET"])
def prediction_graph():

    job_id = request.args.get("job_id")

    if not job_id:

        return jsonify({
            "error": "job_id is required."
        }), 400

    results, error = get_job_results(job_id)

    if error:
        return error

    return jsonify(
        results["predictions"]
    )


# --------------------------------------------------
# FUTURE FORECAST
# --------------------------------------------------

@app.route("/data/future_forecast", methods=["GET"])
def future_forecast():

    job_id = request.args.get("job_id")

    if not job_id:

        return jsonify({
            "error": "job_id is required."
        }), 400

    results, error = get_job_results(job_id)

    if error:
        return error

    return jsonify(
        results["future_forecast"]
    )


# --------------------------------------------------
# OPTIMIZATION
# --------------------------------------------------

@app.route("/data/optimization", methods=["GET"])
def optimization():

    job_id = request.args.get("job_id")

    if not job_id:

        return jsonify({
            "error": "job_id is required."
        }), 400

    results, error = get_job_results(job_id)

    if error:
        return error

    return jsonify(
        results["optimization"]
    )


# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------

@app.route("/data/anomaly_detection", methods=["GET"])
def anomaly_detection():

    job_id = request.args.get("job_id")

    if not job_id:

        return jsonify({
            "error": "job_id is required."
        }), 400

    results, error = get_job_results(job_id)

    if error:
        return error

    return jsonify({
        "metrics": results["anomaly_metrics"],
        "chart_data": results["anomaly_chart"]
    })


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/data/dashboard", methods=["GET"])
def dashboard():

    job_id = request.args.get("job_id")

    if not job_id:

        return jsonify({
            "error": "job_id is required."
        }), 400

    results, error = get_job_results(job_id)

    if error:
        return error

    return jsonify({
        "regression": {
            "mae": results["regression"]["mae"],
            "r2_score": results["regression"]["r2_score"]
        },

        "classification": {
            "accuracy": results["classification"]["accuracy"]
        },

        "anomaly_metrics": results["anomaly_metrics"]
    })


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

@app.route("/")
def serve_index():

    static_index = BASE_DIR / "static" / "index.html"

    root_index = BASE_DIR / "index.html"

    if static_index.exists():

        return send_from_directory(
            static_index.parent,
            static_index.name
        )

    if root_index.exists():

        return send_from_directory(
            root_index.parent,
            root_index.name
        )

    return jsonify({
        "error": "Frontend index.html was not found."
    }), 404


@app.route("/static/<path:filename>")
def serve_static(filename):

    return send_from_directory(
        app.static_folder,
        filename
    )


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):

    return jsonify({
        "error": "File size exceeds the 10 MB upload limit."
    }), 413


@app.errorhandler(404)
def handle_not_found(error):

    return jsonify({
        "error": "Requested resource was not found."
    }), 404


@app.errorhandler(500)
def handle_internal_error(error):

    logger.exception(
        "Internal server error."
    )

    return jsonify({
        "error": "An internal server error occurred."
    }), 500


# --------------------------------------------------
# APPLICATION START
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )
```
