from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
from src.data_processing.preprocess import preprocess_data
from src.utils.helper import load_config, save_model
from src.pipelines.prediction_pipelines import predict_single_instance, load_artifacts
from src.pipelines.training_pipelines import build_hybrid_model
from src.logger.logs import setup_logger

# ----------------- Flask App -----------------
app = Flask(__name__, template_folder="templates", static_folder="static")
logger = setup_logger()

# ----------------- Config & Artifacts -----------------
config = load_config('src/config/config.yaml')
artifacts = None  # will hold loaded model, scaler, selector, etc.

def load_artifacts_global():
    """Load model artifacts into a global variable."""
    global artifacts
    try:
        artifacts = load_artifacts(config)
        logger.info("Model artifacts loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        artifacts = None
        return False

# ---------- Helpers ----------
REQUIRED_INPUT_KEYS = [
    'Waste Generated (Tons/Day)',
    'Population Density (People/km²)',
    'Municipal Efficiency Score (1-10)',
    'Cost of Waste Management (₹/Ton)',
    'Awareness Campaigns Count',
    'Landfill Capacity (Tons)',
    'Year',
    'City/District',
    'Waste Type',
    'Disposal Method'
]
OPTIONAL_INPUT_KEYS = ['Landfill Name', 'Landfill Location (Lat, Long)']

def coerce_single_payload_from_request(req: request) -> dict:
    """
    Accept both JSON and HTML form posts, normalize to the dict your pipeline expects.
    """
    if req.is_json:
        src = req.get_json() or {}
        mapping = {
            'waste_generated': 'Waste Generated (Tons/Day)',
            'population_density': 'Population Density (People/km²)',
            'efficiency_score': 'Municipal Efficiency Score (1-10)',
            'cost': 'Cost of Waste Management (₹/Ton)',
            'campaigns': 'Awareness Campaigns Count',
            'landfill_capacity': 'Landfill Capacity (Tons)',
            'year': 'Year',
            'city': 'City/District',
            'waste_type': 'Waste Type',
            'disposal_method': 'Disposal Method',
            'landfill_name': 'Landfill Name',
            'landfill_location': 'Landfill Location (Lat, Long)'
        }
        out = {}
        for k in REQUIRED_INPUT_KEYS + OPTIONAL_INPUT_KEYS:
            if k in src:
                out[k] = src[k]
        for short, longk in mapping.items():
            if longk not in out and short in src:
                out[longk] = src[short]
    else:
        f = req.form
        out = {
            'Waste Generated (Tons/Day)': float(f.get('waste_generated', 0)),
            'Population Density (People/km²)': float(f.get('population_density', 0)),
            'Municipal Efficiency Score (1-10)': float(f.get('efficiency_score', 0)),
            'Cost of Waste Management (₹/Ton)': float(f.get('cost', 0)),
            'Awareness Campaigns Count': float(f.get('campaigns', 0)),
            'Landfill Capacity (Tons)': float(f.get('landfill_capacity', 0)),
            'Year': float(f.get('year', 0)),
            'City/District': f.get('city', ''),
            'Waste Type': f.get('waste_type', ''),
            'Disposal Method': f.get('disposal_method', '')
        }
        if f.get('landfill_name'):
            out['Landfill Name'] = f.get('landfill_name')
        if f.get('landfill_location'):
            out['Landfill Location (Lat, Long)'] = f.get('landfill_location')

    numeric_keys = [
        'Waste Generated (Tons/Day)',
        'Population Density (People/km²)',
        'Municipal Efficiency Score (1-10)',
        'Cost of Waste Management (₹/Ton)',
        'Awareness Campaigns Count',
        'Landfill Capacity (Tons)',
        'Year'
    ]
    for k in numeric_keys:
        if k in out:
            try:
                out[k] = float(out[k])
            except Exception:
                out[k] = np.nan

    return out

def predict_batch(df: pd.DataFrame, cfg, arts) -> pd.DataFrame:
    """
    Perform batch predictions by aligning DataFrame columns with the single-instance API.
    """
    if arts is None:
        raise RuntimeError("Artifacts not loaded. Train the model first.")

    col_map = {
        'waste_generated': 'Waste Generated (Tons/Day)',
        'population_density': 'Population Density (People/km²)',
        'efficiency_score': 'Municipal Efficiency Score (1-10)',
        'cost': 'Cost of Waste Management (₹/Ton)',
        'campaigns': 'Awareness Campaigns Count',
        'landfill_capacity': 'Landfill Capacity (Tons)',
        'year': 'Year',
        'city': 'City/District',
        'waste_type': 'Waste Type',
        'disposal_method': 'Disposal Method',
        'landfill_name': 'Landfill Name',
        'landfill_location': 'Landfill Location (Lat, Long)',
    }

    work = df.copy()
    for short, longk in col_map.items():
        if short in work.columns and longk not in work.columns:
            work[longk] = work[short]

    missing = [k for k in REQUIRED_INPUT_KEYS if k not in work.columns]
    if missing:
        raise ValueError(f"Missing columns for batch prediction: {missing}")

    preds = []
    for _, row in work.iterrows():
        payload = {k: row[k] for k in REQUIRED_INPUT_KEYS if k in work.columns}
        for ok in OPTIONAL_INPUT_KEYS:
            if ok in work.columns and pd.notna(row[ok]):
                payload[ok] = row[ok]
        for nk in [
            'Waste Generated (Tons/Day)',
            'Population Density (People/km²)',
            'Municipal Efficiency Score (1-10)',
            'Cost of Waste Management (₹/Ton)',
            'Awareness Campaigns Count',
            'Landfill Capacity (Tons)',
            'Year'
        ]:
            if nk in payload:
                try:
                    payload[nk] = float(payload[nk])
                except Exception:
                    payload[nk] = np.nan
        p = predict_single_instance(payload, cfg, arts)
        preds.append(p)

    out = df.copy()
    out['Predicted Recycling Rate (%)'] = preds
    return out

def save_predictions(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

# ----------------- Routes -----------------
@app.route('/')
def root():
    logger.info("GET / → redirect to /home")
    return render_template('home.html')

@app.route('/home')
def home():
    logger.info("GET /home → home.html")
    return render_template('home.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        logger.info("GET /train → train.html")
        return render_template('train.html')
    elif request.method == 'POST':
        logger.info("POST /train")
        try:
            # Initialize log for training steps
            training_log = []

            # Log: Start training
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Initializing training process...")
            logger.info("Initializing training process...")

            # Preprocess train/test
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Loading and preprocessing data...")
            logger.info("Loading and preprocessing data...")
            X_train, X_test, y_train, y_test, scaler_new, selector_new, selected_features = preprocess_data(config)

            # Log: Data loaded
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")
            logger.info(f"Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")

            # Full dataset for hybrid model
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Preparing full dataset for hybrid model...")
            logger.info("Preparing full dataset for hybrid model...")
            df_aug = pd.read_csv(config['paths']['processed_data'])
            feature_cols = [c for c in df_aug.columns if c not in [
                'City/District', 'Waste Type', 'Disposal Method',
                'Landfill Name', 'Landfill Location (Lat, Long)', 'Recycling Rate (%)'
            ]]
            X_full = df_aug[feature_cols].fillna(df_aug[feature_cols].mean())
            y_full = df_aug['Recycling Rate (%)']

            # Log: Training model
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Training hybrid model...")
            logger.info("Training hybrid model...")
            best_pred, best_r2, best_model_name, best_model = build_hybrid_model(
                X_train, y_train, X_test, y_test, X_full, y_full, scaler_new, config
            )

            # Log: Model trained
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Model trained: {best_model_name}, R²: {best_r2:.4f}")
            logger.info(f"Model trained: {best_model_name}, R²: {best_r2:.4f}")

            # Save artifacts
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Saving model artifacts...")
            logger.info("Saving model artifacts...")
            save_model(scaler_new, config['paths']['scaler'])
            save_model(selector_new, config['paths']['selector'])
            save_model(best_model, config['paths']['model'])
            save_model(best_model_name, config['paths']['model_name'])

            # Reload artifacts
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Reloading artifacts...")
            logger.info("Reloading artifacts...")
            load_artifacts_global()

            # Log: Training completed
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Training completed successfully")
            logger.info("Training completed successfully")
            msg = f"Training completed. Best model: {best_model_name}, R²: {best_r2:.4f}"
            logger.info(msg)

            return jsonify({'status': 'success', 'message': msg, 'logs': training_log})
        except Exception as e:
            logger.exception("Training failed")
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Error: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e), 'logs': training_log}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    if request.method == 'GET':
        logger.info("GET /predict → predict.html")
        return render_template('predict.html')
    elif request.method == 'POST':
        logger.info("POST /predict")
        try:
            if artifacts is None and not load_artifacts_global():
                return jsonify({'status': 'error', 'message': 'Model not trained yet.'}), 400

            data = coerce_single_payload_from_request(request)
            logger.info(f"Predict payload: {data}")

            pred = predict_single_instance(data, config, artifacts)
            logger.info(f"Prediction: {pred:.2f}%")

            return jsonify({'status': 'success', 'recycling_rate': round(float(pred), 4)})
        except Exception as e:
            logger.exception("Prediction failed")
            return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict_route():
    logger.info("POST /batch_predict")
    try:
        if artifacts is None and not load_artifacts_global():
            return jsonify({'status': 'error', 'message': 'Model not trained yet.'}), 400

        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided.'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected.'}), 400

        os.makedirs('temp', exist_ok=True)
        os.makedirs('output', exist_ok=True)

        filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', filename)
        file.save(temp_path)

        df = pd.read_csv(temp_path)
        logger.info(f"Uploaded CSV shape: {df.shape}")

        result_df = predict_batch(df, config, artifacts)
        out_name = f'predictions_{filename}'
        output_path = os.path.join('output', out_name)
        save_predictions(result_df, output_path)

        try:
            os.remove(temp_path)
        except Exception:
            pass

        return jsonify({
            'status': 'success',
            'message': f'Batch prediction completed. {len(result_df)} rows processed.',
            'download_link': f'/download/{out_name}'
        })
    except Exception as e:
        logger.exception("Batch prediction failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Serve files from /output for download."""
    return send_from_directory('output', filename, as_attachment=True)

@app.route('/logs', methods=['GET'])
def get_logs():
    """Fetch all logs from logs/pipeline.log."""
    try:
        log_file_path = os.path.join('logs', 'pipeline.log')
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                logs = f.readlines()
            return jsonify({'status': 'success', 'logs': [log.strip() for log in logs]})
        else:
            logger.warning("Log file logs/pipeline.log not found")
            return jsonify({'status': 'error', 'message': 'Log file not found', 'logs': []}), 404
    except Exception as e:
        logger.error(f"Failed to read logs: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e), 'logs': []}), 500

# ----------------- Entrypoint -----------------
if __name__ == '__main__':
    load_artifacts_global()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
