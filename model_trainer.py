# model_trainer.py
# The Analytical Engine - This file runs as a separate process,
# dedicated to collecting tick data and training sophisticated models
# to provide real-time intelligence to the main trading bot.
# Built with precision to operate in parallel and not block the main process.

import asyncio
import websockets
import json
import numpy as np
import pandas as pd
import time
import os
import warnings
import joblib
import json
import sys
import re
from collections import deque, Counter, defaultdict
from scipy import stats
from typing import Dict, Deque, Tuple, Any, Optional
from multiprocessing import Queue
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import logging

# --- Self-Sufficiency and Dependency Management ---
# This ensures a flawless setup in any environment by installing dependencies
# before anything else. This is our commitment to quality.
def install_and_import(package, install_name=None):
    """Installs and imports a package, handling naming differences."""
    if install_name is None:
        install_name = package
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {install_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
        print(f"Installed {install_name}.")
    finally:
        print(f"Importing {package}...")

required_packages = [
    "websockets", "numpy", "pandas", "scikit-learn",
    "xgboost", "catboost", "tensorflow", "keras", "joblib", "scipy",
]
try:
    import subprocess
    for package in required_packages:
        install_and_import(package)
except Exception as e:
    print(f"Error during package installation: {e}")
    print("Please install the required packages manually.")
    sys.exit(1)

# Now import the installed libraries
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Embedding, Input, Add, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---
warnings.filterwarnings('ignore') # Suppress TensorFlow and other warnings for a clean log
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("trainer_engine.log"),
                        logging.StreamHandler()
                    ])

# --- GLOBAL VARIABLES ---
APP_ID = "85473"
SYMBOL = "1HZ10V"
ROLLING_TRAINING_WINDOW = 3600  # Number of samples for training
MIN_TRAINING_SAMPLES = 1500  # Minimum samples to begin first training
MODEL_UPDATE_INTERVAL = 1500  # Train every 25 minutes
MODEL_DIR = "models"
last_digits_buffer: Deque[int] = deque(maxlen=ROLLING_TRAINING_WINDOW)
last_training_time: float = 0.0
executor = ThreadPoolExecutor(max_workers=1)

# --- UTILITY FUNCTIONS ---
def get_last_digit(price_str: str) -> int:
    """
    Correctly extracts the last digit from a price string, handling dropped zeros.
    """
    price_str = str(price_str)
    if '.' not in price_str:
        return int(price_str[-1])
    else:
        # Handle cases where trailing zeros are dropped
        if price_str.endswith('.0') or price_str.endswith('.00'):
            return 0
        # Use regex to find the last digit in the decimal part
        match = re.search(r'\d$', price_str)
        if match:
            return int(match.group(0))
        return 0

def generate_features_and_sequences(digits: Deque[int]) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """
    Generates features for both classical and sequential models from a deque.
    Creates a target vector (y) for the next digit.
    This function now uses a sliding window for both classical and sequential models.
    """
    digits_list = list(digits)
    
    classical_features_list = []
    y_classical_list = []
    
    sequence_length = 50
    sequential_sequences = []
    sequential_targets = []
    
    if len(digits_list) < sequence_length + 1:
        return pd.DataFrame(), np.array([]), pd.Series()
        
    for i in range(len(digits_list) - sequence_length):
        # Features for classical models (XGBoost, CatBoost)
        window = digits_list[i:i+sequence_length]
        features = {
            'std_digits': np.std(window), 'mean_digits': np.mean(window),
            'skew_digits': stats.skew(window), 'kurtosis_digits': stats.kurtosis(window),
            'max_digit': np.max(window), 'min_digit': np.min(window)
        }
        digit_counts = Counter(window)
        digit_freqs = {f'freq_digit_{i}': digit_counts.get(i, 0) / len(window) for i in range(10)}
        features.update(digit_freqs)
        
        for j in range(1, 11):
            features[f'lag_digit_{j}'] = window[-j]
        
        classical_features_list.append(features)
        y_classical_list.append(digits_list[i+sequence_length])
        
        # Data for sequential models (TCN, GRU, CNN)
        sequential_sequences.append(window)
        sequential_targets.append(digits_list[i+sequence_length])

    X_classical = pd.DataFrame(classical_features_list).dropna(axis=1)
    y_classical = pd.Series(y_classical_list, name='next_digit')
    
    X_seq = np.array(sequential_sequences)
    y_seq = np.array(sequential_targets)
    
    return X_classical, X_seq, y_classical

# --- TCN MODEL FIX ---
def TCN_Block(x, filters, kernel_size, dilation_rate):
    """
    A single TCN block with two dilated causal convolutions and a residual connection.
    Ensures input and output shapes are compatible for summation.
    """
    # First dilated causal convolution
    conv_1 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal', activation='relu')(x)
    # Second dilated causal convolution
    conv_2 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal', activation='relu')(conv_1)

    # Residual connection: Add the original input to the output of the block
    x = Add()([x, conv_2])
    return x

def build_tcn_model(input_shape):
    """
    Builds the Temporal Convolutional Network model.
    The architecture is fixed to ensure consistent shapes.
    """
    inputs = Input(shape=(input_shape,))
    # Embedding layer to handle categorical integer inputs
    x = Embedding(input_dim=10, output_dim=32)(inputs)
    # Initial Conv1D layer to match the filters of the TCN blocks
    x = Conv1D(filters=64, kernel_size=2, padding='causal', activation='relu')(x)
    
    # First TCN block with dilation rate 1
    x = TCN_Block(x, filters=64, kernel_size=2, dilation_rate=1)
    # Second TCN block with dilation rate 2
    x = TCN_Block(x, filters=64, kernel_size=2, dilation_rate=2)
    # Third TCN block with dilation rate 4
    x = TCN_Block(x, filters=64, kernel_size=2, dilation_rate=4)

    # Global average pooling to reduce dimensionality
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- ALL OTHER MODELS (UNCHANGED) ---
def build_gru_model(input_shape):
    """Builds the Gated Recurrent Unit model."""
    model = Sequential([
        Embedding(input_dim=10, output_dim=32, input_length=input_shape),
        GRU(64, return_sequences=True),
        GRU(32),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(input_shape):
    """Builds the Convolutional Neural Network model."""
    model = Sequential([
        Embedding(input_dim=10, output_dim=32, input_length=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        GlobalAveragePooling1D(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_models(models: Dict[str, Any], scaler: StandardScaler, accuracies: Dict[str, float]):
    """Saves trained models and scaler to disk."""
    logging.info("💾 Saving trained models to disk...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
        if 'XGBoost' in models:
            joblib.dump(models['XGBoost'], os.path.join(MODEL_DIR, 'xgb.joblib'))
        if 'CatBoost' in models:
            joblib.dump(models['CatBoost'], os.path.join(MODEL_DIR, 'cat.joblib'))
        if 'GRU' in models:
            models['GRU'].save(os.path.join(MODEL_DIR, 'gru.keras'))
        if 'TCN' in models:
            models['TCN'].save(os.path.join(MODEL_DIR, 'tcn.keras'))
        if 'CNN' in models:
            models['CNN'].save(os.path.join(MODEL_DIR, 'cnn.keras'))
        with open(os.path.join(MODEL_DIR, 'accuracies.json'), 'w') as f:
            json.dump(accuracies, f)
        logging.info("✅ Models saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save models: {e}")

# --- TRAINING FUNCTIONS ---
def train_classical_models(X_classical: pd.DataFrame, y_classical: pd.Series) -> Dict[str, Any]:
    """Trains XGBoost and CatBoost models."""
    trained_models = {}
    if len(y_classical) < 2:
        logging.info("Not enough samples for classical models. Skipping.")
        return trained_models

    X_train, _, y_train, _ = train_test_split(X_classical, y_classical, test_size=0.1, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Handle class imbalance based on unique classes in the training set
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Train XGBoost
    try:
        logging.info("Training XGBoost model...")
        # Removed use_label_encoder=False as it is deprecated and causing warnings
        xgb_model = XGBClassifier(eval_metric='mlogloss', n_jobs=-1)
        xgb_model.fit(X_train_scaled, y_train, sample_weight=np.array([class_weights[c] for c in y_train]))
        trained_models['XGBoost'] = xgb_model
        trained_models['scaler'] = scaler
    except Exception as e:
        logging.error(f"XGBoost training failed: {e}")

    # Train CatBoost
    try:
        logging.info("Training CatBoost model...")
        # Removed the incorrect 'classes_' parameter
        cat_model = CatBoostClassifier(verbose=0, class_weights=class_weights, thread_count=-1, allow_writing_files=False)
        cat_model.fit(X_train_scaled, y_train)
        trained_models['CatBoost'] = cat_model
    except Exception as e:
        logging.error(f"CatBoost training failed: {e}")
        
    return trained_models

def train_sequential_models(X_seq: np.ndarray, y_seq: np.ndarray) -> Dict[str, Any]:
    """Trains GRU, TCN, and CNN models."""
    trained_models = {}
    if len(y_seq) < 2:
        logging.info("Not enough sequential samples. Skipping sequential models.")
        return trained_models

    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train GRU
    try:
        logging.info("Training GRU model...")
        gru_model = build_gru_model(X_train.shape[1])
        gru_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)
        trained_models['GRU'] = gru_model
        logging.info(f"GRU training complete. Accuracy: {accuracy_score(y_val, np.argmax(gru_model.predict(X_val, verbose=0), axis=1)):.2%}")
    except Exception as e:
        logging.error(f"GRU training failed: {e}")

    # Train TCN
    try:
        logging.info("Training TCN model...")
        tcn_model = build_tcn_model(X_train.shape[1])
        tcn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)
        trained_models['TCN'] = tcn_model
        logging.info(f"TCN training complete. Accuracy: {accuracy_score(y_val, np.argmax(tcn_model.predict(X_val, verbose=0), axis=1)):.2%}")
    except Exception as e:
        logging.error(f"TCN training failed: {e}")

    # Train CNN
    try:
        logging.info("Training CNN model...")
        cnn_model = build_cnn_model(X_train.shape[1])
        cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)
        trained_models['CNN'] = cnn_model
        logging.info(f"CNN training complete. Accuracy: {accuracy_score(y_val, np.argmax(cnn_model.predict(X_val, verbose=0), axis=1)):.2%}")
    except Exception as e:
        logging.error(f"CNN training failed: {e}")
    
    return trained_models

def train_and_send_models(queue: Queue):
    """Main training function to be executed in a separate thread."""
    logging.info("🤖 Initiating model training process...")
    logging.info(f"Training on {len(last_digits_buffer)} samples.")

    X_classical, X_seq, y_classical = generate_features_and_sequences(last_digits_buffer)
    y_seq = y_classical.values
    
    all_models = {}
    all_accuracies = {}
    scaler = None

    # Train classical models
    classical_models = train_classical_models(X_classical, y_classical)
    all_models.update({k: v for k,v in classical_models.items() if k != 'scaler'})
    scaler = classical_models.get('scaler')
    
    # Train sequential models
    sequential_models = train_sequential_models(X_seq, y_seq)
    all_models.update(sequential_models)

    # Calculate and store accuracies for all models
    for name, model in all_models.items():
        if name in ['XGBoost', 'CatBoost']:
            if scaler is not None and len(X_classical) > 0:
                X_scaled = scaler.transform(X_classical)
                pred = model.predict(X_scaled)
                acc = accuracy_score(y_classical, pred)
            else:
                acc = 0.0
        elif name in ['GRU', 'TCN', 'CNN']:
            if len(X_seq) > 0:
                pred = np.argmax(model.predict(X_seq, verbose=0), axis=1)
                acc = accuracy_score(y_seq, pred)
            else:
                acc = 0.0
        all_accuracies[name] = acc
        logging.info(f"📊 Model {name} accuracy: {acc:.2%}")

    if all_models:
        queue.put((all_models, all_accuracies, scaler))
        save_models(all_models, scaler, all_accuracies)
        logging.info("✅ New models and accuracies sent to the main process via the queue.")
    else:
        logging.warning("No models were successfully trained. Queue remains empty.")

async def run_tick_collector_and_trainer(queue: Queue):
    """
    Establishes WebSocket connection and manages the data collection loop.
    This function is run by the `main.py` child process.
    """
    global last_training_time
    last_training_time = time.time()
    
    try:
        async with websockets.connect(f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}") as ws:
            logging.info("Connected to Deriv WebSocket. Awaiting ticks...")
            await ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))
            
            while True:
                data = json.loads(await ws.recv())
                if data.get("msg_type") == "tick":
                    tick = data['tick']
                    last_digit = get_last_digit(tick['quote'])
                    last_digits_buffer.append(last_digit)
                    logging.debug(f"Received tick: {last_digit}. Buffer size: {len(last_digits_buffer)}")

                    if len(last_digits_buffer) >= MIN_TRAINING_SAMPLES and (time.time() - last_training_time) >= MODEL_UPDATE_INTERVAL:
                        logging.info("Training conditions met. Starting model training in a separate thread.")
                        await asyncio.get_event_loop().run_in_executor(executor, train_and_send_models, queue)
                        last_training_time = time.time()
                
                await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        logging.info("Tick collector process cancelled. Shutting down...")
    except Exception as e:
        logging.critical(f"🔥🔥 CRITICAL ERROR in trainer: {e}🔥🔥")
    finally:
        executor.shutdown(wait=True)
        logging.info("✅ Analytical engine shut down gracefully.")