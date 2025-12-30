# main.py
# The Executive Brain - The sole entry point for the bot.
# It manages real-time trading, risk, and launches the analytical engine
# (model_trainer.py) in a separate process.

#To maximize it's potential and execution speeds, please run it on google colab

import asyncio
import websockets
import json
import random
from datetime import datetime
import time
import os
import numpy as np
import logging
import joblib
import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue
from typing import Dict, Any, Optional, Tuple
from collections import deque, Counter, defaultdict

# --- Self-Sufficiency and Dependency Management ---
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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
# Force TensorFlow to use the CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("main_bot.log"),
                        logging.StreamHandler()
                    ])

# --- CONFIG ---
API_TOKEN = "6iGI0fIFqDzZsN0"  # Relace with your actual Api token. Get yours at https://api.deriv.com/
SYMBOL = "1HZ10V" # Replce with your desired symbol. On deriv trader interface, check the url for clues eg R_100 etc

# Feel free to adjust the constants below to your liking. Note, only use this with demo accounts only. This bot does not guarantee profits in any case. Feel free to optimize the bot to your best liking.
DAILY_LOSS_LIMIT = 3.00
DAILY_PROFIT_TARGET = 5.00
MIN_CONFIDENCE_THRESHOLD = 0.51
LOSS_STREAK_LIMIT = 3
TRADE_DURATION_SECONDS = 1
MODEL_DIR = "models"
MODEL_COMM_QUEUE = None  # The multiprocessing queue
BACKGROUND_PROCESS = None # The multiprocessing process
os.makedirs(MODEL_DIR, exist_ok=True)


class TradingBrain:
    """Encapsulates the bot's state, models, and trading logic."""
    def __init__(self):
        self.account_balance: float = 0.0
        self.daily_profit: float = 0.0
        self.total_trades: int = 0
        self.trading_active: bool = True
        self.consecutive_losses: int = 0
        self.last_digits_buffer: deque = deque(maxlen=200)
        self.active_models: Dict[str, Any] = {}
        self.model_accuracies: Dict[str, float] = {}
        self.scaler: Optional[StandardScaler] = None
        self.last_prediction_time: float = 0.0

    def get_last_digit(self, price_str: str) -> int:
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
    
    def generate_features_and_sequences(self, digits: deque) -> Tuple[Any, Any]:
        """Generates features for classical and sequential models from a deque."""
        digits_list = list(digits)
        sequence_length = 50
        
        if len(digits_list) < sequence_length + 1:
            return None, None

        # Get the last window for prediction
        window = digits_list[-sequence_length:]
        
        # Features for classical models (XGBoost, CatBoost)
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
        
        classical_features = pd.DataFrame([features])
        X_seq = np.array([window])
        
        return classical_features, X_seq

    def load_latest_models(self):
        """Loads the latest models from the queue, if available, or from disk."""
        # First, try to load from the queue (newly trained models)
        if not MODEL_COMM_QUEUE.empty():
            try:
                new_models, new_accuracies, new_scaler = MODEL_COMM_QUEUE.get_nowait()
                self.active_models = new_models
                self.model_accuracies = new_accuracies
                self.scaler = new_scaler
                logging.info("🧠 Brain updated! Loaded new models from the analytical engine.")
                return True
            except Exception as e:
                logging.error(f"Error loading models from queue: {e}")
                return False
        
        # If queue is empty, try to load from disk
        if not self.active_models:
            logging.info("No new models in queue. Attempting to load from disk...")
            try:
                # Load scaler first, as it's a dependency for classical models
                self.scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
                # Load models if their files exist
                if os.path.exists(os.path.join(MODEL_DIR, 'xgb.joblib')):
                    self.active_models['XGBoost'] = joblib.load(os.path.join(MODEL_DIR, 'xgb.joblib'))
                if os.path.exists(os.path.join(MODEL_DIR, 'cat.joblib')):
                    self.active_models['CatBoost'] = joblib.load(os.path.join(MODEL_DIR, 'cat.joblib'))
                if os.path.exists(os.path.join(MODEL_DIR, 'gru.keras')):
                    self.active_models['GRU'] = keras.models.load_model(os.path.join(MODEL_DIR, 'gru.keras'))
                if os.path.exists(os.path.join(MODEL_DIR, 'tcn.keras')):
                    self.active_models['TCN'] = keras.models.load_model(os.path.join(MODEL_DIR, 'tcn.keras'))
                if os.path.exists(os.path.join(MODEL_DIR, 'cnn.keras')):
                    self.active_models['CNN'] = keras.models.load_model(os.path.join(MODEL_DIR, 'cnn.keras'))

                with open(os.path.join(MODEL_DIR, 'accuracies.json')) as f:
                    self.model_accuracies = json.load(f)
                
                logging.info("✅ Models loaded successfully from disk.")
                return True
            except Exception as e:
                logging.warning(f"Failed to load models from disk: {e}")
                return False

    def get_prediction(self) -> Tuple[int, float]:
        """Gets a prediction from the weighted ensemble model and its confidence."""
        if not self.active_models or len(self.last_digits_buffer) < 51:
            return -1, 0.0

        try:
            classical_features, X_seq = self.generate_features_and_sequences(self.last_digits_buffer)
            if classical_features is None or X_seq is None:
                return -1, 0.0

            ensemble_probabilities = defaultdict(float)
            total_weight = 0.0

            # Get predictions and weights from all models
            for model_name, model in self.active_models.items():
                if model_name in self.model_accuracies:
                    weight = self.model_accuracies[model_name]
                    if weight == 0.0: continue
                    total_weight += weight

                    if model_name in ['XGBoost', 'CatBoost']:
                        if self.scaler:
                            X_classical_scaled = self.scaler.transform(classical_features)
                            proba = model.predict_proba(X_classical_scaled)[0]
                        else:
                            continue # Skip if no scaler for classical models
                    elif model_name in ['GRU', 'TCN', 'CNN']:
                        proba = model.predict(X_seq, verbose=0)[0]
                    else:
                        continue
                    
                    for digit_idx in range(10):
                        ensemble_probabilities[digit_idx] += proba[digit_idx] * weight
            
            if total_weight == 0.0:
                return -1, 0.0

            # Normalize probabilities
            for digit in ensemble_probabilities:
                ensemble_probabilities[digit] /= total_weight

            predicted_digit = max(ensemble_probabilities, key=ensemble_probabilities.get)
            confidence = ensemble_probabilities[predicted_digit]
            
            return int(predicted_digit), float(confidence)

        except Exception as e:
            logging.error(f"PREDICTION ERROR: {e}")
            return -1, 0.0

async def run_bot():
    """Main execution loop for the bot."""
    global BACKGROUND_PROCESS, MODEL_COMM_QUEUE
    
    # 1. Initialize the communication queue and start the analytical process
    MODEL_COMM_QUEUE = Queue()
    BACKGROUND_PROCESS = Process(target=start_model_trainer_process, args=(MODEL_COMM_QUEUE,))
    BACKGROUND_PROCESS.start()
    logging.info("🧠 Analytical engine (model_trainer.py) started as a background process.")

    brain = TradingBrain()
    
    try:
        logging.info("Connecting to Deriv WebSocket...")
        async with websockets.connect(f"wss://ws.derivws.com/websockets/v3?app_id=85473") as ws:
            logging.info("Sending authorization request...")
            await ws.send(json.dumps({"authorize": API_TOKEN}))
            auth_response = json.loads(await ws.recv())
            
            if "error" in auth_response:
                logging.error(f"Authentication Failed! Error: {auth_response['error']['message']}")
                BACKGROUND_PROCESS.terminate()
                return

            brain.account_balance = float(auth_response["authorize"]["balance"])
            logging.info(f"Authorization successful. Starting balance: ${brain.account_balance:.2f}")
            await ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))
            await asyncio.sleep(1)
            
            # Attempt to load models from disk on startup
            brain.load_latest_models()

            logging.info("Bot is active, awaiting trade opportunities.")
            
            while True:
                # 2. Check for new models from the background process
                brain.load_latest_models()
                
                # Receive tick data and add to buffer
                response = json.loads(await ws.recv())
                if response.get("msg_type") == "tick":
                    last_digit = brain.get_last_digit(response['tick']['quote'])
                    brain.last_digits_buffer.append(last_digit)
                else:
                    logging.debug(f"Received a non-tick message type: {response.get('msg_type')}. Ignoring for now.")
                    continue

                # 3. Get prediction and trade if conditions are met
                predicted_digit, confidence = brain.get_prediction()
                
                if predicted_digit == -1 or confidence < MIN_CONFIDENCE_THRESHOLD:
                    logging.info(f"Prediction confidence too low ({confidence:.2%}) or models not ready. Skipping trade.")
                    continue
                
                if not brain.trading_active:
                    logging.info("Daily limits reached. Trading paused.")
                    await asyncio.sleep(60)
                    continue

                # 4. Execute the trade
                stake_to_use = 0.35 + (confidence - MIN_CONFIDENCE_THRESHOLD) * 0.5
                stake_to_use = round(stake_to_use, 2)
                
                trade_type = "DIGITMATCH"
                barrier_digit = predicted_digit

                logging.info(f"🔮 Attempting trade: Pred={predicted_digit} (Conf: {confidence:.2%}), Stake=${stake_to_use:.2f}")

                try:
                    proposal_req = {
                        "proposal": 1, "amount": stake_to_use, "basis": "stake",
                        "contract_type": trade_type, "currency": "USD", "duration": TRADE_DURATION_SECONDS,
                        "duration_unit": "t", "symbol": SYMBOL, "barrier": str(barrier_digit)
                    }
                    
                    await ws.send(json.dumps(proposal_req))
                    proposal = json.loads(await ws.recv())
                    if "error" in proposal:
                        logging.error(f"Proposal Error: {proposal['error']['message']}. Skipping.")
                        continue

                    await ws.send(json.dumps({"buy": proposal["proposal"]["id"], "price": stake_to_use}))
                    buy_response = json.loads(await ws.recv())
                    if "error" in buy_response:
                        logging.error(f"Buy Error: {buy_response['error']['message']}. Skipping.")
                        continue
                except Exception as e:
                    logging.error(f"API Call Error: {e}. Skipping this trade.")
                    continue
                
                # Wait for the next tick to get the outcome
                try:
                    next_tick_data = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                    while next_tick_data.get('msg_type') != 'tick':
                        next_tick_data = json.loads(await ws.recv())
                except asyncio.TimeoutError:
                    logging.error("Timeout waiting for next tick. Cannot determine trade outcome.")
                    brain.consecutive_losses += 1
                    continue
                except Exception as e:
                    logging.error(f"Error receiving next tick: {e}")
                    brain.consecutive_losses += 1
                    continue

                outcome_digit = brain.get_last_digit(next_tick_data['tick']['quote'])
                win = outcome_digit == predicted_digit
                profit_amount = buy_response['buy']['transaction']['profit'] if 'transaction' in buy_response['buy'] else (stake_to_use if win else -stake_to_use)

                # Update the buffer with the new digit
                brain.last_digits_buffer.append(outcome_digit)

                brain.account_balance += profit_amount
                brain.daily_profit += profit_amount
                brain.consecutive_losses = 0 if win else brain.consecutive_losses + 1
                brain.total_trades += 1
                
                outcome_str = "WIN" if win else "LOSS"
                logging.info(f"✅ Trade Result: {outcome_str}. P/L: {profit_amount:+.2f}. Consecutive losses: {brain.consecutive_losses}. Total Trades: {brain.total_trades}")
                logging.info(f"📊 Daily P/L: ${brain.daily_profit:+.2f}")
                
                if brain.daily_profit <= -DAILY_LOSS_LIMIT or brain.daily_profit >= DAILY_PROFIT_TARGET:
                    brain.trading_active = False
                    logging.info("DAILY LIMIT REACHED. TRADING PAUSED.")
                
                # After a trade, take a short break to not overwhelm the API
                await asyncio.sleep(1)


    except (asyncio.CancelledError, KeyboardInterrupt):
        logging.info("🚨 Bot process cancelled. Shutting down analytical engine...")
    except Exception as e:
        logging.critical(f"🔥🔥 CRITICAL ERROR: {e}🔥🔥")
    finally:
        if BACKGROUND_PROCESS:
            BACKGROUND_PROCESS.terminate()
            logging.info("✅ Analytical engine shut down gracefully.")

def start_model_trainer_process(queue: Queue):
    """Function to be run by the child process."""
    try:
        from model_trainer import run_tick_collector_and_trainer
        asyncio.run(run_tick_collector_and_trainer(queue))
    except ImportError as e:
        logging.error(f"Failed to import model_trainer.py. Ensure the file exists: {e}")
    except Exception as e:
        logging.critical(f"Error in model_trainer process: {e}")

if __name__ == "__main__":

    asyncio.run(run_bot())
