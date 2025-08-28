The Analytical Engine: A Tick-Based Trading Bot 🤖

This project consists of a high-frequency trading bot designed to predict the last digit of a synthetic index's price tick. It uses a dual-process architecture to separate real-time trading from intensive model training, ensuring optimal performance and responsiveness.

✨ Features
Dual-Process Architecture: The core trading logic runs in a main process (main.py) while a separate process (model_trainer.py) continuously collects data and trains machine learning models in the background. This prevents the bot from freezing during model updates.

Multi-Model Ensemble: The bot trains and uses an ensemble of both classical and deep learning models, including XGBoost, CatBoost, GRU, TCN, and CNN.

Dynamic Confidence-Based Trading: Trades are only executed when the ensemble model's confidence in its prediction exceeds a predefined threshold.

Self-Sufficient Setup: The script automatically installs all necessary Python libraries on the first run, making it easy to set up.

Robust Risk Management: It includes built-in daily loss and profit limits, as well as a consecutive loss counter to prevent major drawdowns.

💻 How It Works
The system is divided into two main components:

1. main.py (The Executive Brain 🧠)
This is the heart of the bot. It's responsible for:

Connecting to the Deriv WebSocket API.

Managing the trading logic, including executing trades and managing risk.

Starting model_trainer.py as a separate, non-blocking process.

Receiving updated models and accuracies from the trainer process via a multiprocessing queue.

2. model_trainer.py (The Analytical Engine 🔬)
This process runs in the background and is responsible for:

Collecting a continuous stream of tick data from the WebSocket.

Periodically generating features and sequences from the collected data.

Training and evaluating all the machine learning models.

Saving the best-performing models to disk.

Sending the newly trained models to the main process for use in live trading.

🚀 Getting Started
Prerequisites
A Deriv API token get one here https://api.deriv.com/.

Python 3.7 or higher.

Installation
Clone this repository:

Navigate to the project directory:
cd your-repo-name

Update the API_TOKEN variable in main.py with your personal token.

Run the main script:
python main.py

The script will handle the rest, automatically installing the required dependencies and starting the trading bot.

☁️ Recommended: Run on Google Colab
For a significant performance boost and seamless execution, we highly recommend running this script on Google Colab. The enhanced computing power, particularly the dedicated GPU access, will dramatically speed up the training of the deep learning models (GRU, TCN, and CNN), leading to faster and more frequent model updates.

To use it on Colab, simply upload both main.py and model_trainer.py to your session and run main.py in a code cell.
