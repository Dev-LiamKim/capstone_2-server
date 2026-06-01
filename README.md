# EMG Keyboard Classifier

8-channel EMG data collection and right-hand keyboard classification project.

This repository contains tools for:

- receiving EMG samples from an ESP32 over TCP
- visualizing raw and processed EMG signals in real time
- recording key-labeled CSV datasets through a typing practice UI
- training a PyTorch CNN+LSTM classifier
- analyzing dataset/event quality

## Project Structure

```text
.
├── main.py                 # Real-time EMG collector app
├── network.py              # TCP receiver for ESP32 packets
├── gui.py                  # PyQtGraph signal visualizer
├── typing_practice.py      # Key prompt UI and CSV event recording
├── train.py                # PyTorch training pipeline
├── inference.py            # Legacy TensorFlow/Keras inference script
├── analyzer.py             # Dataset event count analyzer
├── check_class_jitter.py   # Event-to-peak jitter analysis
├── check_sync_pure.py      # Class-wise physical peak report
├── config.py               # Shared constants and key mapping
└── requirements.txt
```

## Data and Model Artifacts

Datasets and experiment reports can be committed with this project:

- `dataset/`
- `new_dataset/`
- `results/`

Local runtime artifacts and model checkpoints are intentionally excluded from Git:

- `*.pt`
- `*.keras`
- `*.npy`
- `*.db`
- `*.docx`

Keep trained model files in the project directory when running experiments, but avoid committing them unless a release artifact is intentionally needed.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

If installing CUDA-enabled PyTorch fails from `requirements.txt`, install the PyTorch build matching your local CUDA version from the official PyTorch instructions, then install the remaining requirements.

## Collect Data

Run the collector and connect the ESP32 client to `SERVER_PORT` from `config.py`.

```powershell
.\venv\Scripts\python.exe main.py
```

The app opens:

- a real-time EMG signal visualizer
- a typing practice window for event labeling

Successful sessions are saved as CSV files under `dataset/`.

## Train

The current best experiment used `new_dataset`, a 200-sample window, 80 pre-event samples, and 20 Hz high-pass filtering.

```powershell
.\venv\Scripts\python.exe train.py --dataset-dir new_dataset --seed 42 --window-size 200 --pre-event 80 --max-epochs 80 --patience 8 --filter-mode highpass_20 --model cnn_lstm
```

Useful options:

```text
--dataset-dir
--window-size
--pre-event
--filter-mode
--notch-filter
--align-peak
--model
--seed
--max-epochs
--patience
```

Training writes:

- `best_emg_model.pt`
- `results/runs/<timestamp>/report.txt`
- `results/runs/<timestamp>/training_log.csv`
- `results/runs/<timestamp>/training_history.png`
- `results/runs/<timestamp>/confusion_matrix.png`

## Current Best Result

See the local report:

```text
EMG_모델_개선_실험_보고서_20260601.md
```

Summary:

- Dataset: `new_dataset`
- Model: CNN+LSTM
- Window: 200 samples
- Pre/Post event: 80 / 120 samples
- Filter: high-pass 20 Hz
- Test Accuracy: 76.84%

## Notes

`inference.py` currently loads a TensorFlow/Keras model (`best_emg_model.keras`), while `train.py` now trains a PyTorch model (`best_emg_model.pt`). For deployment, the inference path should be updated to use the PyTorch model or a model conversion step should be added.
