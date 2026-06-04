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

## Real-Time Inference

Run real-time inference with the trained PyTorch model:

```powershell
.\venv\Scripts\python.exe inference.py --model-path best_emg_model.pt --window-size 200 --filter-mode highpass_20
```

Run with the status GUI:

```powershell
.\venv\Scripts\python.exe inference.py --gui --model-path best_emg_model.pt --window-size 200 --filter-mode highpass_20
```

The script waits for the ESP32 TCP client on `SERVER_PORT` from `config.py`, keeps a rolling signal buffer, applies the selected streaming filter, calibrates the RMS trigger threshold, stabilizes repeated predictions with a short vote window, and prints or displays predicted keys.

Useful options:

```text
--gui                 Show a PyQt status window
--threshold           Manual RMS fallback threshold, default 80000
--manual-threshold    Disable automatic threshold calibration
--calibration-seconds Idle calibration duration, default 3.0
--threshold-multiplier threshold = idle_mean + multiplier * idle_std
--min-confidence      Minimum softmax confidence, default 0.35
--min-margin          Minimum top-1 minus top-2 confidence margin, default 0.10
--vote-window         Number of recent candidates to vote over, default 3
--min-votes           Required votes for one emitted prediction, default 2
--cooldown-samples    Samples to wait after a prediction, default 200
--print-rms           Print recent RMS once per interval
--device              auto, cpu, or cuda
```

Threshold calibration example:

```powershell
.\venv\Scripts\python.exe inference.py --print-rms --manual-threshold --threshold 999999999
```

Watch the idle RMS and typing RMS, then rerun with a threshold between those two ranges.

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

`inference.py` uses the PyTorch model (`best_emg_model.pt`) trained by `train.py`.
