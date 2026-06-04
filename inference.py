import argparse
import time
from collections import Counter, deque

import numpy as np
import torch
from scipy import signal

from config import (
    BATCH_SIZE,
    CHANNELS,
    NOTCH_F0,
    NOTCH_Q,
    RIGHT_HAND_KEYS,
    SAMPLING_RATE_THEORETICAL,
    SERVER_PORT,
)
from network import EMGReceiver
from models import CNNLSTM, ResNet1D


NUM_CLASSES = len(set(RIGHT_HAND_KEYS.values()))


class ThresholdCalibrator:
    def __init__(self, enabled, duration_sec, multiplier, fallback_threshold):
        self.enabled = enabled and duration_sec > 0
        self.duration_sec = duration_sec
        self.multiplier = multiplier
        self.fallback_threshold = fallback_threshold
        self.started_at = None
        self.samples = []
        self.threshold = fallback_threshold
        self.done = not self.enabled

    def update(self, rms):
        if self.done:
            return self.threshold

        now = time.time()
        if self.started_at is None:
            self.started_at = now
        self.samples.append(rms)

        if now - self.started_at >= self.duration_sec and self.samples:
            mean = float(np.mean(self.samples))
            std = float(np.std(self.samples))
            self.threshold = mean + self.multiplier * std
            self.done = True
            print(
                "[CALIBRATION] "
                f"mean={mean:.1f}, std={std:.1f}, threshold={self.threshold:.1f}"
            )
        return self.threshold

    def progress(self):
        if self.done:
            return 1.0
        if self.started_at is None:
            return 0.0
        return min(1.0, (time.time() - self.started_at) / self.duration_sec)


class PredictionSmoother:
    def __init__(self, min_confidence, min_margin, vote_window, min_votes):
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.vote_window = max(1, vote_window)
        self.min_votes = max(1, min_votes)
        self.votes = deque(maxlen=self.vote_window)

    def update(self, candidate):
        if candidate["confidence"] < self.min_confidence:
            return {
                "type": "skip",
                "reason": "low confidence",
                "candidate": candidate,
            }

        if candidate["margin"] < self.min_margin:
            return {
                "type": "skip",
                "reason": "low margin",
                "candidate": candidate,
            }

        self.votes.append(candidate)
        counts = Counter(item["key"] for item in self.votes)
        key, count = counts.most_common(1)[0]

        if count >= self.min_votes:
            selected = max(
                (item for item in self.votes if item["key"] == key),
                key=lambda item: item["confidence"],
            )
            self.votes.clear()
            return {"type": "emit", "candidate": selected}

        return {
            "type": "pending",
            "reason": f"vote {count}/{self.min_votes}",
            "candidate": candidate,
        }


class StreamingFilter:
    def __init__(self, mode, fs):
        self.mode = mode
        self.fs = fs
        self.filters = self._build_filters(mode, fs)
        self.states = [
            [signal.lfilter_zi(b, a) * 0.0 for b, a in self.filters]
            for _ in range(CHANNELS)
        ]

    def _build_filters(self, mode, fs):
        if mode == "raw":
            return []
        if mode == "notch":
            return [signal.iirnotch(NOTCH_F0, NOTCH_Q, fs=fs)]
        if mode == "highpass_20":
            return [signal.butter(4, 20.0, btype="highpass", fs=fs)]
        if mode == "highpass_20_notch":
            return [
                signal.butter(4, 20.0, btype="highpass", fs=fs),
                signal.iirnotch(NOTCH_F0, NOTCH_Q, fs=fs),
            ]
        raise ValueError(f"Unsupported filter mode: {mode}")

    def process(self, batch):
        if self.mode == "raw":
            return batch.astype(np.float32)

        processed = np.empty_like(batch, dtype=np.float64)
        for ch_idx in range(CHANNELS):
            x = batch[ch_idx].astype(np.float64)
            for filter_idx, (b, a) in enumerate(self.filters):
                x, self.states[ch_idx][filter_idx] = signal.lfilter(
                    b,
                    a,
                    x,
                    zi=self.states[ch_idx][filter_idx],
                )
            processed[ch_idx] = x
        return processed.astype(np.float32)


class EMGRealTimeInference:
    def __init__(self, args):
        self.args = args
        self.device = self._select_device(args.device)
        self.receiver = EMGReceiver(args.port)
        self.model = self._load_model(args.model_path, args.model)
        self.model.eval()

        self.raw_labels = sorted(set(RIGHT_HAND_KEYS.values()))
        self.idx_to_label = {idx: raw_id for idx, raw_id in enumerate(self.raw_labels)}
        self.inv_keys = {v: k for k, v in RIGHT_HAND_KEYS.items()}

        self.data_buffer = np.zeros((CHANNELS, args.buffer_size), dtype=np.float32)
        self.filter = StreamingFilter(args.filter_mode, args.sample_rate)
        self.calibrator = ThresholdCalibrator(
            enabled=args.auto_threshold,
            duration_sec=args.calibration_seconds,
            multiplier=args.threshold_multiplier,
            fallback_threshold=args.threshold,
        )
        self.smoother = PredictionSmoother(
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            vote_window=args.vote_window,
            min_votes=args.min_votes,
        )
        self.threshold = args.threshold
        self.cooldown = 0
        self.connected_addr = None
        self.last_rms_log = time.time()
        self.prediction_count = 0
        self.last_status = {
            "type": "boot",
            "message": "model loaded",
            "rms": 0.0,
            "threshold": self.threshold,
            "top": "-",
        }

    def _select_device(self, requested):
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _load_model(self, model_path, model_name):
        if model_name == "resnet1d":
            model = ResNet1D(CHANNELS, NUM_CLASSES)
        else:
            model = CNNLSTM(CHANNELS, NUM_CLASSES)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        print(f"[MODEL] loaded {model_name} from {model_path} on {self.device}")
        return model

    def run(self):
        self.wait_for_connection()

        while True:
            self.process_once()

    def wait_for_connection(self):
        if self.args.auto_threshold:
            print(
                "[CALIBRATION] Keep your hand relaxed for "
                f"{self.args.calibration_seconds:.1f}s after connection."
            )
        self.connected_addr = self.receiver.wait_for_connection()
        print(f"[READY] ESP32 connected from {self.connected_addr}")
        print(
            "[CONFIG] "
            f"window={self.args.window_size}, filter={self.args.filter_mode}, "
            f"threshold={self.args.threshold}, cooldown={self.args.cooldown_samples}, "
            f"votes={self.args.min_votes}/{self.args.vote_window}"
        )
        return self.connected_addr

    def process_once(self):
        batch = self.receiver.receive_batch()
        if batch is None:
            return None

        processed_batch = self.filter.process(batch)
        self._append_buffer(processed_batch)

        recent_rms = self._recent_rms(processed_batch)
        self._maybe_log_rms(recent_rms)
        self.threshold = self.calibrator.update(recent_rms)

        if not self.calibrator.done:
            self.last_status = {
                "type": "calibrating",
                "message": "calibrating threshold",
                "rms": recent_rms,
                "threshold": self.threshold,
                "progress": self.calibrator.progress(),
                "top": "-",
            }
            return self.last_status

        if self.cooldown > 0:
            self.cooldown = max(0, self.cooldown - BATCH_SIZE)
            self.last_status = {
                "type": "cooldown",
                "message": f"cooldown {self.cooldown} samples",
                "rms": recent_rms,
                "threshold": self.threshold,
                "top": "-",
            }
            return self.last_status

        if recent_rms < self.threshold:
            self.last_status = {
                "type": "idle",
                "message": "waiting for EMG trigger",
                "rms": recent_rms,
                "threshold": self.threshold,
                "top": "-",
            }
            return self.last_status

        candidate = self._classify(recent_rms)
        decision = self.smoother.update(candidate)
        self.last_status = self._handle_decision(decision, recent_rms)
        return self.last_status

    def _append_buffer(self, batch):
        self.data_buffer = np.roll(self.data_buffer, -BATCH_SIZE, axis=1)
        self.data_buffer[:, -BATCH_SIZE:] = batch

    def _recent_rms(self, batch):
        return float(np.sqrt(np.mean(np.square(batch))))

    def _maybe_log_rms(self, recent_rms):
        if not self.args.print_rms:
            return
        now = time.time()
        if now - self.last_rms_log >= self.args.rms_log_interval:
            print(f"[RMS] {recent_rms:.1f}")
            self.last_rms_log = now

    def _classify(self, recent_rms):
        window = self.data_buffer[:, -self.args.window_size:].T
        x = self._normalize_window(window)
        x_tensor = torch.from_numpy(x[None, :, :]).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        sorted_indices = np.argsort(probs)[::-1]
        second_confidence = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = confidence - second_confidence
        raw_id = self.idx_to_label[pred_idx]
        key_char = self.inv_keys.get(raw_id, f"Unknown({raw_id})")
        return {
            "key": key_char,
            "raw_id": raw_id,
            "confidence": confidence,
            "margin": margin,
            "rms": recent_rms,
            "top": self._format_top_k(probs, self.args.top_k),
        }

    def _handle_decision(self, decision, recent_rms):
        candidate = decision["candidate"]
        if decision["type"] == "skip":
            print(
                f"[SKIP] reason={decision['reason']} key='{candidate['key']}' "
                f"confidence={candidate['confidence'] * 100:.1f}% "
                f"margin={candidate['margin'] * 100:.1f}% rms={recent_rms:.1f}"
            )
            return {
                "type": "skip",
                "message": decision["reason"],
                "rms": recent_rms,
                "threshold": self.threshold,
                **candidate,
            }

        if decision["type"] == "pending":
            return {
                "type": "pending",
                "message": decision["reason"],
                "rms": recent_rms,
                "threshold": self.threshold,
                **candidate,
            }

        self.prediction_count += 1
        print(
            f"[PREDICT #{self.prediction_count}] key='{candidate['key']}' "
            f"confidence={candidate['confidence'] * 100:.1f}% "
            f"margin={candidate['margin'] * 100:.1f}% rms={recent_rms:.1f} "
            f"top={candidate['top']}"
        )
        self.cooldown = self.args.cooldown_samples
        return {
            "type": "predict",
            "message": "prediction emitted",
            "rms": recent_rms,
            "threshold": self.threshold,
            "count": self.prediction_count,
            **candidate,
        }

    def _normalize_window(self, window):
        mean = np.mean(window, axis=0, keepdims=True)
        std = np.std(window, axis=0, keepdims=True) + 1e-8
        return ((window - mean) / std).astype(np.float32)

    def _format_top_k(self, probs, top_k):
        if top_k <= 0:
            return "-"
        indices = np.argsort(probs)[::-1][:top_k]
        items = []
        for idx in indices:
            raw_id = self.idx_to_label[int(idx)]
            key_char = self.inv_keys.get(raw_id, f"Unknown({raw_id})")
            items.append(f"{key_char}:{probs[idx] * 100:.1f}%")
        return ", ".join(items)


class InferenceStatusWindow:
    def __init__(self, engine):
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.QtCore = QtCore
        self.engine = engine
        self.widget = QtWidgets.QWidget()
        self.widget.setWindowTitle("EMG Real-Time Inference")
        self.widget.resize(620, 420)

        root = QtWidgets.QVBoxLayout(self.widget)
        root.setSpacing(10)

        self.lbl_connection = QtWidgets.QLabel("Connected")
        self.lbl_connection.setStyleSheet("font-size: 14px; color: #2e7d32;")
        root.addWidget(self.lbl_connection)

        self.lbl_key = QtWidgets.QLabel("-")
        self.lbl_key.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_key.setMinimumHeight(110)
        self.lbl_key.setStyleSheet(
            "font-size: 72px; font-weight: bold; color: #0d47a1; "
            "background: #eef4ff; border: 1px solid #c7d7f2; border-radius: 6px;"
        )
        root.addWidget(self.lbl_key)

        self.lbl_status = QtWidgets.QLabel("Waiting for signal")
        self.lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold;")
        root.addWidget(self.lbl_status)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        root.addLayout(grid)

        self.lbl_rms = self._make_value_label()
        self.lbl_threshold = self._make_value_label()
        self.lbl_confidence = self._make_value_label()
        self.lbl_margin = self._make_value_label()

        self._add_metric(grid, 0, "RMS", self.lbl_rms)
        self._add_metric(grid, 1, "Threshold", self.lbl_threshold)
        self._add_metric(grid, 2, "Confidence", self.lbl_confidence)
        self._add_metric(grid, 3, "Margin", self.lbl_margin)

        self.lbl_top = QtWidgets.QLabel("-")
        self.lbl_top.setWordWrap(True)
        self.lbl_top.setStyleSheet("font-size: 15px; color: #333;")
        root.addWidget(self.lbl_top)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)

    def _make_value_label(self):
        from pyqtgraph.Qt import QtWidgets

        label = QtWidgets.QLabel("-")
        label.setStyleSheet("font-size: 18px; font-weight: bold;")
        return label

    def _add_metric(self, grid, row, name, value_label):
        from pyqtgraph.Qt import QtWidgets

        name_label = QtWidgets.QLabel(name)
        name_label.setStyleSheet("font-size: 14px; color: #666;")
        grid.addWidget(name_label, row, 0)
        grid.addWidget(value_label, row, 1)

    def show(self, interval_ms):
        self.widget.show()
        self.timer.start(interval_ms)

    def tick(self):
        state = self.engine.process_once()
        if state is None:
            state = self.engine.last_status
        self.update_state(state)

    def update_state(self, state):
        state_type = state.get("type", "idle")
        key = state.get("key", "-") if state_type in {"predict", "pending", "skip"} else "-"
        confidence = state.get("confidence", 0.0)
        margin = state.get("margin", 0.0)
        progress = int(state.get("progress", 1.0) * 100)

        self.lbl_key.setText(key)
        self.lbl_status.setText(state.get("message", state_type))
        self.lbl_rms.setText(f"{state.get('rms', 0.0):.1f}")
        self.lbl_threshold.setText(f"{state.get('threshold', 0.0):.1f}")
        self.lbl_confidence.setText(f"{confidence * 100:.1f}%")
        self.lbl_margin.setText(f"{margin * 100:.1f}%")
        self.lbl_top.setText(f"Top candidates: {state.get('top', '-')}")

        if state_type == "calibrating":
            self.progress.setValue(progress)
            self.progress.setFormat("Calibrating %p%")
        else:
            self.progress.setValue(100)
            self.progress.setFormat("Ready")

        if state_type == "predict":
            self.lbl_key.setStyleSheet(
                "font-size: 72px; font-weight: bold; color: #1b5e20; "
                "background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 6px;"
            )
        elif state_type in {"skip", "pending"}:
            self.lbl_key.setStyleSheet(
                "font-size: 72px; font-weight: bold; color: #e65100; "
                "background: #fff3e0; border: 1px solid #ffcc80; border-radius: 6px;"
            )
        else:
            self.lbl_key.setStyleSheet(
                "font-size: 72px; font-weight: bold; color: #0d47a1; "
                "background: #eef4ff; border: 1px solid #c7d7f2; border-radius: 6px;"
            )


def run_gui(args):
    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication([])
    engine = EMGRealTimeInference(args)
    engine.wait_for_connection()
    window = InferenceStatusWindow(engine)
    window.show(args.gui_interval_ms)
    app.exec_()


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time EMG keyboard inference.")
    parser.add_argument("--model-path", default="best_emg_model.pt")
    parser.add_argument("--model", choices=["cnn_lstm", "resnet1d"], default="cnn_lstm")
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sample-rate", type=float, default=SAMPLING_RATE_THEORETICAL)
    parser.add_argument(
        "--filter-mode",
        choices=["raw", "notch", "highpass_20", "highpass_20_notch"],
        default="highpass_20",
    )
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=80000.0)
    parser.add_argument("--auto-threshold", action="store_true", default=True)
    parser.add_argument("--manual-threshold", action="store_false", dest="auto_threshold")
    parser.add_argument("--calibration-seconds", type=float, default=3.0)
    parser.add_argument("--threshold-multiplier", type=float, default=4.0)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--vote-window", type=int, default=3)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--cooldown-samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--print-rms", action="store_true")
    parser.add_argument("--rms-log-interval", type=float, default=1.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-interval-ms", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.gui:
        run_gui(args)
    else:
        EMGRealTimeInference(args).run()
