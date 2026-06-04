import argparse
import time

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
        self.cooldown = 0
        self.last_rms_log = time.time()
        self.prediction_count = 0

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
        addr = self.receiver.wait_for_connection()
        print(f"[READY] ESP32 connected from {addr}")
        print(
            "[CONFIG] "
            f"window={self.args.window_size}, filter={self.args.filter_mode}, "
            f"threshold={self.args.threshold}, cooldown={self.args.cooldown_samples}"
        )

        while True:
            batch = self.receiver.receive_batch()
            if batch is None:
                continue

            processed_batch = self.filter.process(batch)
            self._append_buffer(processed_batch)

            recent_rms = self._recent_rms(processed_batch)
            self._maybe_log_rms(recent_rms)

            if self.cooldown > 0:
                self.cooldown = max(0, self.cooldown - BATCH_SIZE)
                continue

            if recent_rms >= self.args.threshold:
                self._predict(recent_rms)
                self.cooldown = self.args.cooldown_samples

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

    def _predict(self, recent_rms):
        window = self.data_buffer[:, -self.args.window_size:].T
        x = self._normalize_window(window)
        x_tensor = torch.from_numpy(x[None, :, :]).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        raw_id = self.idx_to_label[pred_idx]
        key_char = self.inv_keys.get(raw_id, f"Unknown({raw_id})")

        if confidence < self.args.min_confidence:
            print(
                f"[SKIP] key='{key_char}' confidence={confidence * 100:.1f}% "
                f"rms={recent_rms:.1f}"
            )
            return

        self.prediction_count += 1
        top_k = self._format_top_k(probs, self.args.top_k)
        print(
            f"[PREDICT #{self.prediction_count}] key='{key_char}' "
            f"confidence={confidence * 100:.1f}% rms={recent_rms:.1f} top={top_k}"
        )

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
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--cooldown-samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--print-rms", action="store_true")
    parser.add_argument("--rms-log-interval", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    EMGRealTimeInference(parse_args()).run()
