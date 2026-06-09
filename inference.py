import argparse
import csv
import os
import time
from collections import Counter, deque
from datetime import datetime

import numpy as np
import torch
from scipy import signal

from config import (
    BATCH_SIZE,
    CHANNELS,
    INFERENCE_AUTO_THRESHOLD,
    INFERENCE_BUFFER_SIZE,
    INFERENCE_CALIBRATION_SECONDS,
    INFERENCE_COOLDOWN_SAMPLES,
    INFERENCE_DEVICE,
    INFERENCE_FILTER_MODE,
    INFERENCE_GUI,
    INFERENCE_GUI_INTERVAL_MS,
    INFERENCE_GUI_KEY_HOLD_MS,
    INFERENCE_ACTIVE_CALIBRATION_SECONDS,
    INFERENCE_CALIBRATION_MODE,
    INFERENCE_CALIBRATION_ONLY,
    INFERENCE_LOG_ALL_STATES,
    INFERENCE_LOG_DIR,
    INFERENCE_LOG_ENABLED,
    INFERENCE_MIN_CONFIDENCE,
    INFERENCE_MIN_MARGIN,
    INFERENCE_MIN_VOTES,
    INFERENCE_MODEL_PATH,
    INFERENCE_MODEL_TYPE,
    INFERENCE_PRINT_RMS,
    INFERENCE_REPLAY_CSV,
    INFERENCE_REPLAY_LOOP,
    INFERENCE_REPLAY_REALTIME,
    INFERENCE_REPLAY_SPEED,
    INFERENCE_REPLAY_AUTO_THRESHOLD,
    INFERENCE_REPLAY_THRESHOLD_PERCENTILE,
    INFERENCE_RMS_LOG_INTERVAL,
    INFERENCE_THRESHOLD,
    INFERENCE_THRESHOLD_MULTIPLIER,
    INFERENCE_TOP_K,
    INFERENCE_VOTE_WINDOW,
    INFERENCE_WINDOW_SIZE,
    NOTCH_F0,
    NOTCH_Q,
    RIGHT_HAND_KEYS,
    SAMPLING_RATE_THEORETICAL,
    SERVER_PORT,
)
from network import EMGReceiver
from models import CNNLSTM, ResNet1D


NUM_CLASSES = len(set(RIGHT_HAND_KEYS.values()))


class ReplayEMGReceiver:
    """CSV 파일을 센서 입력처럼 batch 단위로 재생하는 입력 소스.

    network.py를 거치지 않고 inference.py 내부에서 직접 데이터를 공급합니다.
    모델/threshold/smoothing/GUI/logging을 빠르게 확인할 때 사용합니다.
    """

    def __init__(self, csv_path, sample_rate, speed=1.0, loop=False, realtime=True):
        self.csv_path = csv_path
        self.sample_rate = sample_rate
        self.speed = max(0.01, speed)
        self.loop = loop
        self.realtime = realtime
        self.cursor = 0
        self.last_emit = None
        self.samples = self._load_samples(csv_path)
        self.events = self._load_events(csv_path)

    def _load_samples(self, csv_path):
        data = np.loadtxt(
            csv_path,
            delimiter=",",
            skiprows=1,
            usecols=range(CHANNELS),
            dtype=np.float32,
        )
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != CHANNELS:
            raise ValueError(f"Replay CSV must contain CH0..CH{CHANNELS - 1} columns.")
        return data

    def _load_events(self, csv_path):
        try:
            events = np.loadtxt(
                csv_path,
                delimiter=",",
                skiprows=1,
                usecols=CHANNELS,
                dtype=np.float32,
            )
        except (IndexError, ValueError):
            return None
        return np.atleast_1d(events)

    def wait_for_connection(self):
        print(f"[REPLAY] loaded {len(self.samples)} samples from {self.csv_path}")
        return ("replay", os.path.abspath(self.csv_path))

    def receive_batch(self):
        if self.cursor + BATCH_SIZE > len(self.samples):
            if not self.loop:
                raise EOFError("Replay finished.")
            self.cursor = 0
            self.last_emit = None

        if self.realtime:
            # 실제 샘플링 속도를 흉내 내기 위해 batch 간격만큼 대기합니다.
            now = time.time()
            if self.last_emit is not None:
                target_interval = (BATCH_SIZE / self.sample_rate) / self.speed
                elapsed = now - self.last_emit
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
            self.last_emit = time.time()

        # EMGReceiver.receive_batch()와 동일하게 (channel, batch) 형태로 반환합니다.
        batch = self.samples[self.cursor : self.cursor + BATCH_SIZE].T
        self.cursor += BATCH_SIZE
        return batch

    def close(self):
        """실시간 TCP 수신기와 동일한 종료 인터페이스를 맞추기 위한 no-op."""
        return None


class InferenceLogger:
    """실시간 추론 상태를 CSV로 남기는 로거.

    predict만 남기면 오탐/미탐 원인 분석이 어렵기 때문에 기본값은 idle,
    skip, pending, cooldown까지 모두 저장하도록 되어 있습니다.
    """

    def __init__(self, enabled, log_dir, log_all_states, source_name):
        self.enabled = enabled
        self.log_all_states = log_all_states
        self.started_at = time.time()
        self.path = None
        self.file = None
        self.writer = None
        if not enabled:
            return

        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_source = source_name.replace(os.sep, "_").replace(":", "_")
        self.path = os.path.join(log_dir, f"inference_{timestamp}_{safe_source}.csv")
        self.file = open(self.path, mode="w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "timestamp",
                "elapsed_sec",
                "source",
                "state",
                "message",
                "rms",
                "threshold",
                "progress",
                "key",
                "raw_id",
                "confidence",
                "margin",
                "top",
                "emitted_count",
                "cooldown_remaining",
            ],
        )
        self.writer.writeheader()
        print(f"[LOG] writing inference log to {self.path}")

    def log(self, state):
        if not self.enabled or self.writer is None:
            return
        state_type = state.get("type", "")
        if not self.log_all_states and state_type not in {"predict", "skip", "pending", "end"}:
            return
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_sec": f"{time.time() - self.started_at:.3f}",
            "source": state.get("source", ""),
            "state": state_type,
            "message": state.get("message", ""),
            "rms": f"{state.get('rms', 0.0):.3f}",
            "threshold": f"{state.get('threshold', 0.0):.3f}",
            "progress": f"{state.get('progress', 0.0):.3f}",
            "key": state.get("key", ""),
            "raw_id": state.get("raw_id", ""),
            "confidence": f"{state.get('confidence', 0.0):.6f}",
            "margin": f"{state.get('margin', 0.0):.6f}",
            "top": state.get("top", ""),
            "emitted_count": state.get("count", ""),
            "cooldown_remaining": state.get("cooldown", ""),
        }
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


class ThresholdCalibrator:
    """초기 안정 상태 RMS로 trigger threshold를 자동 계산합니다."""

    def __init__(self, enabled, duration_sec, multiplier, fallback_threshold):
        self.enabled = enabled and duration_sec > 0
        self.duration_sec = duration_sec
        self.multiplier = multiplier
        self.fallback_threshold = fallback_threshold
        self.started_at = None
        self.samples = []
        self.threshold = fallback_threshold
        self.done = not self.enabled
        self.stats = None

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
            self.stats = {
                "mean": mean,
                "std": std,
                "min": float(np.min(self.samples)),
                "max": float(np.max(self.samples)),
            }
            # 안정 상태 평균에서 표준편차의 multiplier배만큼 떨어진 값을 입력 감지 기준으로 사용합니다.
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
    """단일 모델 출력이 바로 키 입력으로 나가지 않도록 안정화합니다.

    confidence, top1-top2 margin, 최근 vote window를 모두 통과해야 최종 예측을
    출력합니다. 실시간 키 입력에서는 중복/오탐을 줄이는 역할이 큽니다.
    """

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
    """실시간 batch에 순차적으로 적용하는 streaming filter.

    train.py의 offline filter와 달리, 실시간에서는 이전 batch의 filter state를
    보존해야 신호가 끊기지 않습니다.
    """

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
                # lfilter 상태를 채널/필터별로 유지해서 batch 경계의 왜곡을 줄입니다.
                x, self.states[ch_idx][filter_idx] = signal.lfilter(
                    b,
                    a,
                    x,
                    zi=self.states[ch_idx][filter_idx],
                )
            processed[ch_idx] = x
        return processed.astype(np.float32)


class EMGRealTimeInference:
    """실시간 EMG 수신부터 예측 출력까지 담당하는 메인 엔진."""

    def __init__(self, args):
        self.args = args
        self.device = self._select_device(args.device)
        if args.replay_csv:
            # replay_csv가 있으면 TCP 대신 CSV 입력을 사용합니다.
            self.receiver = ReplayEMGReceiver(
                csv_path=args.replay_csv,
                sample_rate=args.sample_rate,
                speed=args.replay_speed,
                loop=args.replay_loop,
                realtime=args.replay_realtime,
            )
            self.source_name = "replay"
        else:
            # 실제 센서 또는 esp32-simulator.py는 이 TCP receiver로 들어옵니다.
            self.receiver = EMGReceiver(args.port)
            self.source_name = "tcp"
        self.model = self._load_model(args.model_path, args.model)
        self.model.eval()

        self.raw_labels = sorted(set(RIGHT_HAND_KEYS.values()))
        self.idx_to_label = {idx: raw_id for idx, raw_id in enumerate(self.raw_labels)}
        self.inv_keys = {v: k for k, v in RIGHT_HAND_KEYS.items()}

        # raw_buffer는 main.py의 RMS 표시 방식과 맞춘 trigger/calibration 계산에 사용합니다.
        # data_buffer는 모델 입력용으로 filter가 적용된 신호를 유지합니다.
        self.raw_buffer = np.zeros((CHANNELS, args.buffer_size), dtype=np.float32)
        self.data_buffer = np.zeros((CHANNELS, args.buffer_size), dtype=np.float32)
        self.buffered_samples = 0
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
        self.emitted_keys = deque(maxlen=20)
        self.done = False
        self.active_calibration_started_at = None
        self.active_calibration_seconds = max(0.001, args.active_calibration_seconds)
        self.active_rms_samples = []
        self.active_candidates = []
        self.active_calibration_done = not args.calibration_mode
        self.logger = InferenceLogger(
            enabled=args.log,
            log_dir=args.log_dir,
            log_all_states=args.log_all_states,
            source_name=self.source_name,
        )
        self.last_status = {
            "type": "boot",
            "message": "model loaded",
            "rms": 0.0,
            "threshold": self.threshold,
            "top": "-",
            "source": self.source_name,
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
        try:
            self.wait_for_connection()
            while not self.done:
                self.process_once()
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user.")
        finally:
            self.receiver.close()
            self.logger.close()

    def wait_for_connection(self):
        self._prepare_replay_threshold()
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
            f"threshold={self.threshold:.1f}, cooldown={self.args.cooldown_samples}, "
            f"votes={self.args.min_votes}/{self.args.vote_window}, "
            f"log={self.args.log}, calibration_mode={self.args.calibration_mode}"
        )
        return self.connected_addr

    def _prepare_replay_threshold(self):
        if not self.args.replay_csv or not self.args.replay_auto_threshold:
            return

        threshold, stats = self._estimate_replay_threshold()
        if threshold is None:
            print(
                "[REPLAY THRESHOLD] failed to estimate from dataset; "
                f"using fallback threshold={self.threshold:.1f}"
            )
            return

        self.threshold = threshold
        self.calibrator.threshold = threshold
        self.calibrator.done = True
        self.calibrator.stats = stats
        self.args.auto_threshold = False

        if stats["method"] == "event_label":
            print(
                "[REPLAY THRESHOLD] "
                f"threshold={threshold:.1f}, balanced_acc={stats['balanced_acc']:.3f}, "
                f"idle_batches={stats['idle_count']}, active_batches={stats['active_count']}"
            )
        else:
            print(
                "[REPLAY THRESHOLD] "
                f"threshold={threshold:.1f} from p{self.args.replay_threshold_percentile:.1f} "
                f"RMS percentile; event labels unavailable"
            )

    def _estimate_replay_threshold(self):
        samples = getattr(self.receiver, "samples", None)
        if samples is None or len(samples) < BATCH_SIZE:
            return None, None

        temp_raw_buffer = np.zeros((CHANNELS, self.args.buffer_size), dtype=np.float32)
        buffered_samples = 0
        rms_values = []
        active_flags = []
        events = getattr(self.receiver, "events", None)

        usable = (len(samples) // BATCH_SIZE) * BATCH_SIZE
        for start in range(0, usable, BATCH_SIZE):
            batch = samples[start : start + BATCH_SIZE].T
            temp_raw_buffer = np.roll(temp_raw_buffer, -BATCH_SIZE, axis=1)
            temp_raw_buffer[:, -BATCH_SIZE:] = batch
            buffered_samples = min(buffered_samples + BATCH_SIZE, self.args.buffer_size)
            rms_values.append(self._activity_rms_from_buffer(temp_raw_buffer, buffered_samples))

            if events is not None and len(events) >= start + BATCH_SIZE:
                event_batch = events[start : start + BATCH_SIZE]
                active_flags.append(bool(np.any(event_batch != 0)))

        if not rms_values:
            return None, None

        rms = np.array(rms_values, dtype=np.float32)
        stats = {
            "mean": float(np.mean(rms)),
            "std": float(np.std(rms)),
            "min": float(np.min(rms)),
            "max": float(np.max(rms)),
        }

        if active_flags and any(active_flags) and not all(active_flags):
            threshold, score = self._find_best_binary_threshold(rms, np.array(active_flags, dtype=bool))
            idle_count = int(np.sum(~np.array(active_flags, dtype=bool)))
            active_count = int(np.sum(np.array(active_flags, dtype=bool)))
            stats.update(
                {
                    "method": "event_label",
                    "balanced_acc": score,
                    "idle_count": idle_count,
                    "active_count": active_count,
                }
            )
            return threshold, stats

        percentile = min(100.0, max(0.0, self.args.replay_threshold_percentile))
        threshold = float(np.percentile(rms, percentile))
        stats.update({"method": "percentile"})
        return threshold, stats

    def _find_best_binary_threshold(self, rms, active):
        sorted_rms = np.unique(np.sort(rms))
        if len(sorted_rms) == 1:
            return float(sorted_rms[0]), 0.5

        candidates = (sorted_rms[:-1] + sorted_rms[1:]) / 2.0
        candidates = np.concatenate(([sorted_rms[0] - 1.0], candidates, [sorted_rms[-1] + 1.0]))

        best_threshold = float(candidates[0])
        best_score = -1.0
        for threshold in candidates:
            predicted_active = rms >= threshold
            tp = np.sum(predicted_active & active)
            tn = np.sum(~predicted_active & ~active)
            fp = np.sum(predicted_active & ~active)
            fn = np.sum(~predicted_active & active)
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            tnr = tn / (tn + fp) if (tn + fp) else 0.0
            score = float((tpr + tnr) / 2.0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return best_threshold, best_score

    def process_once(self):
        # GUI timer와 CLI loop가 공통으로 호출하는 한 batch 처리 단위입니다.
        try:
            batch = self.receiver.receive_batch()
        except EOFError:
            self.done = True
            return self._set_status(
                {
                    "type": "end",
                    "message": "replay finished",
                    "rms": 0.0,
                    "threshold": self.threshold,
                    "top": "-",
                }
            )

        if batch is None:
            return None

        self._append_raw_buffer(batch)
        processed_batch = self.filter.process(batch)
        self._append_buffer(processed_batch)

        recent_rms = self._activity_rms()
        self._maybe_log_rms(recent_rms)
        self.threshold = self.calibrator.update(recent_rms)

        if not self.calibrator.done:
            return self._set_status(
                {
                    "type": "calibrating",
                    "message": "idle threshold calibration",
                    "rms": recent_rms,
                    "threshold": self.threshold,
                    "progress": self.calibrator.progress(),
                    "top": "-",
                }
            )

        if not self.active_calibration_done:
            return self._process_active_calibration(recent_rms)

        if self.cooldown > 0:
            self.cooldown = max(0, self.cooldown - BATCH_SIZE)
            return self._set_status(
                {
                    "type": "cooldown",
                    "message": f"cooldown {self.cooldown} samples",
                    "rms": recent_rms,
                    "threshold": self.threshold,
                    "top": "-",
                }
            )

        if recent_rms < self.threshold:
            return self._set_status(
                {
                    "type": "idle",
                    "message": "threshold not reached",
                    "rms": recent_rms,
                    "threshold": self.threshold,
                    "top": "-",
                }
            )

        # threshold를 넘은 batch만 모델에 넣고, smoother가 최종 출력 여부를 결정합니다.
        candidate = self._classify(recent_rms)
        decision = self.smoother.update(candidate)
        return self._set_status(self._handle_decision(decision, recent_rms))

    def _set_status(self, state):
        state.setdefault("source", self.source_name)
        state.setdefault("threshold", self.threshold)
        state.setdefault("cooldown", self.cooldown)
        state.setdefault("emitted_history", " ".join(self.emitted_keys))
        state.setdefault("last_emitted_key", self.emitted_keys[-1] if self.emitted_keys else "")
        self.last_status = state
        self.logger.log(state)
        return state

    def _append_buffer(self, batch):
        self.data_buffer = np.roll(self.data_buffer, -BATCH_SIZE, axis=1)
        self.data_buffer[:, -BATCH_SIZE:] = batch

    def _append_raw_buffer(self, batch):
        self.raw_buffer = np.roll(self.raw_buffer, -BATCH_SIZE, axis=1)
        self.raw_buffer[:, -BATCH_SIZE:] = batch
        self.buffered_samples = min(self.buffered_samples + BATCH_SIZE, self.args.buffer_size)

    def _recent_rms(self, batch):
        return float(np.sqrt(np.mean(np.square(batch))))

    def _activity_rms(self):
        return self._activity_rms_from_buffer(self.raw_buffer, self.buffered_samples)

    def _activity_channel_rms(self):
        return self._activity_channel_rms_from_buffer(self.raw_buffer, self.buffered_samples)

    def _activity_rms_from_buffer(self, buffer, buffered_samples):
        channel_rms = self._activity_channel_rms_from_buffer(buffer, buffered_samples)
        if len(channel_rms) == 0:
            return 0.0
        # main.py는 채널별 RMS bar를 보여주므로, 실시간 trigger는 가장 강한 채널을 기준으로 잡습니다.
        return float(np.max(channel_rms))

    def _activity_channel_rms_from_buffer(self, buffer, buffered_samples):
        valid_len = min(buffered_samples, buffer.shape[1])
        if valid_len <= 0:
            return np.array([], dtype=np.float32)

        valid = buffer[:, -valid_len:]
        baseline = np.mean(valid, axis=1, keepdims=True)
        centered = valid - baseline
        rms_len = min(100, valid_len)
        recent = centered[:, -rms_len:]
        return np.sqrt(np.mean(np.square(recent), axis=1))

    def _maybe_log_rms(self, recent_rms):
        if not self.args.print_rms:
            return
        now = time.time()
        if now - self.last_rms_log >= self.args.rms_log_interval:
            channel_rms = self._activity_channel_rms()
            channel_text = ", ".join(f"{value:.0f}" for value in channel_rms)
            print(f"[RMS] max={recent_rms:.1f} ch=[{channel_text}]")
            self.last_rms_log = now

    def _process_active_calibration(self, recent_rms):
        # idle threshold 보정 이후 실제 키 입력을 받아 추천 confidence/margin을 계산합니다.
        now = time.time()
        if self.active_calibration_started_at is None:
            self.active_calibration_started_at = now
            print(
                "[CALIBRATION] Press several target keys naturally for "
                f"{self.active_calibration_seconds:.1f}s."
            )

        elapsed = now - self.active_calibration_started_at
        self.active_rms_samples.append(recent_rms)

        candidate = None
        if recent_rms >= self.threshold:
            candidate = self._classify(recent_rms)
            self.active_candidates.append(candidate)

        if elapsed >= self.active_calibration_seconds:
            self.active_calibration_done = True
            self._print_calibration_report()
            if self.args.calibration_only:
                self.done = True
                return self._set_status(
                    {
                        "type": "end",
                        "message": "calibration finished",
                        "rms": recent_rms,
                        "threshold": self.threshold,
                        "top": candidate["top"] if candidate else "-",
                    }
                )
            print("[CALIBRATION] Finished. Switching to normal inference.")

        state = {
            "type": "active_calibrating",
            "message": "active calibration",
            "rms": recent_rms,
            "threshold": self.threshold,
            "progress": min(1.0, elapsed / self.active_calibration_seconds),
            "top": candidate["top"] if candidate else "-",
        }
        if candidate:
            state.update(candidate)
        return self._set_status(state)

    def _print_calibration_report(self):
        idle = self.calibrator.stats or {}
        active = np.array(self.active_rms_samples, dtype=np.float32)
        triggered = [item for item in self.active_candidates if item["rms"] >= self.threshold]
        active_mean = float(np.mean(active)) if len(active) else 0.0
        active_min = float(np.min(active)) if len(active) else 0.0
        active_max = float(np.max(active)) if len(active) else 0.0

        if idle and len(active):
            recommended_threshold = (idle.get("max", self.threshold) + active_max) / 2.0
        else:
            recommended_threshold = self.threshold

        confidences = [item["confidence"] for item in triggered]
        margins = [item["margin"] for item in triggered]
        recommended_confidence = max(0.20, min(0.70, float(np.mean(confidences) * 0.65))) if confidences else self.args.min_confidence
        recommended_margin = max(0.05, min(0.30, float(np.mean(margins) * 0.65))) if margins else self.args.min_margin

        print("[CALIBRATION REPORT]")
        if idle:
            print(
                "  idle_rms: "
                f"mean={idle['mean']:.1f}, std={idle['std']:.1f}, "
                f"min={idle['min']:.1f}, max={idle['max']:.1f}"
            )
        print(
            "  active_rms: "
            f"mean={active_mean:.1f}, min={active_min:.1f}, max={active_max:.1f}"
        )
        print(
            "  triggered_predictions: "
            f"{len(triggered)} / {len(self.active_rms_samples)} batches"
        )
        print(
            "  recommended: "
            f"threshold={recommended_threshold:.1f}, "
            f"min_confidence={recommended_confidence:.2f}, "
            f"min_margin={recommended_margin:.2f}"
        )

    def _classify(self, recent_rms):
        # 학습 때와 동일하게 (time, channel) window를 만들고 channel별 표준화 후 모델에 입력합니다.
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
        self.emitted_keys.append(candidate["key"])
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
        self.widget.setWindowTitle("EMG Keyboard Training Interface")
        self.widget.resize(780, 680)

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
        self.lbl_state = self._make_value_label()
        self.lbl_source = self._make_value_label()
        self.lbl_count = self._make_value_label()
        self.lbl_cooldown = self._make_value_label()

        self._add_metric(grid, 0, "RMS", self.lbl_rms)
        self._add_metric(grid, 1, "Threshold", self.lbl_threshold)
        self._add_metric(grid, 2, "Confidence", self.lbl_confidence)
        self._add_metric(grid, 3, "Margin", self.lbl_margin)
        self._add_metric(grid, 4, "State", self.lbl_state)
        self._add_metric(grid, 5, "Source", self.lbl_source)
        self._add_metric(grid, 6, "Predictions", self.lbl_count)
        self._add_metric(grid, 7, "Cooldown", self.lbl_cooldown)

        self.lbl_top = QtWidgets.QLabel("-")
        self.lbl_top.setWordWrap(True)
        self.lbl_top.setStyleSheet("font-size: 15px; color: #333;")
        root.addWidget(self.lbl_top)

        self.lbl_history = QtWidgets.QLabel("Emitted keys: -")
        self.lbl_history.setWordWrap(True)
        self.lbl_history.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #263238; "
            "background: #fafafa; border: 1px solid #d6dde2; border-radius: 6px; padding: 8px;"
        )
        root.addWidget(self.lbl_history)

        self.key_hold_seconds = max(0.0, engine.args.gui_key_hold_ms / 1000.0)
        self.held_key = "-"
        self.held_until = 0.0

        self.keyboard = VirtualKeyboardWidget(engine.args.gui_key_hold_ms)
        root.addWidget(self.keyboard.widget)

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
        now = time.time()
        current_key = state.get("key", "-") if state_type in {"predict", "pending", "skip"} else "-"
        confidence = state.get("confidence", 0.0)
        margin = state.get("margin", 0.0)
        progress = int(state.get("progress", 1.0) * 100)

        if state_type == "predict":
            self.held_key = current_key
            self.held_until = now + self.key_hold_seconds
            display_key = current_key
            display_state = "predict"
        elif now < self.held_until:
            display_key = self.held_key
            display_state = "predict"
        else:
            display_key = current_key
            display_state = state_type

        self.lbl_key.setText(display_key)
        self.lbl_status.setText(state.get("message", state_type))
        self.lbl_rms.setText(f"{state.get('rms', 0.0):.1f}")
        self.lbl_threshold.setText(f"{state.get('threshold', 0.0):.1f}")
        self.lbl_confidence.setText(f"{confidence * 100:.1f}%")
        self.lbl_margin.setText(f"{margin * 100:.1f}%")
        self.lbl_state.setText(state_type)
        self.lbl_source.setText(state.get("source", "-"))
        self.lbl_count.setText(str(state.get("count", self.engine.prediction_count)))
        self.lbl_cooldown.setText(str(state.get("cooldown", 0)))
        self.lbl_top.setText(f"Top candidates: {state.get('top', '-')}")
        self.lbl_history.setText(f"Emitted keys: {state.get('emitted_history') or '-'}")
        self.keyboard.update_state(state)

        if state_type in {"calibrating", "active_calibrating"}:
            self.progress.setValue(progress)
            self.progress.setFormat("Calibrating %p%")
        elif state_type == "end":
            self.progress.setValue(100)
            self.progress.setFormat("Finished")
            self.timer.stop()
            self.engine.logger.close()
        else:
            self.progress.setValue(100)
            self.progress.setFormat("Ready")

        if display_state == "predict":
            self.lbl_key.setStyleSheet(
                "font-size: 72px; font-weight: bold; color: #1b5e20; "
                "background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 6px;"
            )
        elif display_state in {"skip", "pending"}:
            self.lbl_key.setStyleSheet(
                "font-size: 72px; font-weight: bold; color: #e65100; "
                "background: #fff3e0; border: 1px solid #ffcc80; border-radius: 6px;"
            )
        else:
            self.lbl_key.setStyleSheet(
                "font-size: 72px; font-weight: bold; color: #0d47a1; "
                "background: #eef4ff; border: 1px solid #c7d7f2; border-radius: 6px;"
            )


class VirtualKeyboardWidget:
    """예측된 키 위치를 화면 키보드에 표시하는 훈련용 피드백 패널."""

    KEY_ROWS = [
        ["y", "u", "i", "o", "p", "[", "]", "\\"],
        ["h", "j", "k", "l", ";", "'"],
        ["n", "m", ",", ".", "/"],
    ]

    def __init__(self, key_hold_ms=1200):
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.QtCore = QtCore
        self.buttons = {}
        self.key_hold_seconds = max(0.0, key_hold_ms / 1000.0)
        self.held_key = None
        self.held_until = 0.0
        self.widget = QtWidgets.QGroupBox("Virtual Keyboard Feedback")
        self.widget.setStyleSheet(
            "QGroupBox { font-size: 15px; font-weight: bold; color: #263238; }"
        )

        root = QtWidgets.QVBoxLayout(self.widget)
        root.setSpacing(8)

        self.lbl_hint = QtWidgets.QLabel(
            "Recognized sEMG candidates are highlighted on the keyboard layout."
        )
        self.lbl_hint.setStyleSheet("font-size: 13px; color: #546e7a;")
        root.addWidget(self.lbl_hint)

        for row_idx, row_keys in enumerate(self.KEY_ROWS):
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(6)
            row.setContentsMargins(row_idx * 28, 0, 0, 0)
            for key in row_keys:
                key_label = QtWidgets.QLabel(self._display_key(key))
                key_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                key_label.setFixedSize(72, 48)
                key_label.setStyleSheet(self._style("default"))
                row.addWidget(key_label)
                self.buttons[key] = key_label
            row.addStretch(1)
            root.addLayout(row)

        self.lbl_feedback = QtWidgets.QLabel("Waiting for EMG input")
        self.lbl_feedback.setStyleSheet("font-size: 14px; color: #37474f;")
        root.addWidget(self.lbl_feedback)

        legend = QtWidgets.QHBoxLayout()
        legend.setSpacing(8)
        for label, state in [
            ("Emitted", "predict"),
            ("Top candidate", "top1"),
            ("Pending", "pending"),
            ("Skipped", "skip"),
            ("Calibration", "active_calibrating"),
        ]:
            legend.addWidget(self._make_legend_chip(label, state))
        legend.addStretch(1)
        root.addLayout(legend)

    def _make_legend_chip(self, label, state):
        from pyqtgraph.Qt import QtWidgets

        chip = QtWidgets.QLabel(label)
        chip.setAlignment(self.QtCore.Qt.AlignmentFlag.AlignCenter)
        chip.setFixedHeight(26)
        chip.setMinimumWidth(94)
        chip.setStyleSheet(
            self._style(state).replace("font-size: 22px;", "font-size: 12px;")
        )
        return chip

    def update_state(self, state):
        self._reset()
        now = time.time()
        state_type = state.get("type", "idle")
        key = state.get("key")
        top_candidates = self._parse_top_candidates(state.get("top", ""))

        if state_type == "predict" and key:
            self.held_key = key
            self.held_until = now + self.key_hold_seconds

        for rank, candidate_key in enumerate(top_candidates):
            if candidate_key in self.buttons:
                self.buttons[candidate_key].setStyleSheet(self._style(f"top{rank + 1}"))

        if now < self.held_until and self.held_key in self.buttons:
            self.buttons[self.held_key].setStyleSheet(self._style("predict"))

        if key in self.buttons and state_type in {"predict", "pending", "skip", "active_calibrating"}:
            self.buttons[key].setStyleSheet(self._style(state_type))

        if state_type == "predict":
            self.lbl_feedback.setText(
                f"Emitted key: {self._display_key(key)} "
                f"({state.get('confidence', 0.0) * 100:.1f}%)"
            )
        elif state_type in {"pending", "skip", "active_calibrating"} and key:
            self.lbl_feedback.setText(
                f"Current candidate: {self._display_key(key)} "
                f"({state.get('confidence', 0.0) * 100:.1f}%)"
            )
        elif state_type == "calibrating":
            self.lbl_feedback.setText("Keep hand relaxed for threshold calibration")
        elif state_type == "cooldown":
            if now < self.held_until and self.held_key:
                self.lbl_feedback.setText(f"Last emitted key: {self._display_key(self.held_key)}")
            else:
                self.lbl_feedback.setText("Cooldown: holding output to prevent duplicates")
        elif state_type == "end":
            self.lbl_feedback.setText("Replay or calibration finished")
        else:
            self.lbl_feedback.setText("Waiting for EMG input")

    def _reset(self):
        for key_label in self.buttons.values():
            key_label.setStyleSheet(self._style("default"))

    def _parse_top_candidates(self, top_text):
        if not top_text or top_text == "-":
            return []
        keys = []
        for item in top_text.split(","):
            key = item.split(":", 1)[0].strip()
            if key:
                keys.append(key)
        return keys[:3]

    def _display_key(self, key):
        if key == "\\":
            return "\\"
        if key == " ":
            return "Space"
        return str(key)

    def _style(self, state):
        base = (
            "font-size: 22px; font-weight: bold; border-radius: 6px; "
            "border: 1px solid {border}; color: {color}; background: {bg};"
        )
        colors = {
            "default": {"bg": "#fafafa", "border": "#cfd8dc", "color": "#263238"},
            "top1": {"bg": "#e3f2fd", "border": "#64b5f6", "color": "#0d47a1"},
            "top2": {"bg": "#eef7ff", "border": "#90caf9", "color": "#1565c0"},
            "top3": {"bg": "#f7fbff", "border": "#bbdefb", "color": "#1976d2"},
            "predict": {"bg": "#c8e6c9", "border": "#43a047", "color": "#1b5e20"},
            "pending": {"bg": "#fff3e0", "border": "#fb8c00", "color": "#e65100"},
            "skip": {"bg": "#ffebee", "border": "#e57373", "color": "#b71c1c"},
            "active_calibrating": {"bg": "#ede7f6", "border": "#7e57c2", "color": "#4527a0"},
        }
        return base.format(**colors.get(state, colors["default"]))


def run_gui(args):
    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication([])
    engine = EMGRealTimeInference(args)
    try:
        engine.wait_for_connection()
        window = InferenceStatusWindow(engine)
        app.aboutToQuit.connect(engine.logger.close)
        window.show(args.gui_interval_ms)
        if hasattr(app, "exec"):
            app.exec()
        else:
            app.exec_()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
    finally:
        engine.receiver.close()
        engine.logger.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time EMG keyboard inference.")
    parser.add_argument("--model-path", default=INFERENCE_MODEL_PATH)
    parser.add_argument("--model", choices=["cnn_lstm", "resnet1d"], default=INFERENCE_MODEL_TYPE)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=INFERENCE_DEVICE)
    parser.add_argument("--sample-rate", type=float, default=SAMPLING_RATE_THEORETICAL)
    parser.add_argument(
        "--filter-mode",
        choices=["raw", "notch", "highpass_20", "highpass_20_notch"],
        default=INFERENCE_FILTER_MODE,
    )
    parser.add_argument("--window-size", type=int, default=INFERENCE_WINDOW_SIZE)
    parser.add_argument("--buffer-size", type=int, default=INFERENCE_BUFFER_SIZE)
    parser.add_argument("--threshold", type=float, default=INFERENCE_THRESHOLD)
    parser.add_argument("--auto-threshold", action="store_true", default=INFERENCE_AUTO_THRESHOLD)
    parser.add_argument("--manual-threshold", action="store_false", dest="auto_threshold")
    parser.add_argument("--calibration-seconds", type=float, default=INFERENCE_CALIBRATION_SECONDS)
    parser.add_argument("--threshold-multiplier", type=float, default=INFERENCE_THRESHOLD_MULTIPLIER)
    parser.add_argument("--min-confidence", type=float, default=INFERENCE_MIN_CONFIDENCE)
    parser.add_argument("--min-margin", type=float, default=INFERENCE_MIN_MARGIN)
    parser.add_argument("--vote-window", type=int, default=INFERENCE_VOTE_WINDOW)
    parser.add_argument("--min-votes", type=int, default=INFERENCE_MIN_VOTES)
    parser.add_argument("--cooldown-samples", type=int, default=INFERENCE_COOLDOWN_SAMPLES)
    parser.add_argument("--top-k", type=int, default=INFERENCE_TOP_K)
    parser.add_argument("--print-rms", action="store_true", default=INFERENCE_PRINT_RMS)
    parser.add_argument("--rms-log-interval", type=float, default=INFERENCE_RMS_LOG_INTERVAL)
    parser.add_argument("--gui", action="store_true", default=INFERENCE_GUI)
    parser.add_argument("--no-gui", action="store_false", dest="gui")
    parser.add_argument("--gui-interval-ms", type=int, default=INFERENCE_GUI_INTERVAL_MS)
    parser.add_argument("--gui-key-hold-ms", type=int, default=INFERENCE_GUI_KEY_HOLD_MS)
    parser.add_argument("--log", action="store_true", default=INFERENCE_LOG_ENABLED)
    parser.add_argument("--no-log", action="store_false", dest="log")
    parser.add_argument("--log-dir", default=INFERENCE_LOG_DIR)
    parser.add_argument("--log-all-states", action="store_true", default=INFERENCE_LOG_ALL_STATES)
    parser.add_argument("--log-events-only", action="store_false", dest="log_all_states")
    parser.add_argument("--calibration-mode", action="store_true", default=INFERENCE_CALIBRATION_MODE)
    parser.add_argument("--no-calibration-mode", action="store_false", dest="calibration_mode")
    parser.add_argument("--calibration-only", action="store_true", default=INFERENCE_CALIBRATION_ONLY)
    parser.add_argument(
        "--active-calibration-seconds",
        type=float,
        default=INFERENCE_ACTIVE_CALIBRATION_SECONDS,
    )
    parser.add_argument("--replay-csv", default=INFERENCE_REPLAY_CSV)
    parser.add_argument("--replay-loop", action="store_true", default=INFERENCE_REPLAY_LOOP)
    parser.add_argument("--no-replay-loop", action="store_false", dest="replay_loop")
    parser.add_argument("--replay-realtime", action="store_true", default=INFERENCE_REPLAY_REALTIME)
    parser.add_argument("--no-replay-realtime", action="store_false", dest="replay_realtime")
    parser.add_argument("--replay-speed", type=float, default=INFERENCE_REPLAY_SPEED)
    parser.add_argument("--replay-auto-threshold", action="store_true", default=INFERENCE_REPLAY_AUTO_THRESHOLD)
    parser.add_argument("--no-replay-auto-threshold", action="store_false", dest="replay_auto_threshold")
    parser.add_argument("--replay-threshold-percentile", type=float, default=INFERENCE_REPLAY_THRESHOLD_PERCENTILE)
    args = parser.parse_args()
    if args.buffer_size < args.window_size:
        parser.error("--buffer-size must be greater than or equal to --window-size")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.gui:
        run_gui(args)
    else:
        EMGRealTimeInference(args).run()
