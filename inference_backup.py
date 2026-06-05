import argparse
import time

import numpy as np
import torch
from scipy import signal

# 외부 설정 파일(config.py) 및 통신, 모델 아키텍처 모듈 로드
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

# RIGHT_HAND_KEYS 딕셔너리의 고유한 밸류(분류 대상 키 개수)를 기반으로 총 클래스 수 정의
NUM_CLASSES = len(set(RIGHT_HAND_KEYS.values()))


class StreamingFilter:
    """
    실시간 스트리밍 데이터를 위한 필터 클래스.
    데이터가 연속으로 들어올 때 필터의 경계면 찌그러짐(과도 응답)을 막기 위해 
    직전 연산의 필터 상태(Internal State)를 유지하며 신호를 정제함.
    """
    def __init__(self, mode, fs):
        self.mode = mode  # 필터 모드 선택 ("raw", "notch", "highpass_20", "highpass_20_notch")
        self.fs = fs      # 샘플링 레이트 (Hz)
        self.filters = self._build_filters(mode, fs)  # 필터 계수(b, a) 리스트 생성
        # 각 채널별, 각 필터별 초기 내부 상태 상태(zi)를 0.0으로 초기화하여 저장
        self.states = [
            [signal.lfilter_zi(b, a) * 0.0 for b, a in self.filters]
            for _ in range(CHANNELS)
        ]

    def _build_filters(self, mode, fs):
        """선택한 모드에 맞는 디지털 필터 계수(b: 분자, a: 분모)를 생성"""
        if mode == "raw":
            return []
        if mode == "notch":
            # 60Hz 등 특정 전원 노이즈 제거용 노치 필터
            return [signal.iirnotch(NOTCH_F0, NOTCH_Q, fs=fs)]
        if mode == "highpass_20":
            # 20Hz 미만의 저주파 움직임 노이즈(Baseline Wander) 제거용 하이패스 필터
            return [signal.butter(4, 20.0, btype="highpass", fs=fs)]
        if mode == "highpass_20_notch":
            # 하이패스 필터와 노치 필터를 결합하여 순차 적용
            return [
                signal.butter(4, 20.0, btype="highpass", fs=fs),
                signal.iirnotch(NOTCH_F0, NOTCH_Q, fs=fs),
            ]
        raise ValueError(f"Unsupported filter mode: {mode}")

    def process(self, batch):
        """입력된 데이터 배치([CHANNELS, BATCH_SIZE])에 대해 채널별 연속 필터링 수행"""
        if self.mode == "raw":
            return batch.astype(np.float32)

        processed = np.empty_like(batch, dtype=np.float64)
        for ch_idx in range(CHANNELS):
            x = batch[ch_idx].astype(np.float64)
            # 해당 채널에 등록된 모든 필터 계수와 내부 상태를 순차적으로 통과시킴
            for filter_idx, (b, a) in enumerate(self.filters):
                # lfilter에 zi를 입력하고 출력 결과와 함께 업데이트된 새 상태를 반환받음 (연속성 유지 핵심)
                x, self.states[ch_idx][filter_idx] = signal.lfilter(
                    b,
                    a,
                    x,
                    zi=self.states[ch_idx][filter_idx],
                )
            processed[ch_idx] = x
        return processed.astype(np.float32)


class EMGRealTimeInference:
    """실시간으로 EMG 신호를 수신, 전처리, 임계값 검사, 딥러닝 추론을 제어하는 메인 엔진"""
    def __init__(self, args):
        self.args = args
        self.device = self._select_device(args.device)          # 추론에 사용할 하드웨어(CPU/GPU) 설정
        self.receiver = EMGReceiver(args.port)                  # 소켓 통신을 통한 데이터 수신 객체
        self.model = self._load_model(args.model_path, args.model) # 가중치 파일 기반 딥러닝 모델 로드
        self.model.eval()                                       # 모델을 평가(추론) 모드로 전환 (드롭아웃 등 비활성화)

        # 모델 출력을 실제 키보드 문자로 변환하기 위한 매핑 딕셔너리 구축
        self.raw_labels = sorted(set(RIGHT_HAND_KEYS.values()))
        self.idx_to_label = {idx: raw_id for idx, raw_id in enumerate(self.raw_labels)}
        self.inv_keys = {v: k for k, v in RIGHT_HAND_KEYS.items()}

        # 고정 크기 모델 입력을 유연하게 추출하기 위한 2차원 링 버퍼 생성 ([채널 수, 대형 버퍼 크기])
        self.data_buffer = np.zeros((CHANNELS, args.buffer_size), dtype=np.float32)
        self.filter = StreamingFilter(args.filter_mode, args.sample_rate) # 실시간 필터 세팅
        self.cooldown = 0                       # 타건 발생 후 연속 중복 예측을 막기 위한 쿨다운 샘플 카운터
        self.last_rms_log = time.time()         # RMS 로그 출력 주기 제어용 타임스탬프
        self.prediction_count = 0               # 누적 타건 감지 횟수

    def _select_device(self, requested):
        """하드웨어 장치 가속(CUDA GPU) 사용 여부 선택"""
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _load_model(self, model_path, model_name):
        """지정된 아키텍처에 맞게 모델 인스턴스를 생성하고 학습된 가중치 파라미터 주입"""
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
        """실시간 루프를 가동하여 데이터를 수신하고 조건 만족 시 추론 프로세스 트리거"""
        addr = self.receiver.wait_for_connection() # 송신단(ESP32 등) 접속 대기
        print(f"[READY] ESP32 connected from {addr}")
        print(
            "[CONFIG] "
            f"window={self.args.window_size}, filter={self.args.filter_mode}, "
            f"threshold={self.args.threshold}, cooldown={self.args.cooldown_samples}"
        )

        while True:
            batch = self.receiver.receive_batch() # 최신 데이터 배치 수신 ([CHANNELS, BATCH_SIZE])
            if batch is None:
                continue

            processed_batch = self.filter.process(batch) # 1. 실시간 필터 처리
            self._append_buffer(processed_batch)         # 2. 전처리 완료된 데이터를 링 버퍼에 추가

            recent_rms = self._recent_rms(processed_batch) # 3. 이번 배치의 실효값(에너지 수준) 계산
            self._maybe_log_rms(recent_rms)                # 옵션 플래그가 켜진 경우 주기적으로 RMS 로그 출력

            # 쿨다운 제한 장치가 작동 중인 경우, 수신한 샘플 수만큼 차감하고 추론 단계 건너뜀
            if self.cooldown > 0:
                self.cooldown = max(0, self.cooldown - BATCH_SIZE)
                continue

            # 4. 신호 에너지가 사용자가 설정한 활성화 임계값을 넘었는지 검사
            if recent_rms >= self.args.threshold:
                self._predict(recent_rms)                  # 5. 임계값 상회 시 딥러닝 추론 실행
                self.cooldown = self.args.cooldown_samples # 6. 추론 직후 고정 샘플만큼 쿨다운 작동

    def _append_buffer(self, batch):
        """링 버퍼의 데이터를 왼쪽(과거)으로 밀어내고 오른쪽 끝(현재)에 새 배치를 채워 넣음"""
        self.data_buffer = np.roll(self.data_buffer, -BATCH_SIZE, axis=1)
        self.data_buffer[:, -BATCH_SIZE:] = batch

    def _recent_rms(self, batch):
        """현재 유입된 신호 배치의 전체 제곱평균제곱근(RMS)을 구해 에너지 강도 측정"""
        return float(np.sqrt(np.mean(np.square(batch))))

    def _maybe_log_rms(self, recent_rms):
        """모니터링용 기능: 지정된 시간 간격마다 현재 신호 강도를 콘솔에 출력"""
        if not self.args.print_rms:
            return
        now = time.time()
        if now - self.last_rms_log >= self.args.rms_log_interval:
            print(f"[RMS] {recent_rms:.1f}")
            self.last_rms_log = now

    def _predict(self, recent_rms):
        """버퍼에서 최근 데이터를 잘라내어 정규화 후 딥러닝 모델 연산 수행"""
        # 버퍼 맨 우측 끝에서 모델이 요구하는 window_size 만큼 타임스텝 데이터 추출 후 축 정렬(Transpose)
        # 결과 형태: [타임스텝(window_size), 채널수(CHANNELS)]
        window = self.data_buffer[:, -self.args.window_size:].T
        x = self._normalize_window(window) # 채널별 독립 Z-score 정규화 실행
        
        # 파이토치 입력을 위해 배치 차원을 가공 추가하여 텐서로 변환: [1, window_size, CHANNELS]
        x_tensor = torch.from_numpy(x[None, :, :]).float().to(self.device)

        with torch.no_grad(): # 역전파 그라디언트 계산을 비활성화하여 메모리 절약 및 연산 가속
            logits = self.model(x_tensor)
            # 로짓 결과값에 소프트맥스를 씌워 모델이 확신하는 클래스별 확률 배열([0, 1]) 추출
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))  # 가장 확률이 높은 가중치 인덱스 선택
        confidence = float(probs[pred_idx]) # 해당 인덱스의 확률값 (신뢰도)
        raw_id = self.idx_to_label[pred_idx]
        key_char = self.inv_keys.get(raw_id, f"Unknown({raw_id})") # 인덱스 번호를 실제 영문자 키로 치환

        # 확률 모델 예측값이 지정한 최소 신뢰도 기준 미만이면 잘못된 활성화로 보고 무시(출력 스킵)
        if confidence < self.args.min_confidence:
            print(
                f"[SKIP] key='{key_char}' confidence={confidence * 100:.1f}% "
                f"rms={recent_rms:.1f}"
            )
            return

        self.prediction_count += 1
        top_k = self._format_top_k(probs, self.args.top_k) # 차순위 후보군 확률 포맷팅
        print(
            f"[PREDICT #{self.prediction_count}] key='{key_char}' "
            f"confidence={confidence * 100:.1f}% rms={recent_rms:.1f} top={top_k}"
        )

    def _normalize_window(self, window):
        """추출된 단일 윈도우 데이터 내에서 채널별 평균 0, 표준편차 1 단위 표준화 처리"""
        mean = np.mean(window, axis=0, keepdims=True)
        std = np.std(window, axis=0, keepdims=True) + 1e-8 # 0 나누기 오류 방지용 입실론 가산
        return ((window - mean) / std).astype(np.float32)

    def _format_top_k(self, probs, top_k):
        """가장 높은 확률을 보인 상위 K개의 클래스 문자 및 확률 백분율 정보 문자열 가공"""
        if top_k <= 0:
            return "-"
        indices = np.argsort(probs)[::-1][:top_k] # 내림차순 정렬 후 K개 슬라이싱
        items = []
        for idx in indices:
            raw_id = self.idx_to_label[int(idx)]
            key_char = self.inv_keys.get(raw_id, f"Unknown({raw_id})")
            items.append(f"{key_char}:{probs[idx] * 100:.1f}%")
        return ", ".join(items)


def parse_args():
    """터미널 커맨드라인 매개변수 빌더 및 기본 파라미터 프리셋 정의"""
    parser = argparse.ArgumentParser(description="Run real-time EMG keyboard inference.")
    parser.add_argument("--model-path", default="best_emg_model.pt") # 가중치 파일 경로
    parser.add_argument("--model", choices=["cnn_lstm", "resnet1d"], default="cnn_lstm") # 코어 네트워크 구조
    parser.add_argument("--port", type=int, default=SERVER_PORT) # 데이터 수신 네트워크 포트
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto") # 연산 장치
    parser.add_argument("--sample-rate", type=float, default=SAMPLING_RATE_THEORETICAL) # 수신 신호 주파수
    parser.add_argument(
        "--filter-mode",
        choices=["raw", "notch", "highpass_20", "highpass_20_notch"],
        default="highpass_20",
    ) # 적용할 실시간 스트리밍 필터 종류
    parser.add_argument("--window-size", type=int, default=200)       # 모델 입력용 시퀀스 길이
    parser.add_argument("--buffer-size", type=int, default=2000)      # 전체 내부 링버퍼 가로 길이
    parser.add_argument("--threshold", type=float, default=80000.0)   # 타건 인정을 위한 최소 배치의 RMS 임계 수치
    parser.add_argument("--min-confidence", type=float, default=0.35) # 소프트맥스 출력 하한선 필터 필터
    parser.add_argument("--cooldown-samples", type=int, default=200)  # 타건 예측 후 다음 추론을 잠글 샘플 범위
    parser.add_argument("--top-k", type=int, default=3)               # 로그에 띄워줄 상위 클래스 정보 수
    parser.add_argument("--print-rms", action="store_true")           # 실시간 RMS 모니터 덤프 활성화 여부
    parser.add_argument("--rms-log-interval", type=float, default=1.0) # RMS 콘솔 출력 시간 주기 (초 단위)
    return parser.parse_args()


if __name__ == "__main__":
    # 인자 파싱 후 곧바로 객체 생성 및 실시간 엔진 실행
    EMGRealTimeInference(parse_args()).run()