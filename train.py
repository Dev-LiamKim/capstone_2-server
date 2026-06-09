import os
import glob
import itertools
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from config import RIGHT_HAND_KEYS, CHANNELS

# ==============================================================================
# 1. 하이퍼파라미터 및 환경 설정
# ==============================================================================
WINDOW_SIZE  = 100
PRE_EVENT    = 40
POST_EVENT   = WINDOW_SIZE - PRE_EVENT
NUM_CHANNELS = CHANNELS
BATCH_SIZE   = 64   # GPU 병렬 처리 효율을 위해 기존 32에서 증가

raw_labels   = sorted(list(set(RIGHT_HAND_KEYS.values())))
label_to_idx = {raw_id: idx for idx, raw_id in enumerate(raw_labels)}
idx_to_label = {idx: raw_id for idx, raw_id in enumerate(raw_labels)}
NUM_CLASSES  = len(raw_labels)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[CONFIG] 클래스 수: {NUM_CLASSES}, 디바이스: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[GPU] {torch.cuda.get_device_name(0)}")


def set_seed(seed=42):
    """데이터 분할, 증강, 모델 초기화를 최대한 재현 가능하게 고정합니다."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ==============================================================================
# 2. CSV 데이터 로딩 및 윈도우 추출
# ==============================================================================
def extract_windows_from_df(
    df,
    notch_filter=False,
    filter_mode='raw',
    align_peak=False,
    peak_search_before=80,
    peak_search_after=120,
    fs=350.0,
    notch_f0=60.0,
    notch_q=30.0
):
    """이벤트 마커 기준 PRE_EVENT 이전 ~ POST_EVENT 이후 윈도우 추출.

    align_peak=True이면 이벤트 마커 주변에서 에너지가 가장 큰 지점을 다시 찾아
    window 중심으로 사용합니다. 키 입력 이벤트와 실제 근수축 피크 사이의 작은
    시간 차이를 보정하기 위한 실험 옵션입니다.
    """
    ch_cols = [f"CH{i}" for i in range(NUM_CHANNELS)]
    signal  = df[ch_cols].values
    if notch_filter and filter_mode == 'raw':
        filter_mode = 'notch'

    if filter_mode != 'raw':
        # 학습 전처리는 CSV 전체 세션에 offline으로 적용합니다.
        # 실시간 추론의 StreamingFilter는 batch 사이 filter state를 유지하는 방식입니다.
        signal = signal.astype(np.float64)
        if filter_mode == 'notch':
            b, a = scipy_signal.iirnotch(notch_f0, notch_q, fs=fs)
        elif filter_mode == 'bandpass_20_150':
            b, a = scipy_signal.butter(4, [20.0, 150.0], btype='bandpass', fs=fs)
        elif filter_mode == 'bandpass_20_175':
            b, a = scipy_signal.butter(4, [20.0, 174.0], btype='bandpass', fs=fs)
        elif filter_mode == 'highpass_20':
            b, a = scipy_signal.butter(4, 20.0, btype='highpass', fs=fs)
        elif filter_mode == 'highpass_20_notch':
            hb, ha = scipy_signal.butter(4, 20.0, btype='highpass', fs=fs)
            nb, na = scipy_signal.iirnotch(notch_f0, notch_q, fs=fs)
            signal = np.stack([
                scipy_signal.lfilter(nb, na, scipy_signal.lfilter(hb, ha, signal[:, ch] - np.mean(signal[:, ch])))
                for ch in range(NUM_CHANNELS)
            ], axis=1)
            b, a = None, None
        else:
            raise ValueError(f"Unsupported filter_mode: {filter_mode}")

        if b is not None and a is not None:
            signal = np.stack([
                scipy_signal.lfilter(b, a, signal[:, ch] - np.mean(signal[:, ch]))
                for ch in range(NUM_CHANNELS)
            ], axis=1)

    events  = df[df['Event'] != 0]

    X_list, y_list = [], []
    for row_idx, row in events.iterrows():
        center_idx = row_idx
        if align_peak:
            search_start = max(0, row_idx - peak_search_before)
            search_end = min(len(signal), row_idx + peak_search_after + 1)
            search_window = signal[search_start:search_end]
            search_window = search_window - np.mean(search_window, axis=0, keepdims=True)
            row_energy = np.sum(np.square(search_window), axis=1)
            center_idx = search_start + int(np.argmax(row_energy))

        start = center_idx - PRE_EVENT
        end   = center_idx + POST_EVENT
        if start < 0 or end > len(signal):
            continue
        label = int(row['Event'])
        if label not in label_to_idx:
            continue
        X_list.append(signal[start:end])
        y_list.append(label_to_idx[label])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def load_dataset(
    dataset_dir='dataset',
    notch_filter=False,
    filter_mode='raw',
    align_peak=False,
    peak_search_before=80,
    peak_search_after=120
):
    # 스크립트 파일 위치 기준으로 절대경로 변환 (cwd에 무관하게 동작)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, dataset_dir)
    csv_files = sorted(glob.glob(os.path.join(dataset_dir, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"[ERROR] '{dataset_dir}/' 에서 CSV 파일을 찾을 수 없습니다.")

    print(f"[LOAD] 발견된 CSV 파일: {len(csv_files)}개")
    X_all, y_all, session_ids = [], [], []
    session_names = []
    for session_idx, path in enumerate(csv_files):
        df = pd.read_csv(path)
        X, y = extract_windows_from_df(
            df,
            notch_filter=notch_filter,
            filter_mode=filter_mode,
            align_peak=align_peak,
            peak_search_before=peak_search_before,
            peak_search_after=peak_search_after
        )
        if len(X) == 0:
            print(f"  [WARN] {os.path.basename(path)}: 유효한 윈도우 없음, 건너뜀")
            continue
        # 세션별 정규화: 세션 내 신호의 절대적 편차 제거
        # axis=(0,1) → 전체 윈도우 × 시간 축에 걸쳐 채널별 통계 계산
        sess_mean = X.mean(axis=(0, 1), keepdims=True)  # (1, 1, 8)
        sess_std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
        X = (X - sess_mean) / sess_std

        X_all.append(X)
        y_all.append(y)
        session_ids.append(np.full(len(X), session_idx, dtype=np.int32))
        session_name = os.path.basename(path)
        session_names.append(session_name)
        print(f"  [OK] {session_name}: {len(X)}개 윈도우 추출 (세션 정규화 완료)")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    session_ids = np.concatenate(session_ids, axis=0)
    print(f"[DATASET] 전체 샘플 수: {len(X_all)}")
    return X_all, y_all, session_ids, session_names


# ==============================================================================
# 3. 데이터 증강 (train set에만 적용, 정규화 이후 수행)
# ==============================================================================
def augment_emg(X, y, noise_std=0.05, max_shift=5, scale_range=(0.9, 1.1), aug_factor=3):
    """
    ① 가우시안 노이즈  ② 시간축 이동  ③ 진폭 스케일링
    aug_factor: 원본 1개당 추가 생성 수 → 총 샘플 = 원본 × (1 + aug_factor)
    """
    rng = np.random.default_rng(seed=42)
    X_aug_list, y_aug_list = [X], [y]

    for _ in range(aug_factor):
        X_new = X.copy()
        X_new += rng.normal(0, noise_std, X_new.shape).astype(np.float32)

        shifts = rng.integers(-max_shift, max_shift + 1, size=len(X_new))
        for i, shift in enumerate(shifts):
            if shift == 0:
                continue
            X_new[i] = np.roll(X_new[i], shift, axis=0)
            if shift > 0:
                X_new[i, :shift, :] = 0.0
            else:
                X_new[i, shift:, :] = 0.0

        scales = rng.uniform(*scale_range, size=(len(X_new), 1, 1)).astype(np.float32)
        X_new *= scales
        X_aug_list.append(X_new)
        y_aug_list.append(y.copy())

    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    perm  = rng.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]

FINGER_MAPPING = {
    'Thumb':  [' '],
    'Index':  ['y', 'h', 'n', 'u', 'j', 'm'],
    'Middle': ['i', 'k', ','],
    'Ring':   ['o', 'l', '.'],
    'Pinky':  ['p', '[', ']', '\\', ';', "'", '/']
}

# 함수 외부 전역 공간에 반드시 위치해야 함
KEY_TO_FINGER = {}
for finger, keys in FINGER_MAPPING.items():
    for key in keys:
        KEY_TO_FINGER[key] = finger

def process_file_list(
    file_list,
    notch_filter=False,
    filter_mode='raw',
    align_peak=False,
    peak_search_before=80,
    peak_search_after=120
):
    """주어진 CSV 파일 리스트 단위로 데이터를 추출하고 세션 정규화를 적용"""
    X_all, y_all = [], []
    for path in file_list:
        df = pd.read_csv(path)
        X, y = extract_windows_from_df(
            df,
            notch_filter=notch_filter,
            filter_mode=filter_mode,
            align_peak=align_peak,
            peak_search_before=peak_search_before,
            peak_search_after=peak_search_after
        )
        if len(X) == 0:
            print(f"  [WARN] {os.path.basename(path)}: 유효한 윈도우 없음, 건너뜀")
            continue
            
        # 세션별 정규화
        sess_mean = X.mean(axis=(0, 1), keepdims=True)
        sess_std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
        X = (X - sess_mean) / sess_std

        X_all.append(X)
        y_all.append(y)
        print(f"  [OK] {os.path.basename(path)}: {len(X)}개 윈도우 추출 완료")

    if not X_all:
        return np.array([]), np.array([])
        
    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)

# ==============================================================================
# 4. 전처리 파이프라인 (분할 → 정규화 → 증강)
# ==============================================================================
def preprocess(
    dataset_dir='dataset',
    seed=42,
    notch_filter=False,
    filter_mode='raw',
    align_peak=False,
    peak_search_before=80,
    peak_search_after=120,
    split_mode='random'
):
    X_all, y_all, session_ids, session_names = load_dataset(
        dataset_dir,
        notch_filter=notch_filter,
        filter_mode=filter_mode,
        align_peak=align_peak,
        peak_search_before=peak_search_before,
        peak_search_after=peak_search_after
    )

    split_info = {
        'mode': split_mode,
        'train_sessions': '-',
        'val_sessions': '-',
        'test_sessions': '-',
    }

    if split_mode == 'random':
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_all, y_all, test_size=0.30, random_state=seed, stratify=y_all
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
        )
    elif split_mode == 'session':
        unique_session_ids = np.arange(len(session_names))
        if len(unique_session_ids) < 4:
            raise ValueError("--split-mode session requires at least 4 valid CSV sessions.")

        train_sessions, temp_sessions = train_test_split(
            unique_session_ids, test_size=0.30, random_state=seed, shuffle=True
        )
        val_sessions, test_sessions = train_test_split(
            temp_sessions, test_size=0.50, random_state=seed, shuffle=True
        )

        train_mask = np.isin(session_ids, train_sessions)
        val_mask = np.isin(session_ids, val_sessions)
        test_mask = np.isin(session_ids, test_sessions)

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        split_info.update({
            'train_sessions': ', '.join(session_names[i] for i in train_sessions),
            'val_sessions': ', '.join(session_names[i] for i in val_sessions),
            'test_sessions': ', '.join(session_names[i] for i in test_sessions),
        })
        print("[SESSION SPLIT]")
        print(f"  Train sessions: {split_info['train_sessions']}")
        print(f"  Val sessions  : {split_info['val_sessions']}")
        print(f"  Test sessions : {split_info['test_sessions']}")
        _warn_missing_classes(y_train, y_val, y_test)
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    # Train 데이터에만 증강 적용
    if len(X_train) > 0:
        X_train, y_train = augment_emg(X_train, y_train, aug_factor=3)

    print(f"[SPLIT+AUG] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, split_info


def _warn_missing_classes(y_train, y_val, y_test):
    split_labels = {
        'train': set(y_train.tolist()),
        'val': set(y_val.tolist()),
        'test': set(y_test.tolist()),
    }
    expected = set(range(NUM_CLASSES))
    for split_name, labels in split_labels.items():
        missing = sorted(expected - labels)
        if missing:
            missing_keys = [str(idx_to_label[idx]) for idx in missing]
            print(f"  [WARN] {split_name} split missing classes: {', '.join(missing_keys)}")


# ==============================================================================
# 5. PyTorch Dataset
# ==============================================================================
class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================================================
# 6. CNN + LSTM 모델
# ==============================================================================
class CNNLSTM(nn.Module):
    """
    입력: (batch, WINDOW_SIZE=100, NUM_CHANNELS=8)
    Conv1d는 (batch, channels, length) 형태를 요구하므로 permute로 전환.
    CNN → LSTM → Dense 순서로 처리.
    """
    def __init__(self, num_channels, num_classes):
        super().__init__()

        # CNN: 국소 시간 특징 추출
        self.cnn = nn.Sequential(
            # Block 1: (B, 8, 100) → (B, 32, 50)
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Block 2: (B, 32, 50) → (B, 64, 25)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        # LSTM: CNN 출력 시퀀스(25 타임스텝)의 시간 순서 패턴 학습
        self.lstm         = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)

        # Dense Head: BN 제거, Dropout만 유지
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 100, 8)
        x = x.permute(0, 2, 1)     # → (B, 8, 100)
        x = self.cnn(x)             # → (B, 64, 25)
        x = x.permute(0, 2, 1)     # → (B, 25, 64)
        x, _ = self.lstm(x)         # → (B, 25, 64)
        x = x[:, -1, :]             # 마지막 타임스텝 → (B, 64)
        x = self.dropout_lstm(x)
        return self.classifier(x)   # → (B, num_classes)


class ResidualBlock1D(nn.Module):
    """ResNet1D에서 사용하는 skip connection 블록."""

    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # 입력 x를 block 출력에 더해 깊은 모델의 학습 안정성을 높입니다.
        return self.relu(x + self.block(x))

# ==============================================================================
# 12-2. 동일/이전 손가락 간 성능 분석 및 저장
# ==============================================================================
def analyze_finger_performance(cm, class_names, save_dir):
    """혼동 행렬 기반 동일 손가락 및 타 손가락 분류 오차 분석"""
    num_classes = len(class_names)
    
    # 1. 원본 수치 행렬 생성
    finger_cm = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        row_sum = np.sum(cm[i]) if np.sum(cm[i]) > 0 else 1
        finger_cm[i] = cm[i] / row_sum  # 행 정규화 비율 수치 활용
        
    same_finger_scores = []
    diff_finger_scores = []
    
    # 클래스 쌍 조합 반복 탐색
    for i, src_key in enumerate(class_names):
        src_finger = KEY_TO_FINGER.get(src_key, 'Unknown')
        
        for j, tgt_key in enumerate(class_names):
            if i == j:
                continue # 정답 예측 제외
                
            tgt_finger = KEY_TO_FINGER.get(tgt_key, 'Unknown')
            val = finger_cm[i, j]
            
            if src_finger == tgt_finger:
                same_finger_scores.append(val)
            else:
                diff_finger_scores.append(val)

    # 2. 결과 표(텍스트 요약본) 구성 및 출력
    avg_same = np.mean(same_finger_scores) * 100 if same_finger_scores else 0.0
    avg_diff = np.mean(diff_finger_scores) * 100 if diff_finger_scores else 0.0
    
    summary_lines = [
        "\n" + "="*50,
        " [손가락별 동적 변별력 오류 요약 표]",
        "-"*50,
        f" 동일 손가락 내 자판간 오분류 비율 평균 : {avg_same:.2f}%",
        f" 타 손가락 자판간 오분류 비율 평균     : {avg_diff:.2f}%",
        "="*50,
        " * 분석 결과 해석: 동일 손가락 수치가 높을수록 해당 손가락의",
        "   미세 근전도 신호 패턴 간 유사도가 높아 독립 변별이 어려움을 뜻함.",
        "="*50
    ]
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    with open(os.path.join(save_dir, 'finger_analysis_table.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)

    # 3. 데이터 시각화 (막대그래프) 생성
    plt.figure(figsize=(6, 5))
    categories = ['Same Finger', 'Different Finger']
    values = [avg_same, avg_diff]
    colors = ['#ff7f0e', '#1f77b4']
    
    bars = plt.bar(categories, values, color=colors, width=0.5)
    plt.ylabel('Average Misclassification Rate (%)', fontsize=11)
    plt.title('Error Distribution: Same vs Diff Finger', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 막대 위 수치 표기
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.5, f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'finger_error_comparison.png'), dpi=150)
    plt.close()
    print(f"[SAVED] 손가락 비교 그래프: {os.path.join(save_dir, 'finger_error_comparison.png')}")


class ResNet1D(nn.Module):
    """CNNLSTM과 비교하기 위한 1D ResNet 대안 모델."""

    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(64, dropout=0.2),
            ResidualBlock1D(64, dropout=0.2),
            ResidualBlock1D(64, dropout=0.2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


# ==============================================================================
# 7. EarlyStopping (최적 가중치 자동 저장)
# ==============================================================================
class EarlyStopping:
    """validation loss가 개선될 때만 지정된 모델 파일을 갱신합니다."""

    def __init__(self, patience=15, path='best_emg_model.pt'):
        self.patience   = patience
        self.path       = path
        self.best_loss  = float('inf')
        self.counter    = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
            return True   # 개선됨
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


# ==============================================================================
# 8. 에포크 단위 학습 / 평가 함수
# ==============================================================================
def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # Mixed Precision: GPU에서 float16 연산으로 속도 향상
            with torch.autocast(device_type=device.type,
                                 enabled=(device.type == 'cuda')):
                outputs = model(X_batch)
                loss    = criterion(outputs, y_batch)

            if is_train:
                optimizer.zero_grad(set_to_none=True)   # 메모리 효율 최적화
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(y_batch)
            correct    += (outputs.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


# ==============================================================================
# 9. 학습 곡선 시각화
# ==============================================================================
def plot_history(history, save_path='training_history.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_acc'], label='Train Acc')
    axes[0].plot(history['val_acc'],   label='Val Acc')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_loss'], label='Train Loss')
    axes[1].plot(history['val_loss'],   label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SAVED] 학습 곡선: {save_path}")


# ==============================================================================
# 10. 예측값 수집 (혼동 행렬 / 리포트용)
# ==============================================================================
def get_predictions(model, loader, device):
    """테스트 셋 전체 예측값과 실제 레이블 반환"""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            outputs = model(X_batch)
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_true.extend(y_batch.numpy())
    return np.array(y_true), np.array(y_pred)


# ==============================================================================
# 11. 에포크 로그 CSV 저장
# ==============================================================================
def save_epoch_log(history, save_path):
    """에포크별 train/val loss·accuracy를 CSV로 저장"""
    df = pd.DataFrame({
        'epoch':      range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss':   history['val_loss'],
        'train_acc':  [round(a * 100, 2) for a in history['train_acc']],
        'val_acc':    [round(a * 100, 2) for a in history['val_acc']],
    })
    df.to_csv(save_path, index=False)
    print(f"[SAVED] 에포크 로그: {save_path}")


# ==============================================================================
# 12. 혼동 행렬 시각화
# ==============================================================================
def plot_confusion_matrix(cm, class_names, save_path):
    """혼동 행렬 히트맵 PNG 저장"""
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center', fontsize=8,
                color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('실제 레이블', fontsize=12)
    ax.set_xlabel('예측 레이블', fontsize=12)
    ax.set_title('Confusion Matrix (Test Set)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SAVED] 혼동 행렬: {save_path}")


# ==============================================================================
# 13. 텍스트 요약 리포트
# ==============================================================================
def save_text_report(class_names, y_true, y_pred,
                     test_loss, test_acc, history,
                     dataset_info, save_path):
    """실험 조건, 전체 성능, 클래스별 F1 리포트를 텍스트 파일로 저장합니다."""
    lines = []
    lines.append("=" * 60)
    lines.append("   EMG 키보드 분류 모델 학습 결과 리포트")
    lines.append("=" * 60)

    lines.append("\n[하이퍼파라미터]")
    lines.append(f"  WINDOW_SIZE  : {WINDOW_SIZE} samples (~{WINDOW_SIZE/350*1000:.0f}ms @ 350Hz)")
    lines.append(f"  PRE_EVENT    : {PRE_EVENT} / POST_EVENT : {POST_EVENT}")
    lines.append(f"  BATCH_SIZE   : {BATCH_SIZE}")
    lines.append(f"  NUM_CLASSES  : {NUM_CLASSES}")
    lines.append(f"  NUM_CHANNELS : {NUM_CHANNELS}")
    lines.append(f"  Optimizer    : Adam (lr=1e-3)")
    lines.append(f"  Scheduler    : ReduceLROnPlateau (factor=0.5, patience=7)")
    lines.append(f"  EarlyStopping: patience=15")
    lines.append(f"  Augmentation : noise+shift+scale x3 (총 4배)")

    lines.append("\n[데이터셋]")
    for k, v in dataset_info.items():
        lines.append(f"  {k}: {v}")

    lines.append("\n[학습 결과]")
    lines.append(f"  총 에포크      : {len(history['train_loss'])}")
    lines.append(f"  Best Val Loss  : {min(history['val_loss']):.4f}")
    lines.append(f"  Best Val Acc   : {max(history['val_acc'])*100:.2f}%")
    lines.append(f"  Test Loss      : {test_loss:.4f}")
    lines.append(f"  Test Accuracy  : {test_acc*100:.2f}%")

    lines.append("\n[클래스별 성능 (Test Set)]")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0,
    )
    lines.append(report)

    text = "\n".join(lines)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"[SAVED] 텍스트 리포트: {save_path}")
    print(text)


# ==============================================================================
# 14. 메인 실행 (Windows 멀티프로세싱 안전을 위해 if __name__ 가드 필수)
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the EMG keyboard classifier.')
    parser.add_argument(
        '--dataset-dir',
        default='dataset',
        help='CSV dataset directory to use for training. Defaults to dataset.'
    )
    parser.add_argument(
        '--notch-filter',
        action='store_true',
        help='Apply a 60 Hz notch filter to each CSV session before window extraction.'
    )
    parser.add_argument(
        '--filter-mode',
        choices=['raw', 'notch', 'bandpass_20_150', 'bandpass_20_175', 'highpass_20', 'highpass_20_notch'],
        default='raw',
        help='Signal filter to apply before window extraction. Defaults to raw.'
    )
    parser.add_argument(
        '--model',
        choices=['cnn_lstm', 'resnet1d'],
        default='cnn_lstm',
        help='Model architecture to train. Defaults to cnn_lstm.'
    )
    parser.add_argument(
        '--align-peak',
        action='store_true',
        help='Center each training window on the local energy peak near the event marker.'
    )
    parser.add_argument(
        '--peak-search-before',
        type=int,
        default=80,
        help='Samples to search before the event marker when --align-peak is enabled.'
    )
    parser.add_argument(
        '--peak-search-after',
        type=int,
        default=120,
        help='Samples to search after the event marker when --align-peak is enabled.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for model initialization and data loading. Defaults to 42.'
    )
    parser.add_argument(
        '--split-mode',
        choices=['random', 'session'],
        default='random',
        help='Data split strategy. random keeps the previous stratified window split; session holds out CSV recordings.'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=WINDOW_SIZE,
        help='Number of samples per training window. Defaults to 100.'
    )
    parser.add_argument(
        '--pre-event',
        type=int,
        default=PRE_EVENT,
        help='Number of samples before the event marker. Defaults to 40.'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=150,
        help='Maximum number of training epochs. Defaults to 150.'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Early stopping patience. Defaults to 15.'
    )
    parser.add_argument(
        '--model-output',
        default='best_emg_model.pt',
        help='Path to save the best model weights. Defaults to best_emg_model.pt.'
    )
    args = parser.parse_args()
    set_seed(args.seed)

    if args.pre_event < 0 or args.pre_event >= args.window_size:
        raise ValueError('--pre-event must be >= 0 and smaller than --window-size')

    WINDOW_SIZE = args.window_size
    PRE_EVENT = args.pre_event
    POST_EVENT = WINDOW_SIZE - PRE_EVENT

    # ── 데이터 준비 ─────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test, split_info = preprocess(
        args.dataset_dir,
        notch_filter=args.notch_filter,
        filter_mode=args.filter_mode,
        align_peak=args.align_peak,
        peak_search_before=args.peak_search_before,
        peak_search_after=args.peak_search_after,
        split_mode=args.split_mode,
        seed=args.seed
    )

    pin = (DEVICE.type == 'cuda')
    train_loader = DataLoader(EMGDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(EMGDataset(X_val,   y_val),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(EMGDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)

    # ── 모델 / 옵티마이저 / 스케줄러 ─────────────────────────────────────
    if args.model == 'resnet1d':
        model = ResNet1D(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    else:
        model = CNNLSTM(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.5, patience=7, min_lr=1e-5)
    # GradScaler: Mixed Precision 학습 시 수치 안정성 보장
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    early_stopping = EarlyStopping(patience=args.patience, path=args.model_output)

    print(f"\n[MODEL] 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, is_train=True)
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, DEVICE, is_train=False)

        scheduler.step(val_loss)
        improved = early_stopping(val_loss, model)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        lr  = optimizer.param_groups[0]['lr']
        tag = '*' if improved else ' '
        print(f"[{tag}] Epoch {epoch:3d} | "
              f"Train {train_loss:.4f} / {train_acc*100:.1f}% | "
              f"Val {val_loss:.4f} / {val_acc*100:.1f}% | "
              f"LR {lr:.1e}")

        if early_stopping.early_stop:
            print(f"\n[EARLY STOP] {epoch} 에포크에서 조기 종료 "
                  f"(best val_loss: {early_stopping.best_loss:.4f})")
            break

    # ── 최적 모델 로드 후 Test 평가 ───────────────────────────────────────
    model.load_state_dict(
        torch.load(args.model_output, map_location=DEVICE, weights_only=True))
    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, None, scaler, DEVICE, is_train=False)
    print(f"\n[RESULT] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

    # ── 결과 저장 ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', 'runs', timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # 클래스 이름 (idx → 키 문자)
    inv_map     = {v: k for k, v in RIGHT_HAND_KEYS.items()}
    class_names = [inv_map[idx_to_label[i]] for i in range(NUM_CLASSES)]

    # 예측값 수집
    y_true, y_pred = get_predictions(model, test_loader, DEVICE)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    dataset_info = {
        'Dataset directory       ': args.dataset_dir,
        'Preprocessing           ': 'notch' if args.notch_filter and args.filter_mode == 'raw' else args.filter_mode,
        'Peak alignment          ': 'on' if args.align_peak else 'off',
        'Peak search before/after': f'{args.peak_search_before}/{args.peak_search_after}',
        'Split mode              ': args.split_mode,
        'Train sessions          ': split_info['train_sessions'],
        'Val sessions            ': split_info['val_sessions'],
        'Test sessions           ': split_info['test_sessions'],
        'Model                   ': args.model,
        'Model output            ': args.model_output,
        'Seed                    ': args.seed,
        'Max epochs              ': args.max_epochs,
        'Patience                ': args.patience,
        'Train 샘플 수 (증강 후)': len(X_train),
        'Val 샘플 수             ': len(X_val),
        'Test 샘플 수            ': len(X_test),
    }

    # ① 학습 곡선
    plot_history(history,
                 save_path=os.path.join(results_dir, 'training_history.png'))
    # ② 에포크 로그 CSV
    save_epoch_log(history,
                   save_path=os.path.join(results_dir, 'training_log.csv'))
    
    analyze_finger_performance(cm, class_names, save_dir=results_dir)
    # ③ 혼동 행렬
    plot_confusion_matrix(cm, class_names,
                          save_path=os.path.join(results_dir, 'confusion_matrix.png'))
    
    
    # ④ 텍스트 요약 리포트
    save_text_report(class_names, y_true, y_pred,
                     test_loss, test_acc, history,
                     dataset_info,
                     save_path=os.path.join(results_dir, 'report.txt'))

    print(f"\n[SUCCESS] 학습 완료 - '{args.model_output}' 및 {results_dir}/ 저장됨")
