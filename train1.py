import os
import glob
import itertools
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from config import RIGHT_HAND_KEYS, CHANNELS

# ==============================================================================
# 1. 하이퍼파라미터 및 환경 설정
# ==============================================================================
WINDOW_SIZE  = 100
PRE_EVENT    = 40
POST_EVENT   = WINDOW_SIZE - PRE_EVENT
NUM_CHANNELS = CHANNELS
BATCH_SIZE   = 64

raw_labels   = sorted(list(set(RIGHT_HAND_KEYS.values())))
label_to_idx = {raw_id: idx for idx, raw_id in enumerate(raw_labels)}
idx_to_label = {idx: raw_id for idx, raw_id in enumerate(raw_labels)}
NUM_CLASSES  = len(raw_labels)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[CONFIG] 클래스 수: {NUM_CLASSES}, 디바이스: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

# ==============================================================================
# 2. CSV 데이터 로딩 및 윈도우 추출
# ==============================================================================
def extract_windows_from_df(df):
    ch_cols = [f"CH{i}" for i in range(NUM_CHANNELS)]
    signal  = df[ch_cols].values
    events  = df[df['Event'] != 0]

    X_list, y_list = [], []
    for row_idx, row in events.iterrows():
        start = row_idx - PRE_EVENT
        end   = row_idx + POST_EVENT
        if start < 0 or end > len(signal):
            continue
        label = int(row['Event'])
        if label not in label_to_idx:
            continue
        X_list.append(signal[start:end])
        y_list.append(label_to_idx[label])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def load_dataset(dataset_dir='dataset'):
    csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"[ERROR] '{dataset_dir}/' 에서 CSV 파일을 찾을 수 없습니다.")

    print(f"[LOAD] 발견된 CSV 파일: {len(csv_files)}개")
    X_all, y_all = [], []
    for path in csv_files:
        df = pd.read_csv(path)
        X, y = extract_windows_from_df(df)
        if len(X) == 0:
            print(f"  ⚠  {os.path.basename(path)}: 유효한 윈도우 없음, 건너뜀")
            continue
        sess_mean = X.mean(axis=(0, 1), keepdims=True)
        sess_std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
        X = (X - sess_mean) / sess_std

        X_all.append(X)
        y_all.append(y)
        print(f"  ✓  {os.path.basename(path)}: {len(X)}개 윈도우 추출 (세션 정규화 완료)")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    print(f"[DATASET] 전체 샘플 수: {len(X_all)}")
    return X_all, y_all

# ==============================================================================
# 3. 데이터 증강 (Time Masking 추가 및 다양화)
# ==============================================================================
def augment_emg(X, y, noise_std=0.05, max_shift=5, scale_range=(0.9, 1.1), aug_factor=3):
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

        # 타임 마스킹 추가 (연속된 시점의 신호 누락 모사)
        max_mask_size = 8
        for i in range(len(X_new)):
            mask_len = rng.integers(1, max_mask_size)
            start = rng.integers(0, X_new.shape[1] - mask_len)
            X_new[i, start:start+mask_len, :] = 0.0

        X_aug_list.append(X_new)
        y_aug_list.append(y.copy())

    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    perm  = rng.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


# ==============================================================================
# 4. 전처리 파이프라인
# ==============================================================================
def preprocess(dataset_dir='dataset'):
    X_all, y_all = load_dataset(dataset_dir)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.30, random_state=42, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    X_train, y_train = augment_emg(X_train, y_train, aug_factor=3)

    print(f"[SPLIT+AUG] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


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
# 6. 고도화된 모델 구조 (용량 확장 및 어텐션 메커니즘 도입)
# ==============================================================================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1) # (B, T, 1)
        context = torch.sum(lstm_output * attn_weights, dim=1)     # (B, H)
        return context

class CNNLSTM(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()

        # CNN 용량 확장 (32/64 -> 64/128) 및 Dropout 상향 (0.2 -> 0.3)
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        # LSTM 은닉 차원 확장 (64 -> 128)
        self.lstm         = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.attention    = Attention(hidden_size=128)
        self.dropout_lstm = nn.Dropout(0.4)

        # 분류기 구조 조정 및 Dropout 상향 (0.4 -> 0.5)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)     # (B, 8, 100)
        x = self.cnn(x)             # (B, 128, 25)
        x = x.permute(0, 2, 1)     # (B, 25, 128)
        x, _ = self.lstm(x)         # (B, 25, 128)
        x = self.attention(x)       # 어텐션 메커니즘 적용 후 (B, 128)
        x = self.dropout_lstm(x)
        return self.classifier(x)


# ==============================================================================
# 7. EarlyStopping
# ==============================================================================
class EarlyStopping:
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
            return True
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

            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(X_batch)
                loss    = criterion(outputs, y_batch)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
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
# 10. 예측값 수집
# ==============================================================================
def get_predictions(model, loader, device):
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
    lines = []
    lines.append("=" * 60)
    lines.append("   EMG 키보드 분류 모델 학습 결과 리포트 (고도화 버전)")
    lines.append("=" * 60)

    lines.append("\n[하이퍼파라미터]")
    lines.append(f"  WINDOW_SIZE  : {WINDOW_SIZE} samples")
    lines.append(f"  BATCH_SIZE   : {BATCH_SIZE}")
    lines.append(f"  Optimizer    : Adam (lr=1e-3, weight_decay=1e-4)")
    lines.append(f"  Scheduler    : CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    lines.append(f"  EarlyStopping: patience=15")
    lines.append(f"  Augmentation : noise+shift+scale+Masking x3")

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
    report = classification_report(y_true, y_pred, target_names=class_names)
    lines.append(report)

    text = "\n".join(lines)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"[SAVED] 텍스트 리포트: {save_path}")


# ==============================================================================
# 14. 메인 실행
# ==============================================================================
if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess()

    pin = (DEVICE.type == 'cuda')
    train_loader = DataLoader(EMGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(EMGDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(EMGDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    model     = CNNLSTM(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    
    # 클래스 불균형 및 취약 클래스 방어를 위한 가중치 연산 및 적용
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # 가중치 감쇠(Weight Decay) 추가를 통한 과적합 방어
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 주기적인 학습률 조정을 위해 주기 스케줄러(CosineAnnealingWarmRestarts) 적용
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    early_stopping = EarlyStopping(patience=15)

    print(f"\n[MODEL] 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, 151):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, DEVICE, is_train=False)

        scheduler.step()
        improved = early_stopping(val_loss, model)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        lr  = optimizer.param_groups[0]['lr']
        tag = '✓' if improved else ' '
        print(f"[{tag}] Epoch {epoch:3d} | Train {train_loss:.4f} / {train_acc*100:.1f}% | Val {val_loss:.4f} / {val_acc*100:.1f}% | LR {lr:.1e}")

        if early_stopping.early_stop:
            print(f"\n[EARLY STOP] {epoch} 에포크에서 조기 종료 (best val_loss: {early_stopping.best_loss:.4f})")
            break

    model.load_state_dict(torch.load('best_emg_model.pt', map_location=DEVICE, weights_only=True))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, scaler, DEVICE, is_train=False)
    print(f"\n[RESULT] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)

    inv_map     = {v: k for k, v in RIGHT_HAND_KEYS.items()}
    class_names = [inv_map[idx_to_label[i]] for i in range(NUM_CLASSES)]

    y_true, y_pred = get_predictions(model, test_loader, DEVICE)
    cm = confusion_matrix(y_true, y_pred)

    dataset_info = {
        'Train 샘플 수 (증강 후)': len(X_train),
        'Val 샘플 수             ': len(X_val),
        'Test 샘플 수            ': len(X_test),
    }

    plot_history(history, save_path=f'results/training_history_{timestamp}.png')
    save_epoch_log(history, save_path=f'results/training_log_{timestamp}.csv')
    plot_confusion_matrix(cm, class_names, save_path=f'results/confusion_matrix_{timestamp}.png')
    save_text_report(class_names, y_true, y_pred, test_loss, test_acc, history, dataset_info, save_path=f'results/report_{timestamp}.txt')

    print("\n[SUCCESS] 학습 완료 — 'best_emg_model.pt' 및 results/ 저장됨")