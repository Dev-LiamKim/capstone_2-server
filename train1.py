import os
import glob
import itertools
import numpy as np
import pandas as pd
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
# 1. 하이퍼파라미터 세팅 (78.12% 재현 버전)
# ==============================================================================
WINDOW_SIZE  = 150
PRE_EVENT    = 90
POST_EVENT   = WINDOW_SIZE - PRE_EVENT
NUM_CHANNELS = CHANNELS
BATCH_SIZE   = 64

EPOCHS       = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4

T_0          = 15
T_MULT       = 2
ETA_MIN      = 1e-5
PATIENCE     = 15

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

raw_labels   = sorted(list(set(RIGHT_HAND_KEYS.values())))
label_to_idx = {raw_id: idx for idx, raw_id in enumerate(raw_labels)}
idx_to_label = {idx: raw_id for idx, raw_id in enumerate(raw_labels)}
NUM_CLASSES  = len(raw_labels)

# ==============================================================================
# 2. 데이터 전처리 (데이터 증강 없음, 순수 분할)
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

def load_dataset(dataset_dir='new_dataset'):
    csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))
    X_all, y_all = [], []
    for path in csv_files:
        df = pd.read_csv(path)
        X, y = extract_windows_from_df(df)
        if len(X) == 0: continue
        
        sess_mean = X.mean(axis=(0, 1), keepdims=True)
        sess_std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
        X = (X - sess_mean) / sess_std

        X_all.append(X)
        y_all.append(y)

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)

def preprocess(dataset_dir='new_dataset'):
    X_all, y_all = load_dataset(dataset_dir)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.30, random_state=42, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

class EMGDataset(Dataset):
    def __init__(self, X, y, augment=False): # augment 인자 추가
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment # 속성 저장
        
    def __len__(self): return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            # 1. 가우시안 노이즈 주입
            if torch.rand(1).item() > 0.5:
                x = x + torch.randn_like(x) * 0.01
            
            # 2. 시간 축 무작위 롤링
            if torch.rand(1).item() > 0.5:
                shift = torch.randint(-3, 3, (1,)).item() # 범위 확장
                x = torch.roll(x, shifts=shift, dims=0)
                
        return x, self.y[idx]
    
# ==============================================================================
# 3. 모델 아키텍처 (오리지널 CNN-LSTM 복원)
# ==============================================================================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1) 
        context = torch.sum(lstm_output * attn_weights, dim=1)     
        return context

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class CNNLSTM(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.res_block1 = ResBlock1D(128)
        self.pool1 = nn.MaxPool1d(2) # 시퀀스 길이: 150 -> 75
        
        self.mid_conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.res_block2 = ResBlock1D(256) # 맥스풀링 제거하여 길이 75 유지

        # Bi-LSTM 구성 (입력: 256, 출력: 128 * 2 = 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size=256)
        self.dropout_lstm = nn.Dropout(0.4)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), # 입력 차원을 Bi-LSTM 출력 크기인 256으로 고정
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [Batch, Window_Size(150), Channels]
        x = x.permute(0, 2, 1) # [Batch, Channels, 150]
        x = self.init_conv(x)
        x = self.res_block1(x)
        x = self.pool1(x) # [Batch, 128, 75]
        
        x = self.mid_conv(x)
        x = self.res_block2(x) # [Batch, 256, 75]
        
        x = x.permute(0, 2, 1) # [Batch, 75, 256] (LSTM 입력 규격인 batch_first=True 만족)
        x, _ = self.lstm(x) # [Batch, 75, 256]
        x = self.attention(x) # [Batch, 256]
        x = self.dropout_lstm(x)
        return self.classifier(x)
    
# ==============================================================================
# 4. Focal Loss 및 훈련 유틸리티
# ==============================================================================
# FocalLoss 구현부 내부 또는 선언부 수정
# nn.CrossEntropyLoss에 label_smoothing 파라미터 연동 구조 매핑 필요시 아래 형태로 대입
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # label_smoothing 옵션 추가
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none', label_smoothing=self.label_smoothing)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss



class EarlyStopping:
    def __init__(self, patience=PATIENCE, path='best_emg_model.pt'):
        self.patience = patience
        self.path = path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return True
        self.counter += 1
        if self.counter >= self.patience: self.early_stop = True
        return False

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
                loss = criterion(outputs, y_batch)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(y_batch)
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

    return total_loss / total, correct / total

# ==============================================================================
# 6. 리포트 생성 및 시각화 유틸리티
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

def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_acc'], label='Train Acc')
    axes[0].plot(history['val_acc'], label='Val Acc')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[1].plot(history['train_loss'], label='Train Loss')
    axes[1].plot(history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].legend()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_text_report(class_names, y_true, y_pred, test_loss, test_acc, history, save_path):
    report = classification_report(y_true, y_pred, target_names=class_names)
    lines = [
        "============================================================",
        "   EMG 키보드 분류 모델 학습 결과 리포트 (78.12% 재현 버전)",
        "============================================================",
        f"  총 에포크      : {len(history['train_loss'])}",
        f"  Best Val Loss  : {min(history['val_loss']):.4f}",
        f"  Best Val Acc   : {max(history['val_acc']):.2f}%",
        f"  Test Loss      : {test_loss:.4f}",
        f"  Test Accuracy  : {test_acc*100:.2f}%\n",
        "[클래스별 성능 (Test Set)]\n",
        report
    ]
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

# ==============================================================================
# 5. 메인 실행 루프
# ==============================================================================
if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess('new_dataset')
    
    pin = (DEVICE.type == 'cuda')
    # train_loader 선언부 수정 (augment=True 설정)
    train_loader = DataLoader(EMGDataset(X_train, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(EMGDataset(X_val, y_val, augment=False), batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(EMGDataset(X_test, y_test, augment=False), batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    model = CNNLSTM(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    # 레이블 스무딩 계수를 0.1에서 0.02로 하향 조정하여 기본 수렴력 확보
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0, reduction='mean', label_smoothing=0.02)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
    early_stopping = EarlyStopping(patience=PATIENCE)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("\n[학습 시작]==============================================")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, is_train=True)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion, optimizer, scaler, DEVICE, is_train=False)

        scheduler.step()
        improved = early_stopping(val_loss, model)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        lr = optimizer.param_groups[0]['lr']
        tag = '✓' if improved else ' '
        print(f"[{tag}] Epoch {epoch:3d} | Train Loss {train_loss:.4f} / Acc {train_acc*100:.1f}% | Val Loss {val_loss:.4f} / Acc {val_acc*100:.1f}% | LR {lr:.1e}")

        if early_stopping.early_stop:
            print(f"\n[EARLY STOP] {epoch} 에포크 조기 종료 (best val_loss: {early_stopping.best_loss:.4f})")
            break
        
    # ==============================================================================
    # 7. 테스트 셋 평가 및 결과물 파일 추출 (메인 실행 루프 최하단 추가)
    # ==============================================================================
    # 1) 최고 성능 가중치 로딩 및 평가
    model.load_state_dict(torch.load('best_emg_model.pt', map_location=DEVICE, weights_only=True))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, scaler, DEVICE, is_train=False)
    print(f"\n[RESULT] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

    # 2) 저장소 및 타임스탬프 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)

    # 3) 테스트 예측 수행 및 행렬 계산
    y_true, y_pred = get_predictions(model, test_loader, DEVICE)
    cm = confusion_matrix(y_true, y_pred)
    
    # 4) 클래스 이름 매핑 (숫자 라벨 -> 알파벳/기호)
    inv_map = {v: k for k, v in RIGHT_HAND_KEYS.items()}
    class_names = [inv_map[idx_to_label[i]] for i in range(NUM_CLASSES)]

    # 5) 시각화 이미지 및 리포트 저장 수행
    plot_history(history, save_path=f'results/training_history_{timestamp}.png')
    plot_confusion_matrix(cm, class_names, save_path=f'results/confusion_matrix_{timestamp}.png')
    save_text_report(class_names, y_true, y_pred, test_loss, test_acc, history, save_path=f'results/report_{timestamp}.txt')

    # 6) 에포크 로그 CSV 덤프
    pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc']
    }).to_csv(f'results/training_log_{timestamp}.csv', index=False)

    print(f"\n[SUCCESS] 모든 분석 리포트 및 로그 저장 완료 (results/ 폴더)")
            
    # 테스트 셋 평가 코드 생략 (필요 시 test_loader 및 get_predictions 구문 추가)