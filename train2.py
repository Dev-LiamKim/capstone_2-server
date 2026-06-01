import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from config import RIGHT_HAND_KEYS, CHANNELS

# ==============================================================================
# 1. 하이퍼파라미터 세팅
# ==============================================================================
WINDOW_SIZE  = 210
PRE_EVENT    = 140
POST_EVENT   = WINDOW_SIZE - PRE_EVENT
NUM_CHANNELS = CHANNELS
BATCH_SIZE   = 64

EPOCHS       = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4

T_0          = EPOCHS
T_MULT       = 2
ETA_MIN      = 1e-5
PATIENCE     = 45

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

raw_labels   = sorted(list(set(RIGHT_HAND_KEYS.values())))
label_to_idx = {raw_id: idx for idx, raw_id in enumerate(raw_labels)}
idx_to_label = {idx: raw_id for idx, raw_id in enumerate(raw_labels)}
NUM_CLASSES  = len(raw_labels)

# ==============================================================================
# 2. 데이터 전처리 및 증강
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
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        
    def __len__(self): return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone() # 원본 데이터 보존을 위한 복제, shape: [Length, Channels]
        
        if self.augment:
            # 기존 증강 (Gaussian Noise & Scale & Shift)
            if torch.rand(1).item() > 0.5:
                x = x + torch.randn_like(x) * 0.05
            if torch.rand(1).item() > 0.5:
                scale = torch.empty(1).uniform_(0.8, 1.2).item()
                x = x * scale
            if torch.rand(1).item() > 0.5:
                shift = torch.randint(-3, 3, (1,)).item()
                x = torch.roll(x, shifts=shift, dims=0)

            # [추가] 1. Time Warping (시간축 압축/팽창)
            if torch.rand(1).item() > 0.5:
                orig_len = x.shape[0]
                warp_scale = torch.empty(1).uniform_(0.8, 1.2).item()
                
                # F.interpolate 연산을 위한 차원 변경: [1, Channels, Length]
                x_view = x.unsqueeze(0).permute(0, 2, 1)
                x_warped = F.interpolate(x_view, scale_factor=warp_scale, mode='linear', align_corners=False)
                # 원래 차원으로 복구: [Warped_Length, Channels]
                x_warped = x_warped.squeeze(0).permute(1, 0)
                
                # 윈도우 사이즈 맞춤 (크롭 또는 패딩)
                if x_warped.shape[0] > orig_len:
                    start = torch.randint(0, x_warped.shape[0] - orig_len + 1, (1,)).item()
                    x = x_warped[start : start + orig_len, :]
                elif x_warped.shape[0] < orig_len:
                    pad_len = orig_len - x_warped.shape[0]
                    x = torch.cat([x_warped, torch.zeros(pad_len, x.shape[1])], dim=0)

            # [추가] 2. Channel Dropout (특정 센서 채널 마스킹)
            if torch.rand(1).item() > 0.5:
                # 0 ~ (채널수-1) 중 무작위 선택하여 0으로 처리
                drop_ch = torch.randint(0, x.shape[1], (1,)).item()
                x[:, drop_ch] = 0.0

        return x, self.y[idx]

# ==============================================================================
# 3. 모델 아키텍처 (하이브리드 입력 구조)
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
            nn.Dropout(0.4),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class HandcraftedFeatures(nn.Module):
    """ EMG 정통 통계적 피처 추출기 (GPU 연산 최적화) """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: [Batch, Channels, Window_Size]
        
        # 1. RMS (Root Mean Square)
        rms = torch.sqrt(torch.mean(x**2, dim=-1) + 1e-8)
        
        # 2. MAV (Mean Absolute Value)
        mav = torch.mean(torch.abs(x), dim=-1)
        
        # 3. WL (Waveform Length)
        wl = torch.sum(torch.abs(x[:, :, 1:] - x[:, :, :-1]), dim=-1)
        
        # 4. ZC (Zero Crossings)
        signs = torch.sign(x)
        zc = torch.sum(torch.abs(signs[:, :, 1:] - signs[:, :, :-1]) == 2, dim=-1).float()
        
        # 채널별 4개 특징 결합 -> [Batch, Channels * 4]
        return torch.cat([rms, mav, wl, zc], dim=-1)

class HybridCNNLSTM(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        
        # --- Deep Learning Feature Extractor ---
        self.init_conv = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.res_block1 = ResBlock1D(128)
        self.pool1 = nn.MaxPool1d(2) 
        
        self.mid_conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.res_block2 = ResBlock1D(256) 

        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size=256)
        self.dropout_lstm = nn.Dropout(0.4)
        
        # --- Handcrafted Feature Extractor ---
        self.hc_extractor = HandcraftedFeatures()
        hc_dim = num_channels * 4
        
        # --- Hybrid Classifier ---
        combined_dim = 256 + hc_dim
        
        # 스케일이 다른 두 피처군의 안정적 결합을 위한 정규화
        self.feature_bn = nn.BatchNorm1d(combined_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [Batch, Length, Channels]
        x_permuted = x.permute(0, 2, 1) # [Batch, Channels, Length]
        
        # 1. 수작업 물리 피처 추출 [Batch, Channels * 4]
        hc_feats = self.hc_extractor(x_permuted)
        
        # 2. 딥러닝 잠재 피처 추출
        dl_x = self.init_conv(x_permuted)
        dl_x = self.res_block1(dl_x)
        dl_x = self.pool1(dl_x) 
        
        dl_x = self.mid_conv(dl_x)
        dl_x = self.res_block2(dl_x) 
        
        dl_x = dl_x.permute(0, 2, 1) 
        dl_x, _ = self.lstm(dl_x) 
        dl_feats = self.attention(dl_x) # [Batch, 256]
        dl_feats = self.dropout_lstm(dl_feats)
        
        # 3. 피처 결합 (Concatenation)
        combined_feats = torch.cat([dl_feats, hc_feats], dim=-1)
        combined_feats = self.feature_bn(combined_feats)
        
        return self.classifier(combined_feats)

# ==============================================================================
# 4. Focal Loss 및 훈련 유틸리티
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.02):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
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

def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train, mixup_alpha=0.2):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # [추가] 3. MixUp 처리 (학습 시에만 50% 확률로 적용)
            apply_mixup = is_train and torch.rand(1).item() > 0.5
            if apply_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                rand_index = torch.randperm(X_batch.size(0)).to(device)
                
                y_batch_a, y_batch_b = y_batch, y_batch[rand_index]
                X_batch = lam * X_batch + (1 - lam) * X_batch[rand_index]

            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(X_batch)
                
                # MixUp 적용 여부에 따른 Loss 계산 분기
                if apply_mixup:
                    loss = lam * criterion(outputs, y_batch_a) + (1 - lam) * criterion(outputs, y_batch_b)
                else:
                    loss = criterion(outputs, y_batch)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(y_batch)
            
            # 정확도 측정 시에는 예측값 중 가장 높은 클래스를 기준으로 단순 산출
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

    return total_loss / total, correct / total

# ==============================================================================
# 5. 리포트 생성 및 시각화 유틸리티
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
        "   EMG 키보드 분류 하이브리드 모델 학습 결과 리포트",
        "============================================================",
        f"  총 에포크      : {len(history['train_loss'])}",
        f"  Best Val Loss  : {min(history['val_loss']):.4f}",
        f"  Best Val Acc   : {max(history['val_acc'])*100:.2f}%",
        f"  Test Loss      : {test_loss:.4f}",
        f"  Test Accuracy  : {test_acc*100:.2f}%\n",
        "[클래스별 성능 (Test Set)]\n",
        report
    ]
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

# ==============================================================================
# 6. 메인 실행 루프
# ==============================================================================
if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess('new_dataset')
    
    pin = (DEVICE.type == 'cuda')
    train_loader = DataLoader(EMGDataset(X_train, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(EMGDataset(X_val, y_val, augment=False), batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(EMGDataset(X_test, y_test, augment=False), batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    # HybridCNNLSTM 모델로 교체 선언
    model = HybridCNNLSTM(NUM_CHANNELS, NUM_CLASSES).to(DEVICE)
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=3.0, reduction='mean', label_smoothing=0.02)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN)
    
    # Deprecation Warning 해결 규격
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
    # 7. 테스트 셋 평가 및 파일 추출
    # ==============================================================================
    model.load_state_dict(torch.load('best_emg_model.pt', map_location=DEVICE, weights_only=True))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, scaler, DEVICE, is_train=False)
    print(f"\n[RESULT] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)

    y_true, y_pred = get_predictions(model, test_loader, DEVICE)
    cm = confusion_matrix(y_true, y_pred)
    
    inv_map = {v: k for k, v in RIGHT_HAND_KEYS.items()}
    class_names = [inv_map[idx_to_label[i]] for i in range(NUM_CLASSES)]

    plot_history(history, save_path=f'results/training_history_{timestamp}.png')
    plot_confusion_matrix(cm, class_names, save_path=f'results/confusion_matrix_{timestamp}.png')
    save_text_report(class_names, y_true, y_pred, test_loss, test_acc, history, save_path=f'results/report_{timestamp}.txt')

    pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc']
    }).to_csv(f'results/training_log_{timestamp}.csv', index=False)