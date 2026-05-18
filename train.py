import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from config import RIGHT_HAND_KEYS, CHANNELS

# ==============================================================================
# 1. 하이퍼파라미터 및 환경 설정
# ==============================================================================
WINDOW_SIZE = 100  # 타건 전후 추출한 시간 축 샘플 수 (250ms 분량)
NUM_CHANNELS = CHANNELS  # EMG 입력 채널 수 (8채널)

# 클래스 ID 재매핑 (sparse_categorical_crossentropy 최적화를 위해 0부터 시작하는 인덱스로 변환)
# 기존 ID 예시: 'y': 11, 'space': 41 -> 변환: 0, 1, 2... 형태로 압축
raw_labels = sorted(list(set(RIGHT_HAND_KEYS.values())))
label_to_idx = {raw_id: idx for idx, raw_id in enumerate(raw_labels)}
idx_to_label = {idx: raw_id for raw_id in enumerate(raw_labels)}
NUM_CLASSES = len(raw_labels)  # 총 분류 대상 키 개수

# ==============================================================================
# 2. 가상 데이터 생성 가이드 (실제 학습 시 CSV 로드 데이터로 대체 필요)
# ==============================================================================
def load_and_preprocess_dataset():
    """
    [CSV 로드 후 전처리 예시 가이드]
    - 추출한 (100, 8) 크기의 EMG 패킷 배열 필요
    - 레이블(Event ID)은 label_to_idx 딕셔너리를 거쳐 정수 인덱스로 변환
    """
    # 데모용 임의 데이터 생성 (샘플 1000개, 시간 100, 채널 8)
    X_raw = np.random.normal(loc=0, scale=100000, size=(1000, WINDOW_SIZE, NUM_CHANNELS))
    y_raw = np.random.choice(raw_labels, size=(1000,))
    
    # 데이터 정규화: 채널별 Z-Score 표준화 (신호 진폭 편차 제거 목적)
    X_scaled = (X_raw - np.mean(X_raw, axis=(0, 1), keepdims=True)) / (np.std(X_raw, axis=(0, 1), keepdims=True) + 1e-8)
    
    # 레이블 인덱싱 적용
    y_indexed = np.array([label_to_idx[y] for y in y_raw])
    
    # 데이터셋 분할 (학습 80%, 검증 20%)
    return train_test_split(X_scaled, y_indexed, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = load_and_preprocess_dataset()

# ==============================================================================
# 3. 1D-CNN 모델 설계 (EMG 시계열 특징 추출 최적화 구조)
# ==============================================================================
def build_emg_1dcnn_model(input_shape, num_classes):
    model = models.Sequential([
        # [Layer 1] 국소적 시간 특징 추출층
        # Input Shape: (100, 8) -> (시간축, 채널축)
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),  # 내부 공변량 변화 방지 및 학습 안정화
        layers.MaxPooling1D(pool_size=2),  # 데이터 다운샘플링 (100 -> 50)
        layers.Dropout(0.2),  # 과적합 방지
        
        # [Layer 2] 심층 특징 및 주파수 패턴 학습층
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),  # 데이터 다운샘플링 (50 -> 25)
        layers.Dropout(0.3),
        
        # [Layer 3] 고차원 공간 특징 병합층
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),  # 시간 축 전체 평균화로 파라미터 경량화 및 과적합 차단
        
        # [Layer 4] 최종 분류 Dense 층
        layers.Dense(units=64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # [Output Layer] Softmax 출력 활성화 함수 적용 (다중 클래스 확률 분포 반환)
        layers.Dense(units=num_classes, activation='softmax')
    ])
    return model

model = build_emg_1dcnn_model(input_shape=(WINDOW_SIZE, NUM_CHANNELS), num_classes=NUM_CLASSES)
model.summary()  # 모델 구조 요약 표 시각화 출력

# ==============================================================================
# 4. 컴파일 및 최적화 설정
# ==============================================================================
# AdamW 계열의 보완 기법인 Adam Optimizer 채택, 정수형 레이블 적용을 위한 Loss 함수 연동
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # 원-핫 인코딩 없이 정수 상태 레이블로 계산 수행
    metrics=['accuracy']
)

# ==============================================================================
# 5. 콜백 함수 및 모델 학습 실행
# ==============================================================================
# 과적합 방지 및 최적 가중치 자동 저장을 위한 콜백 리스트 정의
custom_callbacks = [
    # 검증 손실이 10 에포크 동안 개선되지 않으면 조기 종료
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    # 학습률 스케줄러: 검증 손실 정체 시 학습률을 0.5배 감소시켜 미세 조정 진입
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
    # 모델 체크포인트: 최적의 성능을 낸 가중치 파일 자동 저장
    callbacks.ModelCheckpoint(filepath='best_emg_model.keras', monitor='val_loss', save_best_only=True)
]

# 모델 학습 가동 (배치 크기 32 설정)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,  # config.py 내 BATCH_SIZE 규격 기준
    callbacks=custom_callbacks,
    verbose=1
)

print("\n[SUCCESS] 학습 완료 및 'best_emg_model.keras' 저장 완료")
 