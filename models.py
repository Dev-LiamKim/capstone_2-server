import torch.nn as nn


class CNNLSTM(nn.Module):
    """학습/추론에서 기본으로 사용하는 1D CNN + LSTM 분류 모델.

    입력 shape은 train.py/inference.py 모두 (batch, time, channel)입니다.
    Conv1d는 (batch, channel, time)을 기대하므로 forward에서 축을 바꿉니다.
    """

    def __init__(self, num_channels, num_classes):
        super().__init__()
        # CNN 블록은 짧은 시간 구간의 국소적인 EMG 패턴을 먼저 추출합니다.
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        # LSTM은 CNN이 뽑은 시간축 특징의 순서를 반영합니다.
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # (batch, time, channel) -> (batch, channel, time)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # LSTM 입력 형태로 다시 (batch, time, feature)로 변환합니다.
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        # 마지막 시점의 hidden state를 전체 window 대표 특징으로 사용합니다.
        x = x[:, -1, :]
        x = self.dropout_lstm(x)
        return self.classifier(x)


class ResidualBlock1D(nn.Module):
    """ResNet1D에서 사용하는 1D residual block."""

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
        # 입력을 그대로 더해 깊은 모델에서도 gradient가 잘 흐르도록 합니다.
        return self.relu(x + self.block(x))


class ResNet1D(nn.Module):
    """CNNLSTM과 비교 실험하기 위한 1D ResNet 분류 모델."""

    def __init__(self, num_channels, num_classes):
        super().__init__()
        # stem에서 채널별 원신호를 feature map으로 확장합니다.
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
        # 시간축 평균 풀링으로 window 길이에 덜 민감한 최종 특징을 만듭니다.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # 입력은 CNNLSTM과 동일하게 (batch, time, channel)입니다.
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
