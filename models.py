import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
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
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout_lstm(x)
        return self.classifier(x)


class ResidualBlock1D(nn.Module):
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
        return self.relu(x + self.block(x))


class ResNet1D(nn.Module):
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
