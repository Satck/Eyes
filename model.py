import torch
import torch.nn as nn

class Config:
    seq_length = 3  # 每个患者的时间步长
    dynamic_features = ['ser', 'axial_length', 'k1', 'al_ratio', 'screen_time']
    static_features = ['gender', 'genetic_risk']  #(1,2)
    num_classes = 4  # 新增类别数
    test_size = 0.2
    batch_size = 32
    hidden_size = 128  # 增大隐藏层维度
    num_layers = 2     # Bi_lstm
    dropout = 0.5
    lr = 1e-3             #0.0001
    epochs = 50

class MyopiaPredictor(nn.Module):
    def __init__(self, input_size, static_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=Config.dropout,
            bidirectional=True  # 双向LSTM
        )
        self.static_fc = nn.Sequential(
            nn.Linear(static_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(Config.dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + hidden_size, 256),  # 双向LSTM输出维度加倍
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, seq, static):
        lstm_out, _ = self.lstm(seq)  # seq(batch_size, seq_len, input_size) out(batch_size, seq_len, hidden_size * 2)
        seq_features = lstm_out[:, -1]  # 取最后一个时间步 (batch_size, hidden_size * 2)
        static_features = self.static_fc(static) #(batch_size, hidden_size)
        combined = torch.cat([seq_features, static_features], dim=1)
        return self.classifier(combined)