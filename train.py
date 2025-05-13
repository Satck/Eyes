import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from model import MyopiaPredictor

# 配置参数
class Config:
    seq_length = 3  # 每个患者的时间步长
    dynamic_features = ['ser', 'axial_length', 'k1', 'al_ratio', 'screen_time']
    static_features = ['gender', 'genetic_risk']
    num_classes = 4  # 新增类别数
    test_size = 0.2
    batch_size = 32
    hidden_size = 128  # 增大隐藏层维度
    num_layers = 2
    dropout = 0.5
    lr = 1e-3
    epochs = 50



# 自定义数据集（修正标签类型）
class PatientDataset(Dataset):
    def __init__(self, sequences, static_features, labels):
        self.sequences = sequences.astype(np.float32)
        self.static_features = static_features.astype(np.float32)
        self.labels = labels.astype(np.int64)  # 修改为int64类型

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.static_features[idx],
            self.labels[idx]
        )


# 数据预处理（保持不变）
def preprocess_data(df):
    patients = df.groupby('patient_id')
    sequences = []
    static = []
    labels = []

    for pid, data in patients:
        seq_features = data[Config.dynamic_features].values
        sequences.append(seq_features)
        static.append(data[Config.static_features].iloc[0].values)
        labels.append(data['label'].iloc[-1])

    sequences = np.array(sequences)
    static = np.array(static)
    labels = np.array(labels)

    scaler = StandardScaler()
    seq_shape = sequences.shape
    sequences = scaler.fit_transform(
        sequences.reshape(-1, sequences.shape[-1])
    ).reshape(seq_shape)

    joblib.dump(scaler, "./modelSave/standard_scaler.pkl")  # 新增保存步骤

    return sequences, static, labels


# 改进后的训练函数
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()     # BCELoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-4)      # SGD, Adam
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    best_f1 = 0
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for seq, static, labels in train_loader:
            optimizer.zero_grad()

            # 转换数据到设备
            seq = seq.to(device)
            static = static.to(device)
            labels = labels.to(device)

            outputs = model(seq, static)
            loss = criterion(outputs, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # 修正预测方式
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # 计算训练指标
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        # recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        # precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        # confusion = confusion_matrix(all_labels, all_preds)

        # 验证
        val_metrics = evaluate(model, val_loader)
        scheduler.step(val_metrics['f1'])

        # 打印结果
        print(f"Epoch {epoch + 1}/{Config.epochs}")
        print(f"Train Loss: {total_loss / len(train_loader):.4f} | Acc: {train_acc:.2%} | F1: {train_f1:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.2%} | F1: {val_metrics['f1']:.4f}")
        print("=" * 60)

        # 保存最佳模型
        if val_metrics['f1'] > best_f1:
            torch.save(model.state_dict(), "./modelSave/best_model.pth")
            best_f1 = val_metrics['f1']


# 改进后的评估函数
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seq, static, labels in loader:
            seq = seq.to(device)
            static = static.to(device)
            labels = labels.to(device)

            outputs = model(seq, static)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return {
        'loss': total_loss / len(loader),
        'acc': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }


# 主程序（添加设备配置）
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 读取数据
    df = pd.read_csv("./dataset/balanced_class.csv")

    # 预处理
    sequences, static, labels = preprocess_data(df)

    # 划分数据集
    X_train, X_val, s_train, s_val, y_train, y_val = train_test_split(
        sequences, static, labels,
        test_size=Config.test_size,
        stratify=labels,
        random_state=42
    )

    # 创建DataLoader
    train_dataset = PatientDataset(X_train, s_train, y_train)
    val_dataset = PatientDataset(X_val, s_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        pin_memory=True
    )

    # 初始化模型
    model = MyopiaPredictor(
        input_size=len(Config.dynamic_features),
        static_size=len(Config.static_features),
        hidden_size=Config.hidden_size,
        num_layers=Config.num_layers,
        num_classes=Config.num_classes
    ).to(device)

    # 训练模型
    train_model(model, train_loader, val_loader)