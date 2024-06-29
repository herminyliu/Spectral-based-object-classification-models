import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime


# 自定义数据集
class SpectralDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.scaler = StandardScaler()  # Initialize scaler once for the dataset

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # 添加逗号以确保每行末尾有逗号
                if not line.endswith(','):
                    line += ','
                # Load data
                data = np.loadtxt(file_path, delimiter=',').astype(np.float32)
                feature_index = [279, 888, 889, 1045, 1250, 1251, 1532, 66, 67, 68, 84, 85, 1381, 149, 152, 271, 268, 529]
                feature_elements = [data[i] for i in feature_index]
                lagging_elements = data[-1006:]
                data = np.concatenate((feature_elements, lagging_elements))
                # Reshape data to 2D if it's 1D
                if data.ndim == 1:
                    data = data.reshape(-1, 1)  # Reshape to column vector

                # Normalize features
                normalized_features = self.scaler.fit_transform(data).squeeze()

        return torch.tensor(normalized_features), label


# 数据加载函数
def load_data(base_path):
    categories = ['star', 'qso', 'galaxy']
    file_paths = []
    labels = []
    for category in categories:
        for data_type in ['train', 'valid', 'test']:
            folder_path = os.path.join(base_path, category, data_type)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                file_paths.append(file_path)
                labels.append(category)
    return file_paths, labels


# 数据路径
base_path = r'E:\Studying\24上课程\基于光谱的天体分类模型\dataset'

# 加载数据
file_paths, labels = load_data(base_path)

# 标签编码
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# 数据打乱
file_paths, labels_encoded = shuffle(file_paths, labels_encoded, random_state=42)

# 数据集划分
train_size = len(labels_encoded) // 2
valid_size = len(labels_encoded) // 4
test_size = len(labels_encoded) - train_size - valid_size
my_batch_size = 128
first_lr = 0.001
sec_lr = 0.0002
my_epoch = 20

train_paths, train_labels = file_paths[:train_size], labels_encoded[:train_size]
valid_paths, valid_labels = file_paths[train_size:train_size + valid_size], labels_encoded[train_size:train_size + valid_size]
test_paths, test_labels = file_paths[train_size + valid_size:], labels_encoded[train_size + valid_size:]

train_dataset = SpectralDataset(train_paths, train_labels)
valid_dataset = SpectralDataset(valid_paths, valid_labels)
test_dataset = SpectralDataset(test_paths, test_labels)

train_loader = DataLoader(train_dataset, batch_size=my_batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=my_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=my_batch_size, shuffle=False)

print('数据集划分完毕，开始训练=======')


# 定义神经网络模型
class SpectralNet(nn.Module):
    def __init__(self):
        super(SpectralNet, self).__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 32)
        self.fc9 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.softmax(self.fc9(x))
        return x


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型、损失函数和优化器
model = SpectralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=first_lr)

# 训练模型
num_epochs = my_epoch
best_valid_loss = float('inf')
timestamp = datetime.now().strftime("%m-%d-%H%M%S")

train_losses = []
valid_losses = []


for epoch in range(num_epochs):
    if epoch > 10:
        optimizer = optim.Adam(model.parameters(), lr=sec_lr)
    model.train()
    train_loss = 0.0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * features.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for features, labels in valid_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * features.size(0)

    valid_loss = valid_loss / len(valid_loader.dataset)
    valid_losses.append(valid_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'best_model_{timestamp}.pth')

# 测试模型
model.eval()
test_loss = 0.0
y_true = []
y_pred = []
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

test_loss = test_loss / len(test_loader.dataset)
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')


# 绘制并保存损失图
def plot_losses(train_losses, valid_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss-Epoch {timestamp}')
    plt.legend()
    plt.savefig(f'E:\\Studying\\24上课程\\基于光谱的天体分类模型\\outfig\\loss_epoch_{timestamp}.png', dpi=600)


plot_losses(train_losses, valid_losses)


# 绘制并保存混淆矩阵热力图
def plot_confusion_matrix(conf_matrix):
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix {timestamp}')
    plt.savefig(f'E:\\Studying\\24上课程\\基于光谱的天体分类模型\\outfig\\conf_matrix_{timestamp}.png', dpi=600)


plot_confusion_matrix(conf_matrix)


# 保存结果到CSV文件
def save_results_to_csv(train_losses, valid_losses, test_loss, accuracy, conf_matrix):
    outcsv_path = 'E:\\Studying\\24上课程\\基于光谱的天体分类模型\\outcsv'

    # 保存损失列表
    loss_df = pd.DataFrame({'train_loss': train_losses, 'valid_loss': valid_losses})
    loss_df.to_csv(os.path.join(outcsv_path, f'losses_{timestamp}.csv'), index=False)

    # 保存测试结果和混淆矩阵
    results_df = pd.DataFrame({'test_loss': [test_loss], 'accuracy': [accuracy]})
    results_df.to_csv(os.path.join(outcsv_path, f'test_results_{timestamp}.csv'), index=False)

    conf_matrix_df = pd.DataFrame(conf_matrix, index=le.classes_, columns=le.classes_)
    conf_matrix_df.to_csv(os.path.join(outcsv_path, f'conf_matrix_{timestamp}.csv'))


save_results_to_csv(train_losses, valid_losses, test_loss, accuracy, conf_matrix)


def save_nn(my_model):
    # 获取当前时间戳
    outcsv_path = 'E:\\Studying\\24上课程\\基于光谱的天体分类模型\\outcsv'

    # 保存网络的层数和参数到TXT文件中
    filename = f"model_details_{timestamp}.txt"
    with open(os.path.join(outcsv_path,filename), 'w') as f:
        # 写入模型的结构信息
        f.write(f"time of running start={timestamp}\n")
        f.write(f"my__batch_size={my_batch_size}\n")
        f.write(f"first learning rate={first_lr}\n")
        f.write(f"second learning rate={first_lr}\n")
        f.write(f"my_epoch={my_epoch}\n")
        f.write("Model Architecture:\n")
        f.write(str(my_model))
        f.write("\n\n")

        # 写入每一层的参数信息
        f.write("Model Parameters:\n")
        for name, param in my_model.named_parameters():
            f.write(f"{name}:\n")
            f.write(f"  Shape: {param.shape}\n")
            f.write(f"  Values: {param}\n\n")


save_nn(model)
