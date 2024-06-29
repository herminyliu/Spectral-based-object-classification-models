import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime


# 自定义数据集
class SpectralDataset(Dataset):
    def __init__(self, file_paths, labels, slice_point):
        self.file_paths = file_paths
        self.labels = labels
        self.slice_point = slice_point
        self.scaler = StandardScaler()  # Initialize scaler once for the dataset

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        slice_point = self.slice_point

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # 添加逗号以确保每行末尾有逗号
                if not line.endswith(','):
                    line += ','
                # Load data
                data = np.loadtxt(file_path, delimiter=',')[slice_point:].astype(np.float32)

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

# 数据集划分，boosting不含有验证集
train_size = len(labels_encoded) // 4
test_size = len(labels_encoded) - train_size
my_batch_size = 128
first_lr = 0.001
sec_lr = 0.0002
my_epoch = 20

train_paths, train_labels = file_paths[:train_size], labels_encoded[:train_size]
test_paths, test_labels = file_paths[train_size:], labels_encoded[train_size:]
start_dim = 512
train_dataset = SpectralDataset(train_paths, train_labels, -1 * start_dim)
test_dataset = SpectralDataset(test_paths, test_labels, -1 * start_dim)

print('数据集划分完毕，开始训练=======')


# 定义神经网络模型
class SpectralNet(nn.Module):
    def __init__(self, l1, l2, l3, l4):
        super(SpectralNet, self).__init__()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型、损失函数和优化器
criterion = nn.CrossEntropyLoss()


# 训练模型
data_type_num = 3
best_valid_loss = float('inf')
timestamp = datetime.now().strftime("%m-%d-%H%M%S") + "_boosting_"

# 初始化样本权重
sample_weights = np.ones(len(train_dataset)) / len(train_dataset)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
# 定义弱分类器数量
num_weak_classifiers = 10
weak_classifiers = [SpectralNet(start_dim, 512, 256, data_type_num).to(device) for _ in range(num_weak_classifiers)]

for clf_idx, weak_clf in enumerate(weak_classifiers):
    optimizer = optim.Adam(weak_clf.parameters(), lr=first_lr if clf_idx < 3 else sec_lr)
    train_loader = DataLoader(train_dataset, batch_size=my_batch_size, shuffle=False, sampler=sampler)  # 因为弱分类器要改权重，所以每个弱分类器的loader不一样
    for epoch in range(my_epoch):
        weak_clf.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = weak_clf(features)
            loss = criterion(outputs, labels)
            loss.mean().backward()
            optimizer.step()

    # 评估弱分类器
    weak_clf.eval()
    with torch.no_grad():
        all_preds = []
        for features, labels in train_loader:
            outputs = weak_clf(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    # 更新样本权重
    errors = (np.array(all_preds) != np.array(train_labels))
    error_rate = np.dot(errors, sample_weights) / np.sum(sample_weights)
    print(f"clf_idx = {clf_idx}, error_rate = {error_rate:.4f}")
    beta = error_rate / (1 - error_rate)
    sample_weights = sample_weights * np.exp(beta * errors)
    sample_weights /= np.sum(sample_weights)
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


# 测试模型
def predict_ensemble(weak_classifiers, features):
    pred_outputs = [clf(features) for clf in weak_classifiers]
    pred_outputs = torch.stack(pred_outputs, dim=0)
    pred_outputs = torch.mean(pred_outputs, dim=0)
    return pred_outputs


test_loader = DataLoader(test_dataset, batch_size=my_batch_size, shuffle=False)
y_true = []
y_pred = []
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = predict_ensemble(weak_classifiers, features)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')


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


def save_nn():
    # 获取当前时间戳
    outcsv_path = 'E:\\Studying\\24上课程\\基于光谱的天体分类模型\\outcsv'

    # 保存网络的层数和参数到TXT文件中
    filename = f"model_details_{timestamp}.txt"
    with open(os.path.join(outcsv_path,filename), 'w') as f:
        # 写入模型的结构信息
        f.write("以下是使用boosting集成学习方法学习多个全连接神经网络分类器的运行日志")
        f.write(f"time of running start={timestamp}\n")
        f.write(f"num_weak_classifiers={num_weak_classifiers}")
        f.write(f"my__batch_size={my_batch_size}\n")
        f.write(f"first learning rate={first_lr}\n")
        f.write(f"second learning rate={sec_lr}\n")
        f.write(f"my_epoch={my_epoch}\n")
        for i, classifiy in enumerate(weak_classifiers):
            f.write(f"Model {i} Architecture:\n")
            f.write(str(classifiy))
            f.write("\n\n")

        # # 写入每一层的参数信息
        # f.write("Model Parameters:\n")
        # for name, param in my_model.named_parameters():
        #     f.write(f"{name}:\n")
        #     f.write(f"  Shape: {param.shape}\n")
        #     f.write(f"  Values: {param}\n\n")


save_nn()
