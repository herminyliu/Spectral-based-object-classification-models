import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from datetime import datetime


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


# 读取数据并提取后1100个特征
def read_data(file_paths):
    data = []
    for file_path in file_paths:
        features = np.loadtxt(file_path, delimiter=',').astype(np.float32)
        data.append(features)
    return np.array(data)


# 划分数据集
train_size = len(labels_encoded) // 2
valid_size = len(labels_encoded) // 4
test_size = len(labels_encoded) - train_size - valid_size

train_paths, train_labels = file_paths[:train_size], labels_encoded[:train_size]
valid_paths, valid_labels = file_paths[train_size:train_size + valid_size], labels_encoded[train_size:train_size + valid_size]
test_paths, test_labels = file_paths[train_size + valid_size:], labels_encoded[train_size + valid_size:]

X_train = read_data(train_paths)
X_valid = read_data(valid_paths)
X_test = read_data(test_paths)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# 线性判别分析 (LDA)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, train_labels)
X_valid_lda = lda.transform(X_valid)
X_test_lda = lda.transform(X_test)

# 保存结果到CSV文件
timestamp = datetime.now().strftime("%m-%d-%H%M%S")
outcsv_path = 'E:\\Studying\\24上课程\\基于光谱的天体分类模型\\outcsv'

train_df = pd.DataFrame(X_train_lda, columns=['LD1', 'LD2'])
train_df['label'] = train_labels
train_df.to_csv(os.path.join(outcsv_path, f'train_lda_{timestamp}.csv'), index=False)

valid_df = pd.DataFrame(X_valid_lda, columns=['LD1', 'LD2'])
valid_df['label'] = valid_labels
valid_df.to_csv(os.path.join(outcsv_path, f'valid_lda_{timestamp}.csv'), index=False)

test_df = pd.DataFrame(X_test_lda, columns=['LD1', 'LD2'])
test_df['label'] = test_labels
test_df.to_csv(os.path.join(outcsv_path, f'test_lda_{timestamp}.csv'), index=False)


# 绘制散点图
def plot_lda_scatter(X, y, title):
    plt.figure()
    colors = ['r', 'g', 'b']
    for i, label in enumerate(np.unique(y)):
        mask = (X[y == label, 0] <= 100) & (X[y == label, 0] >= -100) & (X[y == label, 1] <= 100) & (
                    X[y == label, 1] >= -100)
        plt.scatter(X[y == label, 0][mask], X[y == label, 1][mask], color=colors[i], label=le.inverse_transform([label])[0])
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title(f'{title} {timestamp}')
    plt.legend()
    plt.savefig(f'E:\\Studying\\24上课程\\基于光谱的天体分类模型\\outfig\\lda_scatter_{title}_{timestamp}.png', dpi=600)


plot_lda_scatter(X_train_lda, train_labels, 'LDA Train Data')
plot_lda_scatter(X_valid_lda, valid_labels, 'LDA Valid Data')
plot_lda_scatter(X_test_lda, test_labels, 'LDA Test Data')
