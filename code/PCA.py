import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def myPCA(scaled_data, reduced_comp):
    # PCA降维
    pca = PCA(n_components=reduced_comp)
    pca_data = pca.fit_transform(scaled_data)

    # 获取当前时间作为文件名的一部分
    current_time = datetime.now().strftime("%m-%d-%H%M%S")

    # 保存方差贡献量到CSV文件
    variance_ratio = pca.explained_variance_ratio_
    variance_df = pd.DataFrame({'Explained Variance Ratio': variance_ratio})
    variance_file_path = os.path.join(out_csv_folder, f"variance_{current_time}.csv")
    variance_df.to_csv(variance_file_path, index=False)
    print(f"Saved explained variance ratio to {variance_file_path}")

    # 保存降维后的权重到CSV文件
    pca_components = pca.components_
    pca_weights_df = pd.DataFrame(pca_components, columns=[f"PC{i + 1}" for i in range(1100)])  # 删完后还有1100列
    weights_file_path = os.path.join(out_csv_folder, f"weights_{current_time}.csv")
    pca_weights_df.to_csv(weights_file_path, index=False)
    print(f"Saved PCA weights to {weights_file_path}")

    # 保存降维后的数据集到CSV文件
    pca_data_df = pd.DataFrame(pca_data)
    pca_data_file_path = os.path.join(out_csv_folder, f"pca_data_{current_time}.csv")
    pca_data_df.to_csv(pca_data_file_path, index=False, header=False)  # 不保存列名和索引
    print(f"Saved PCA transformed data to {pca_data_file_path}")

    # 绘制PCA权重的热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(pca_weights_df, cmap='coolwarm', cbar=True, square=True, annot=False, fmt=".2f")
    plt.title('PCA Weights Heatmap')
    plt.xlabel('Principal Components')
    plt.ylabel('Original Features')
    plt.yticks([])
    plt.tight_layout()

    # 保存热图到文件
    fig_file_path = os.path.join(out_fig_folder, f"pca_weights_heatmap_{current_time}.png")
    plt.savefig(fig_file_path, dpi=600)
    plt.close()
    print(f"Saved PCA weights heatmap to {fig_file_path}")


if __name__ == "__main__":
    # 文件夹路径定义
    dataset_folder = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\dataset"
    out_csv_folder = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\outcsv\\PCA"
    out_fig_folder = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\outfig"

    # 获取全体数据集文件路径列表
    all_data_files = []
    subfolders = ['star', 'qso', 'galaxy']

    for subfolder in subfolders:
        subfolder_path = os.path.join(dataset_folder, subfolder)
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(subfolder_path, file_name)
                all_data_files.append(file_path)

    # 读取全体数据集并整合
    all_data = []
    for file_path in all_data_files:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # 添加逗号以确保每行末尾有逗号
                    if not line.endswith(','):
                        line += ','
                    try:
                        data = np.fromstring(line, dtype=float, sep=',')
                        all_data.append(data)
                    except ValueError as e:
                        print(f"Error loading file {file_path}: {str(e)}")
                        break  # 如果出现错误，跳出内部循环继续处理下一个文件
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")

    # 检查全体数据集的个数
    total_data_count = len(all_data_files)
    print(f"Total number of samples in dataset: {total_data_count}")
    print("所有数据加载完毕")

    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    # 使用 drop 方法删除前1500列
    df = df.drop(columns=df.columns[:1500])

    # 数据预处理（标准化）
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    myPCA(scaled_data, 2)

