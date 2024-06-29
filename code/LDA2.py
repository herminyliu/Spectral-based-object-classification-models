import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler


def load_data(base_path):
    categories = ['star', 'qso', 'galaxy']
    file_paths = []
    labels = []
    for category in categories:
        for data_type in ['train', 'valid', 'test']:
            folder_path = os.path.join(base_path, 'dataset', category, data_type)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                file_paths.append(file_path)
                labels.append(category)
    return file_paths, labels


class SpectralDataset:
    def __init__(self, file_paths, labels, my_slice):
        self.file_paths = file_paths
        self.labels = labels
        self.my_slice = my_slice
        self.scaler = StandardScaler()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load data
        data = np.loadtxt(file_path, delimiter=',')[-self.my_slice:].astype(np.float32)

        return data, label


def save_to_csv(dataframe, filename):
    outcsv_dir = os.path.dirname(filename)
    os.makedirs(outcsv_dir, exist_ok=True)
    dataframe.to_csv(filename, index=False)
    print(f"数据已保存到 {filename}")


def bulid_dataset(base_path, my_slice):
    # 加载数据
    file_paths, labels = load_data(base_path)

    # 实例化数据集对象
    dataset = SpectralDataset(file_paths, labels, my_slice)

    # 将数据整合成一个大矩阵 X 和标签向量 y
    X = []
    y = []
    for i in range(len(dataset)):
        data, label = dataset[i]
        X.append(data)
        y.append(label)

    X = np.vstack(X)
    y = np.array(y)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return y, X_scaled


def evaluate_lda_performance(base_path, timestamp, y, X_scaled, wavelength):
    # 初始化LDA模型
    reduced_dim = 2
    lda = LDA(n_components=reduced_dim)

    # 拟合LDA模型并进行降维
    X_lda = lda.fit_transform(X_scaled, y)

    # 获取LDA模型的线性判别分量（权重）
    lda_weights = lda.coef_

    # 将降维后的数据和类别保存到DataFrame中
    df_lda = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(reduced_dim)])
    df_lda['Label'] = y

    # 保存降维后的数据集坐标和类别
    lda_coordinates_file_path = os.path.join(base_path, 'outcsv', f'lda_coordinates_{timestamp}_{wavelength}.csv')
    save_to_csv(df_lda, lda_coordinates_file_path)

    # 保存原始维度的权重
    lda_weights_file_path = os.path.join(base_path, 'outcsv', f'lda_weights_{timestamp}_{wavelength}.csv')
    save_to_csv(pd.DataFrame(lda_weights), lda_weights_file_path)

    # 计算类中心
    class_centers = {}
    for category in np.unique(y):
        class_centers[category] = np.mean(X_lda[y == category], axis=0)

    # 计算类内距离（样本与类中心之间的距离）
    intra_cluster_distances = {}
    for category in np.unique(y):
        center = class_centers[category]
        intra_cluster_distances[category] = np.mean(np.linalg.norm(X_lda[y == category] - center, axis=1))

    distances_values = list(intra_cluster_distances.values())
    average_intra_cluster_distances = np.mean(distances_values)

    # 计算类间距离（类中心之间的距离）
    inter_cluster_distances = {}
    for i, category_i in enumerate(np.unique(y)):
        for j, category_j in enumerate(np.unique(y)):
            if j > i:  # 只计算上三角部分，避免重复计算
                inter_cluster_distances[(category_i, category_j)] = np.linalg.norm(class_centers[category_i] - class_centers[category_j])

    distances_values = list(inter_cluster_distances.values())
    average_inter_cluster_distances = np.mean(distances_values)

    # 输出评估结果保存到Excel文件
    result_excel_file = os.path.join(base_path, 'outcsv', f'lda_evaluation_results_{timestamp}_{wavelength}.xlsx')
    with pd.ExcelWriter(result_excel_file) as writer:
        df_centers = pd.DataFrame.from_dict(class_centers, orient='index', columns=[f'LD{i+1}' for i in range(reduced_dim)])
        df_centers.to_excel(writer, sheet_name='类中心坐标')

        df_intra_distances = pd.DataFrame.from_dict(intra_cluster_distances, orient='index', columns=['类内距离'])
        df_intra_distances.index.name = '类别'
        df_intra_distances.to_excel(writer, sheet_name='类内距离')

        df_inter_distances = pd.DataFrame(list(inter_cluster_distances.items()), columns=['类别对', '类间距离'])
        df_inter_distances.to_excel(writer, sheet_name='类间距离', index=False)

    print(f"评估结果已保存到 {result_excel_file}")

    return average_inter_cluster_distances, average_intra_cluster_distances


# 示例用法
if __name__ == "__main__":
    base_path = r'E:\Studying\24上课程\基于光谱的天体分类模型'
    timestamp = pd.Timestamp.now().strftime('%m-%d-%H%M%S')
    avg_inter_lst = []
    avg_intra_lst = []
    ratio_lst = []
    wavelength = range(2600, 2601, 1)

    for i in wavelength:
        y, X = bulid_dataset(base_path, i)
        avg_inter, avg_intra = evaluate_lda_performance(base_path, timestamp, y, X, i)
        avg_intra_lst.append(avg_intra)
        avg_inter_lst.append(avg_inter)
        ratio = avg_inter/avg_intra
        ratio_lst.append(ratio)
        print(f"已完成{(i-1000)/10+1}个循环")

    import matplotlib.pyplot as plt

    plt.figure()  # 设置图的大小，单位是英寸
    plt.plot(wavelength, avg_inter_lst, color='b', label='average_inter_cluster_distances')  # 绘制第一条折线
    plt.plot(wavelength, avg_intra_lst, color='r', label='average_intra_cluster_distances')  # 绘制第二条折线
    plt.plot(wavelength, ratio_lst, color='g', label='ratio')  # 绘制第三条折线
    plt.xlabel('min_wavelength')  # 设置X轴标签
    plt.ylabel('Distance')  # 设置Y轴标签
    plt.title(f'min_wavelength-Distance-{timestamp}')  # 设置图的标题
    plt.legend()  # 显示图例
    plt.savefig(fr"E:\Studying\24上课程\基于光谱的天体分类模型\outfig\lda-distance-min_wavelength-rel-{timestamp}.png", dpi=600)
