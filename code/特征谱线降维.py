import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

# 文件夹路径和输出文件路径
dataset_folder = r"E:\Studying\24上课程\基于光谱的天体分类模型\dataset"
output_csv_path = r"E:\Studying\24上课程\基于光谱的天体分类模型\outcsv"
output_fig_path = r"E:\Studying\24上课程\基于光谱的天体分类模型\outfig"


# 读取数据并进行z-score归一化
def read_and_normalize_data(folder_path):
    data = []
    scaler = StandardScaler()
    for class_folder in ['star', 'qso', 'galaxy']:
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            class_data = []
            for file in glob.glob(os.path.join(class_path, '*.txt')):
                with open(file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        # 添加逗号以确保每行末尾有逗号
                        if not line.endswith(','):
                            line += ','
                        try:
                            row = np.fromstring(line, dtype=float, sep=',').reshape(-1, 1)
                            normalized_row = scaler.fit_transform(row)
                            class_data.append(normalized_row)
                        except ValueError as e:
                            print(f"Error loading file {file}: {str(e)}")
                            break  # 如果出现错误，跳出内部循环继续处理下一个文件
            data.append(class_data)
    return data


# 计算每个类别内每个波段的平均值
def calculate_class_means(data):
    class_means = []
    for class_data in data:
        class_mean = np.mean(class_data, axis=0)
        class_means.append(class_mean)
    return np.squeeze(class_means)


# 计算每个类别内每个波段的最大偏移值
def calculate_class_abs(data):
    class_abses = []
    for class_data in data:
        class_abs = np.median(np.abs(class_data), axis=0)
        class_abses.append(class_abs)
    return np.squeeze(class_abses)


# 计算每个类别内每个波段的平均值
def calculate_class_median(data):
    class_medians = []
    for class_data in data:
        class_median = np.median(class_data, axis=0)
        class_medians.append(class_median)
    return np.squeeze(class_medians)


# 保存结果到CSV文件
def save_to_csv(class_means, output_path):
    df = pd.DataFrame(class_means, index=['star', 'qso', 'galaxy'])
    df.to_csv(output_path, header=None)


# 绘制折线图并保存
def plot_and_save_figure(class_means, output_path, title_str):
    plt.figure(figsize=(10, 6))
    wavelengths = np.arange(1, 2601)  # assuming 2600 columns for wavelengths
    for i, class_mean in enumerate(class_means):
        plt.plot(wavelengths, class_mean, label=['star', 'qso', 'galaxy'][i])
    plt.title(title_str)
    plt.xlabel('Wavelength Sample Point')
    plt.ylabel('Standard Deviation')
    plt.legend()
    current_time = time.strftime("%m%d-%H%M%S")
    plt.savefig(os.path.join(output_path, f'spectra_{current_time}.png'), dpi=600)
    plt.close()


# 主程序
if __name__ == '__main__':
    # 1. 读取数据并进行z-score归一化
    data = read_and_normalize_data(dataset_folder)
    data = np.squeeze(data)

    # 2. 计算每个类别内每个波段的平均值/中间值/最大偏移值
    class_means = calculate_class_means(data)
    class_medians = calculate_class_median(data)
    class_abses = calculate_class_abs(data)

    # 3. 保存结果到CSV文件
    if not os.path.exists(output_csv_path):
        os.makedirs(output_csv_path)
    current_time = time.strftime("%m%d-%H%M%S")
    csv_file_path = os.path.join(output_csv_path, f'class_means_{current_time}.csv')
    save_to_csv(class_means, csv_file_path)
    csv_file_path = os.path.join(output_csv_path, f'class_medians_{current_time}.csv')
    save_to_csv(class_medians, csv_file_path)
    csv_file_path = os.path.join(output_csv_path, f'class_abses_{current_time}.csv')
    save_to_csv(class_abses, csv_file_path)
    print(f"结果保存完毕！")

    # 4. 绘制折线图并保存
    if not os.path.exists(output_fig_path):
        os.makedirs(output_fig_path)
    plot_and_save_figure(class_means, output_fig_path, "Average Intensity")
    plot_and_save_figure(class_abses, output_fig_path, "Median Absolute Intensity")
    plot_and_save_figure(class_medians, output_fig_path, "Median Intensity")
