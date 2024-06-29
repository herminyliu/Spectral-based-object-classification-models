import csv
import os
import random

# 文件路径定义
csv_file_path = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\data_index.csv"
raw_data_folder = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\raw_data"
dataset_folder = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\dataset"
output_csv_path = os.path.join(dataset_folder, "index.csv")

# 创建目标文件夹及其子文件夹（train, test, valid）
categories = ['galaxy', 'qso', 'star']
subfolders = ['train', 'test', 'valid']

for category in categories:
    category_folder = os.path.join(dataset_folder, category)
    os.makedirs(category_folder, exist_ok=True)
    for subfolder in subfolders:
        os.makedirs(os.path.join(category_folder, subfolder), exist_ok=True)

# 读取csv文件内容
samples = []
with open(csv_file_path, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        samples.append(row)

# 初始化计数器
star_count = 0
max_star_count = 5406

# 打开输出的CSV文件
with open(output_csv_path, mode='w', newline='') as output_file:
    fieldnames = ['id', 'type', 'set']
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    # 处理样本文件
    category_samples = {'galaxy': [], 'qso': [], 'star': []}
    for sample in samples:
        id = sample['id']
        type = sample['type']
        src_file = os.path.join(raw_data_folder, f"{id}.txt")

        if type == 'galaxy' or type == 'qso' or (type == 'star' and star_count < max_star_count):
            if type == 'star':
                star_count += 1
            category_samples[type].append((id, src_file))

    for category in categories:
        samples = category_samples[category]
        random.shuffle(samples)
        total_samples = len(samples)
        train_count = int(0.6 * total_samples)
        test_count = int(0.2 * total_samples)

        for i, (id, src_file) in enumerate(samples):
            if i < train_count:
                subfolder = 'train'
            elif i < train_count + test_count:
                subfolder = 'test'
            else:
                subfolder = 'valid'

            dest_folder = os.path.join(dataset_folder, category, subfolder)
            dest_file = os.path.join(dest_folder, f"{id}.txt")

            try:
                with open(src_file, 'rb') as fsrc:
                    with open(dest_file, 'wb') as fdst:
                        fdst.write(fsrc.read())
                print(f"Copied: {id} - {category} - {subfolder}")
                writer.writerow({'id': id, 'type': category, 'set': subfolder})
            except FileNotFoundError:
                print(f"File not found: {src_file}")
            except Exception as e:
                print(f"Error copying {src_file} to {dest_file}: {e}")
