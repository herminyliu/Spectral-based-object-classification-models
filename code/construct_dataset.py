import csv
import os

# 文件路径定义
csv_file_path = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\data_index.csv"
raw_data_folder = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\raw_data"
galaxy_folder = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\dataset\\galaxy"
qso_folder = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\dataset\\qso"
star_folder = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\dataset\\star"
output_csv_path = "E:\\Studying\\24上课程\\基于光谱的天体分类模型\\dataset\\copied_samples.csv"

# 创建目标文件夹（如果不存在）
os.makedirs(galaxy_folder, exist_ok=True)
os.makedirs(qso_folder, exist_ok=True)
os.makedirs(star_folder, exist_ok=True)

# 读取csv文件内容
samples = []
with open(csv_file_path, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        samples.append(row)

print(samples)

# 初始化计数器
star_count = 0
max_star_count = 5406

# 打开输出的CSV文件
with open(output_csv_path, mode='w', newline='') as output_file:
    fieldnames = ['id', 'type']
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    # 处理样本文件
    for sample in samples:
        id = sample['id']
        type = sample['type']
        src_file = os.path.join(raw_data_folder, f"{id}.txt")

        if type == 'galaxy':
            dest_folder = galaxy_folder
        elif type == 'qso':
            dest_folder = qso_folder
        elif type == 'star' and star_count < max_star_count:
            dest_folder = star_folder
            star_count += 1
        else:
            continue

        dest_file = os.path.join(dest_folder, f"{id}.txt")
        try:
            with open(src_file, 'rb') as fsrc:
                with open(dest_file, 'wb') as fdst:
                    fdst.write(fsrc.read())
            print(f"Copied: {id} - {type}")
            writer.writerow({'id': id, 'type': type})
        except FileNotFoundError:
            print(f"File not found: {src_file}")
        except Exception as e:
            print(f"Error copying {src_file} to {dest_file}: {e}")
