import os
import shutil
import random


data_dir = "../AmericanFootball/NFL"
output_dir = "datasets/NFL"
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

# 确保文件数量可以被分成5份
assert len(all_files) >= 5, "数据集文件数量过少，无法分为5组。"

# 打乱文件顺序并划分为5个子集
random.seed(42)
random.shuffle(all_files)
num_folds = 5
fold_size = len(all_files) // num_folds
folds = [all_files[i * fold_size: (i + 1) * fold_size] for i in range(num_folds)]

# 留一法交叉验证：将每一份依次作为测试集，其余4份作为训练/验证集
for i in range(num_folds):
    # 创建本次循环的文件夹结构
    train_dir = os.path.join(output_dir, f"fold_{i+1}", "train")
    val_dir = os.path.join(output_dir, f"fold_{i+1}", "val")
    test_dir = os.path.join(output_dir, f"fold_{i+1}", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 分配测试集：第i个fold
    test_files = folds[i]

    # 分配训练/验证集：其余的fold
    train_val_files = [file for j, fold in enumerate(folds) if j != i for file in fold]

    # 可进一步划分训练集和验证集，假设使用 80% 作为训练集，20% 作为验证集
    split_idx = int(len(train_val_files) * 0.8)
    train_files = train_val_files[:split_idx]
    val_files = train_val_files[split_idx:]


    # 将文件内容读取并转换为浮点数后保存到目标文件夹
    def copy_and_convert_to_float(source_file, target_dir):
        with open(source_file, 'r') as f:
            # 假设数据是空格或逗号分隔的数字
            data = f.read().strip().replace(',', ' ').split()
            # 转换为浮点数列表
            float_data = [float(x) for x in data]


        # 将浮点数据写入目标文件
        target_file = os.path.join(target_dir, os.path.basename(source_file))
        with open(target_file, 'w') as f:
            f.write(' '.join(map(str, float_data)))


    # 处理每种数据集的文件
    for file in train_files:
        copy_and_convert_to_float(file, train_dir)
    for file in val_files:
        copy_and_convert_to_float(file, val_dir)
    for file in test_files:
        copy_and_convert_to_float(file, test_dir)

    print(f"完成 fold {i+1} 的数据划分：训练集 {len(train_files)} 个文件，验证集 {len(val_files)} 个文件，测试集 {len(test_files)} 个文件。")
