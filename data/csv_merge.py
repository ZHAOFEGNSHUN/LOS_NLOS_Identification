import os
import pandas as pd
from tqdm import tqdm

def merge_csv_files(folder_path, output_file):
    # 获取文件夹中的所有 CSV 文件
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # 如果没有 CSV 文件，则打印提示信息并返回
    if not csv_files:
        print("文件夹中没有找到 CSV 文件。")
        return

    # 读取第一个 CSV 文件，创建合并后的 DataFrame
    first_file_path = os.path.join(folder_path, csv_files[0])
    merged_df = pd.read_csv(first_file_path)

    # 循环读取并合并其余 CSV 文件
    for csv_file in tqdm(csv_files[1:]):
        csv_file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    # 将合并后的 DataFrame 写入新的 CSV 文件
    merged_df.to_csv(output_file, index=False)
    print(f"CSV 文件合并完成，保存为 {output_file}")

# 用法示例
folder_path = "/Users/bytedance/Desktop/ZFS/LOS_NLOS_Identification/data"
output_file = "merged_output.csv"
merge_csv_files(folder_path, output_file)
