import os
import pandas as pd

# 指定CSV文件所在的文件夹路径
folder_path = '/Users/bytedance/Desktop/ZFS/LOS_NLOS_Identification/data/dataset'

# 获取文件夹中所有CSV文件的文件名列表
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 初始化一个空的DataFrame来存储合并后的数据
merged_data = pd.DataFrame()

# 遍历每个CSV文件，读取并合并数据
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    merged_data = pd.concat([merged_data, df])

# 将第一列为1的数据排在前面，为0的数据排在后面，并重新索引
sorted_data = pd.concat([merged_data[merged_data.iloc[:, 0] == 1], 
                         merged_data[merged_data.iloc[:, 0] == 0]])

# # 将合并后的数据写入一个新的CSV文件
output_file_path = 'merged_data.csv'
sorted_data.to_csv(output_file_path, index=False)
print(sorted_data.head(10))
# 21000