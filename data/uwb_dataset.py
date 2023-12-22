import os
import pandas as pd
from numpy import vstack
import numpy as np
import matplotlib.pyplot as plt
from MWT import morlet_wavelet, wavelet_transform, transform_to_image


def generate_from_csv_files(rootdir):
    # output_arr = []
    labels_arr = None
    CIRs_arr = None
    first = 1
    for dirpath, dirnames, filenames in os.walk(rootdir):
        for file in filenames:
            filename = os.path.join(dirpath, file).replace('\\', '/')
            # 添加文件类型检查，确保当前文件是 CSV 文件
            if not filename.endswith('.csv'):
                continue
            print(filename)
            # read data from file
            df = pd.read_csv(filename, sep=',', header=0)
            input_data = df.to_numpy()
            label_arr = input_data[:, 0].reshape(-1, 1)
            CIR_arr = input_data[:, 15:]
            # append to array
            if first > 0:
                first = 0
                labels_arr = label_arr  # 修改这一行
                CIRs_arr = CIR_arr  # 修改这一行
                output_arr = input_data
            else:
                labels_arr = vstack((labels_arr, label_arr))
                CIRs_arr = vstack((CIRs_arr, CIR_arr))
                # output_arr = vstack((output_arr, input_data))
    # Save the whole dataset to the .npy file
    labels_arr = labels_arr.reshape(-1, 1)
    labels_path = 'C:/LOS_NLOS_Identification-main/LOS_NLOS_Identification-main/data/dataset/labels.npy'
    CIRs_path = 'C:/LOS_NLOS_Identification-main/LOS_NLOS_Identification-main/data/dataset/CIRs.npy'
    np.save(labels_path, labels_arr)
    np.save(CIRs_path, CIRs_arr)
    return labels_arr, CIRs_arr



def generate_labels_file(raw_data_path, labes_path):
    data_array_all = np.load(raw_data_path)
    ''' Save all of the labels(n = 42000) '''
    labels_arr_all = data_array_all[:, 0]
    np.save(labes_path, labels_arr_all)


# def generate_images_file(idx):
#     for i in range(1, idx):
#         CIR_path = 'C:/LOS_NLOS_Identification-main/LOS_NLOS_Identification-main/data/dataset/CIR_uwb_dataset_part' + str(i) + '.npy'
#         CIR_vec = np.load(CIR_path)
#         scales = np.arange(1, 129)
#         grayscale_images_vector = np.empty((len(CIR_vec), 128, 128))
#         for j in range(len(CIR_vec)):
#             wt_res_i = wavelet_transform(CIR_vec[j], scales, morlet_wavelet)
#             grayscale_image_i = transform_to_image(wt_res_i).reshape(1, 128, 128)
#             grayscale_images_vector[j, :, :] = grayscale_image_i
#             print(str(j + 1) + "/" + str(len(CIR_vec)))
#         gray_images_path = 'C:/LOS_NLOS_Identification-main/LOS_NLOS_Identification-main/data/gray_images_part' + str(i) + '.npy'
#         np.save(gray_images_path, grayscale_images_vector)
#     print("Done")

# def generate_images_file(idx):
#         CIR_path = 'C:/LOS_NLOS_Identification-main/LOS_NLOS_Identification-main/data/dataset/CIR_uwb_dataset_part' + str(idx) + '.npy'
#         CIR_vec = np.load(CIR_path)
#         scales = np.arange(1, 129)
#         grayscale_images_vector = np.empty((len(CIR_vec), 128, 128))
#         for j in range(len(CIR_vec)):
#             wt_res_i = wavelet_transform(CIR_vec[j], scales, morlet_wavelet)
#             grayscale_image_i = transform_to_image(wt_res_i).reshape(1, 128, 128)
#             grayscale_images_vector[j, :, :] = grayscale_image_i
#             print(str(j + 1) + "/" + str(len(CIR_vec)))
#         gray_images_path = 'C:/LOS_NLOS_Identification-main/LOS_NLOS_Identification-main/data/gray_images_part' + str(idx) + '.npy'
#         np.save(gray_images_path, grayscale_images_vector)
#         print("Done")

def generate_images_file():
    CIR_path = 'C:/LOS_NLOS_Identification-main/LOS_NLOS_Identification-main/data/dataset/CIRs.npy'
    CIR_vec = np.load(CIR_path)
    scales = np.arange(1, 129)
    grayscale_images_vector = np.empty((len(CIR_vec), 128, 128))
    # 检查文件是否已存在，如果存在则跳过
    if os.path.exists('C:/LOS_NLOS_Identification-main/LOS_NLOS_Identification-main/data/grayscale_images/gray_images_part.npy'):
        print("Gray images file already exists. Skipping.")
        print("grayscale_images_vector.shape:",grayscale_images_vector.shape)
        return

    for i in range(len(CIR_vec)):
        wt_res_i = wavelet_transform(CIR_vec[i], scales, morlet_wavelet)
        grayscale_image_i = transform_to_image(wt_res_i).reshape(1, 128, 128)
        grayscale_images_vector[i, :, :] = grayscale_image_i
        print(str(i + 1) + "/" + str(len(CIR_vec)))
    gray_images_path = 'C:/LOS_NLOS_Identification-main/LOS_NLOS_Identification-main/data/grayscale_images/gray_images_part.npy'
    np.save(gray_images_path, grayscale_images_vector)
    print("Done")

if __name__ == '__main__':
    # Import raw data from folder with dataset
    rootdir = 'C:/LOS_NLOS_Identification-main/LOS_NLOS_Identification-main/data'
    print("Importing dataset to numpy array")
    print("-------------------------------")
    labels, CIRs = generate_from_csv_files(rootdir=rootdir)
    print("labels.shape:",labels.shape)
    print(labels)
    print("CIRs.shape:",CIRs.shape)
    labels = labels
    CIRs = CIRs
    generate_images_file()
    # print dimensions and data
    # print("Number of samples in dataset: %d" % len(CIRs))
    # print("Length of one sample: %d" % len(CIRs[0]))
    # print("-------------------------------")
    # print("Dataset:")
    # print(CIRs)