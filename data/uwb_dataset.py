import os
import pandas as pd
from numpy import vstack
import numpy as np
import matplotlib.pyplot as plt
from MWT import morlet_wavelet, wavelet_transform, transform_to_image


def generate_from_csv_files(rootdir):
    output_arr = []
    labels_arr = []
    CIRs_arr = []
    first = 1
    for dirpath, dirnames, filenames in os.walk(rootdir):
        for file in filenames:
            filename = os.path.join(dirpath, file).replace('\\', '/')
            print(filename)
            # read data from file
            df = pd.read_csv(filename, sep=',', header=0)
            input_data = df.to_numpy()
            label_arr = input_data[:, 0]
            CIR_arr = input_data[:, 15:]
            # append to array
            if first > 0:
                first = 0
                output_arr = input_data
            else:
                labels_arr = vstack((labels_arr, label_arr))
                CIRs_arr = vstack((CIRs_arr, CIR_arr))
    # Save the whole dataset to the .npy file
    labels_path = './dataset/labels.npy'
    CIRs_path = './dataset/CIRs.npy'
    np.save(labels_path, labels_arr)
    np.save(CIRs_path, CIRs_arr)
    return labels_arr, CIRs_arr


def generate_labels_file(raw_data_path, labes_path):
    data_array_all = np.load(raw_data_path)
    ''' Save all of the labels(n = 42000) '''
    labels_arr_all = data_array_all[:, 0]
    np.save(labes_path, labels_arr_all)


def generate_images_file(idx):
    CIR_path = './npy_files/CIR_uwb_dataset_part' + str(idx) + '.npy'
    CIR_vec = np.load(CIR_path)
    scales = np.arange(1, 129)
    grayscale_images_vector = np.empty((len(CIR_vec), 128, 128))
    for i in range(len(CIR_vec)):
        wt_res_i = wavelet_transform(CIR_vec[i], scales, morlet_wavelet)
        grayscale_image_i = transform_to_image(wt_res_i).reshape(1, 128, 128)
        grayscale_images_vector[i, :, :] = grayscale_image_i
        print(str(i + 1) + "/" + str(len(CIR_vec)))
    gray_images_path = './grayscale_images/gray_images_part' + str(idx) + '.npy'
    np.save(gray_images_path, grayscale_images_vector)
    print("Done")


if __name__ == '__main__':
    # Import raw data from folder with dataset
    rootdir = './'
    print("Importing dataset to numpy array")
    print("-------------------------------")
    labels, CIRs = generate_from_csv_files(rootdir=rootdir)
    print(labels.shape)
    print(CIRs.shape)