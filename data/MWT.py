import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


def morlet_wavelet(t, f_c=1, f_b=5):
    """ Define the wavelet basis function for Morlet wavelet """
    return np.pi ** (-0.25) * np.exp(1j * 2 * np.pi * f_c * t) * np.exp((-(t ** 2)) / f_b)


def wavelet_transform(signal, scales, wavelet_function, f_c=1, f_b=5):
    """ Compute the wavelet transform of a signal using the provided wavelet function. """
    dt = 1
    n = len(signal)  # n = 1016
    t = np.arange(0, n) * dt
    result = np.zeros((len(scales), n), dtype=complex)  # 128 * 1016

    for index, scale in enumerate(scales):
        shifted_t = (t - n // 2) / scale
        scaled_wavelet = wavelet_function(shifted_t, f_c, f_b).conjugate()
        result[index, :] = (scale ** (-0.5)) * np.convolve(signal, scaled_wavelet, mode='same')

    return result


def transform_to_image(MWT_res):
    MWT_res_ = np.abs(MWT_res) ** 2
    normalized_spectrum = (MWT_res_ - np.min(MWT_res_)) / (np.max(MWT_res_) - np.min(MWT_res_))
    # Down sampling using the bi-linear interpolation
    grayscale_image = resize(normalized_spectrum, (128, 128), order=1, mode='reflect', anti_aliasing=True)
    return grayscale_image


if __name__ == '__main__':
    array_test = np.load('./npy_files/uwb_dataset_all.npy')
    CIR_vector_LOS = array_test[0][15:]
    CIR_vector_NLOS = array_test[2][15:]
    ''' Compute the wavelet transform '''
    # Define scales for the wavelet transform
    scales = np.arange(1, 129)
    wt_LOS = wavelet_transform(CIR_vector_LOS, scales, morlet_wavelet)
    wt_NLOS = wavelet_transform(CIR_vector_NLOS, scales, morlet_wavelet)
    grayscale_image_LOS = transform_to_image(wt_LOS)
    grayscale_image_NLOS = transform_to_image(wt_NLOS)

    """ Plot the power-delay profile. """
    plt.stem(range(1016), np.abs(CIR_vector_LOS) ** 2, use_line_collection=True)
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.title('Power Delay Profile(LOS)')
    plt.grid(True)
    plt.show()

    plt.stem(range(1016), np.abs(CIR_vector_NLOS) ** 2, use_line_collection=True)
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.title('Power Delay Profile(NLOS)')
    plt.grid(True)
    plt.show()

    ''' Plot the real part of the wavelet transform '''
    plt.figure(figsize=(7, 5))
    frequencies = 1 / scales
    time = np.arange(10)
    plt.imshow(np.abs(wt_LOS) ** 2, aspect='auto', cmap='jet',
               extent=[time[0], time[-1], frequencies[-1],
                       frequencies[0]])  # Adjust the extent to match frequencies and time
    plt.colorbar(label='Magnitude')
    plt.title('Wavelet Transform Result')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.xlim([6.5, 6.8])
    plt.ylim([0.98, 1])
    plt.show()

    plt.figure(figsize=(7, 5))
    frequencies = 1 / scales
    time = np.arange(10)
    plt.imshow(np.abs(wt_NLOS) ** 2, aspect='auto', cmap='jet',
               extent=[time[0], time[-1], frequencies[-1],
                       frequencies[0]])  # Adjust the extent to match frequencies and time
    plt.colorbar(label='Magnitude')
    plt.title('Wavelet Transform Result')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.xlim([6.5, 6.8])
    plt.ylim([0.98, 1])
    plt.show()

    """ Plot the grayscale image. """
    plt.figure(figsize=(5, 5))
    plt.imshow(grayscale_image_LOS, cmap='gray')
    plt.colorbar(label='Intensity')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.imshow(grayscale_image_NLOS, cmap='gray')
    plt.colorbar(label='Intensity')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()
