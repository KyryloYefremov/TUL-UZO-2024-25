import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn



if __name__ == "__main__":
    ############## 1. ##############
    def shift_quadrants(fft):
        """
        This function shifts the quadrants of the FFT result to center the frequency components.
        :param 
            fft: FFT result
        :return: 
            FFT result with quadrants shifted
        """
        CH, CW = fft.shape[0] // 2, fft.shape[1] // 2  # center height, center width
        fft_centered = np.zeros_like(fft)

        fft_centered[:CH, :CW] = fft[CH:, CW:]
        fft_centered[:CH, CW:] = fft[CH:, :CW]
        fft_centered[CH:, :CW] = fft[:CH, CW:]
        fft_centered[CH:, CW:] = fft[:CH, :CW]
        return fft_centered


    image_path = "cv04/data/cv04c_robotC.bmp"
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # spectrum calc
    fft = np.fft.fft2(gray)
    spectrum_orig = np.log(np.abs(fft) + 1)

    fft_centered = shift_quadrants(fft)
    spectrum_centered = np.log(np.abs(fft_centered) + 1)

    plt.figure(figsize=(10, 3))
    # original spectrum
    plt.subplot(1, 2, 1)
    plt.imshow(spectrum_orig, cmap="jet")
    plt.colorbar()
    plt.title("spectrum")

    # spectrum with shifted quadrants
    plt.subplot(1, 2, 2)
    plt.imshow(spectrum_centered, cmap="jet")
    plt.colorbar()
    plt.title("shifted spectrum")

    plt.show()

    ############## 2. ##############
    def filter_image(fft, pass_filter):
        """
        This function applies a filter to the FFT result, 
        computes the inverse FFT to get the filtered image, and normalizes it. 
        It also computes the log-transformed magnitude of the filtered frequency spectrum.
        :param
            fft: FFT result
            pass_filter: filter to apply
        :return:
            filtered image
            log-transformed magnitude of the filtered frequency spectrum
        """
        filtred_freq_spectrum = fft * pass_filter  # apply filter
        img = np.abs(np.fft.ifft2(filtred_freq_spectrum))
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)  # normalize img to 0-255
        magnitude = np.log(np.abs(filtred_freq_spectrum) + 1)
        magnitude[magnitude == 0] = np.nan
        return img, magnitude
    

    dp_1 = cv2.imread("cv04/data/cv04c_filtDP.bmp", cv2.IMREAD_GRAYSCALE) // 255
    dp_2 = cv2.imread("cv04/data/cv04c_filtDP1.bmp", cv2.IMREAD_GRAYSCALE) // 255
    hp_1 = cv2.imread("cv04/data/cv04c_filtHP.bmp", cv2.IMREAD_GRAYSCALE) // 255
    hp_2 = cv2.imread("cv04/data/cv04c_filtHP1.bmp", cv2.IMREAD_GRAYSCALE) // 255

    # filtr "Lower pass"
    img_dp_1, fft_dp_1_img = filter_image(fft_centered, dp_1)
    img_dp_2, fft_dp_2_img = filter_image(fft_centered, dp_2)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(fft_dp_1_img, cmap='jet')
    ax[0, 0].set_title("Spectrum, Filtr DP")
    ax[0, 1].imshow(img_dp_1, cmap="gray")
    ax[0, 1].set_title("Result")
    ax[1, 0].imshow(fft_dp_2_img, cmap='jet')
    ax[1, 1].imshow(img_dp_2, cmap="gray")
    plt.show()

    # filter "Upper pass"
    img_hp_1, fft_hp_1_img = filter_image(fft_centered, hp_1)
    img_hp_2, fft_hp_2_img = filter_image(fft_centered, hp_2)

    _, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(fft_hp_1_img, cmap='jet')
    ax[0, 0].set_title("Spectrum, Filtr HP")
    ax[0, 1].imshow(img_hp_1, cmap="gray")
    ax[0, 1].set_title("Result")
    ax[1, 0].imshow(fft_hp_2_img, cmap='jet')
    ax[1, 1].imshow(img_hp_2, cmap="gray")
    plt.show()

    ############## 3. ##############
    dctS = dctn(gray)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(gray, cmap="gray")

    clrs = ax[1].imshow(np.log(np.abs(dctS)), cmap="jet")
    fig.colorbar(clrs, ax=ax[1])
    ax[1].set_title("DCT Spectrum")
    plt.show()

    ############## 4. ##############
    def dct_limited_calc_and_plot(dct, n):
        """
        This function calculates and plots the DCT spectrum limited to an n x n area
        and the corresponding image. It is called with different values of n.
        :param
            dct: DCT result
            n: size of the area to limit the DCT spectrum to
        :return:
            None
        """
        # calculate dct spectrum limited to NxN area
        dct_limited = np.zeros_like(dct)
        dct_limited[:n, :n] = dct[:n, :n]
        img = idctn(dct_limited)
        # calculate log and mask zeros for better plot visibility
        dct_limited_img = np.log(np.abs(dct_limited) + 1)
        dct_limited_img[dct_limited_img == 0] = np.nan
        # plot result dct and image
        fig, ax = plt.subplots(1, 2)
        clrs = ax[0].imshow(dct_limited_img, cmap='jet')
        fig.colorbar(clrs, ax=ax[0])
        ax[0].set_title(f"DCT Spectrum {n}x{n}")
        ax[1].imshow(img, cmap="gray")
        plt.show()

    dct_limited_calc_and_plot(dctS, 10)
    dct_limited_calc_and_plot(dctS, 30)
    dct_limited_calc_and_plot(dctS, 50)

    ############## 5. ##############
    def calculate_distances(input_hist, other_hists):
        """
        This function calculates the Euclidean distances between 
        the input histogram and other histograms, 
        and returns them sorted by distance.
        :param
            input_hist: input histogram
            other_hists: other histograms
        :return:
            sorted distances
        """
        # calculate distances between input img histogram and others
        distances = {}
        for f, hist in other_hists.items():
            dist = np.linalg.norm(input_hist - hist)
            distances[f] = dist

        return sorted(distances.items(), key=lambda x: x[1])  # sort distances by dist value


    d_new = 'cv04/data/C01_IT_new/uzo_cv01_IT_im0'
    img_files_new = [
        d_new + im for im in ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']
    ]

    # calculation of 5x5 DCT symptom vector
    dct_hists = {}
    imgs_new = {}
    R = 5  # size of DCT coeffs

    for f in img_files_new:
        img = cv2.imread(f, cv2.IMREAD_COLOR_RGB)
        dctM = dctn(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)) 
        dctRvec = dctM[:R, :R].flatten()  # select top RxR coeffs
        dct_hists[f] = dctRvec
        imgs_new[f] = img

    fig, axes = plt.subplots(len(img_files_new), len(img_files_new), figsize=(15, 5))

    for j, f in enumerate(img_files_new):
        input_img = imgs_new[f]
        input_dct = dct_hists[f]

        sorted_distances = calculate_distances(input_dct, dct_hists)

        for i, (f_sorted, _) in enumerate(sorted_distances):
            img = imgs_new[f_sorted]
            axes[j, i].imshow(img)
            axes[j, i].axis("off")

    plt.show()