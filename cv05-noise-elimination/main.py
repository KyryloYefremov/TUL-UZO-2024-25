import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_spectrum(img):
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    spectrum = np.log(np.abs(fft_shifted) + 1)
    return spectrum


def simple_averaging(img, kernel_size: int):
    """
    Simple averaging filter
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img


def rotating_mask(img, kernel_size: int):
    """
    Rotating mask filter
    """
    filtered_img = np.zeros_like(img)
    klim = kernel_size - 1  # index limit for kernel

    for x in range(klim, img.shape[0] - klim):
        for y in range(klim, img.shape[1] - klim):
            best_mask = None
            best_mask_var = float('inf')  # variance of the best mask
            # for every mask from 8 possible masks and choose the one with the lowest variance
            for i in range(klim):
                for j in range(klim):
                    mask = img[x-klim+i:x+i, y-klim+j:y+j]
                    mask_var = np.std(mask)

                    if mask_var < best_mask_var:
                        best_mask = mask
                        best_mask_var = mask_var
            filtered_img[x, y] = np.mean(best_mask)
                
    return filtered_img


def median_filter(img, kernel_size: int):
    """
    Median filter
    """
    filtered_img = np.zeros_like(img)
    klim = kernel_size - 1

    for x in range(klim, img.shape[0] - klim):
        for y in range(klim, img.shape[1] - klim):
            mask = img[x-klim:x+klim+1, y-klim:y+klim+1]
            filtered_img[x, y] = np.median(mask)
                
    return filtered_img

    

if __name__ == "__main__":
    img_pathes = ['cv05-noise-elimination/cv05_robotS.bmp', 'cv05-noise-elimination/cv05_PSS.bmp']

    for path in img_pathes:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        original_spectrum = get_spectrum(gray)
        #### Apply filters ####
        kernel_size = 3
        filter_fns = [simple_averaging, rotating_mask, median_filter]
        filter_names = ['Averaging', 'Rotating Mask', 'Median']
        # for every filter:
        for filter_fn, filter_name in zip(filter_fns, filter_names):
            # get filtered image and its spectrum
            filtered_gray = filter_fn(gray, kernel_size)
            filtered_spectrum = get_spectrum(filtered_gray)
            # plot result vs original
            plt.figure(figsize=(10, 6))

            plt.subplot(3, 3, 1)
            plt.imshow(gray, cmap='gray')
            plt.title('Original')
            plt.subplot(3, 3, 2)
            plt.imshow(original_spectrum, cmap='jet')
            plt.title('Original Spectrum')
            plt.colorbar()
            plt.subplot(3, 3, 3)
            plt.hist(gray.ravel(), bins=256, range=[0, 256])
            plt.title('Original Histogram')

            plt.subplot(3, 3, 4)
            plt.imshow(filtered_gray, cmap='gray')
            plt.title(f'{filter_name} Filtered')
            plt.subplot(3, 3, 5)
            plt.imshow(filtered_spectrum, cmap='jet')
            plt.title(f'{filter_name} Spectrum')
            plt.colorbar()
            plt.subplot(3, 3, 6)
            plt.hist(filtered_gray.ravel(), bins=256, range=[0, 256])
            plt.title(f'{filter_name} Histogram')

            plt.tight_layout()
            plt.show()