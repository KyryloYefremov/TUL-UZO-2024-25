import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_spectrum(img):
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    spectrum = np.log(np.abs(fft_shifted) + 1)
    return spectrum


def laplace_filter(img, kernel_n=1):
    kernels = [
        np.array([[ 0,  1,  0],
                  [ 1, -4,  1],
                  [ 0,  1,  0]], dtype=np.float32),
        np.array([[ 1,  1,  1],
                  [ 1, -8,  1],
                  [ 1,  1,  1]], dtype=np.float32),
    ]

    kernel = kernels[kernel_n]     
    
    filtered_img = np.zeros_like(img, dtype=np.float32)
    klim = kernel.shape[0] - 1

    for x in range(1, img.shape[0] - klim):
        for y in range(1, img.shape[1] - klim):
            mask = img[x-1:x+klim, y-1:y+klim]
            filtered_img[x, y] = np.sum(mask * kernel)
                
    return filtered_img


def sobel_filter(img, kernel_n=1):
    # definiotions of kernels
    kernels = [
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    ]

    filtered_img = np.zeros_like(img, dtype=np.float32)

    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            mask = img[x-1:x+2, y-1:y+2] # 3x3 výřez

            x_grad = np.sum(mask * kernels[0])
            y_grad = np.sum(mask * kernels[1])

            # Gradient jako velikost vektoru
            filtered_img[x, y] = np.sqrt(x_grad**2 + y_grad**2)
                
    return filtered_img


def kirsch_filter(img):
    # definiotions of kernels
    kernels = [np.rot90(np.array([[ 3,  3,  3], [ 3,  0,  3], [-5, -5, -5]], dtype=np.float32), i) for i in range(8)]

    filtered_img = np.zeros_like(img, dtype=np.float32)

    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            max_grad = float('-inf')
            for kernel in kernels:
                mask = img[x-1:x+2, y-1:y+2] # 3x3 mask
                grad_value = np.sum(mask * kernel)
                if grad_value > max_grad:
                    max_grad = grad_value
            
            filtered_img[x, y] = max_grad # assign max gradient value to the pixel
                
    return filtered_img

    

if __name__ == "__main__":
    img_pathes = ['cv06-edge-detection/cv04c_robotC.bmp']

    for path in img_pathes:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        original_spectrum = get_spectrum(gray)

        #### Apply filters ####
        kernel_size = 3
        filter_fns = [laplace_filter, sobel_filter, kirsch_filter]
        filter_names = ['Laplace', 'Sobel', 'Kirsch']
        # for every filter:
        for filter_fn, filter_name in zip(filter_fns, filter_names):
            # get filtered image and its spectrum
            filtered_gray = filter_fn(gray)
            filtered_spectrum = get_spectrum(filtered_gray)
            # plot result vs original
            plt.figure(figsize=(14, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(gray, cmap='gray')
            plt.title('Original')

            plt.subplot(2, 2, 2)
            plt.imshow(original_spectrum, cmap='jet')
            plt.title('Spectrum')
            plt.colorbar()

            plt.subplot(2, 2, 3)
            plt.imshow(filtered_gray, cmap='jet')
            plt.title(f'{filter_name}')
            plt.colorbar()

            plt.subplot(2, 2, 4)
            plt.imshow(filtered_spectrum, cmap='jet')
            plt.title(f'{filter_name} Spectrum')
            plt.colorbar()
        
            plt.show()