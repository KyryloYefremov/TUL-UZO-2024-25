import cv2
import numpy as np
import matplotlib.pyplot as plt
import pprint

from table import Table


def colouring_areas(bin_img, debug=False):
    counter = 2
    counters_table = Table()

    # add zero padding to upper, left and right side
    img = np.pad(bin_img, ((1, 0), (1, 1)), mode='constant', constant_values=0)
    
    if debug:
        print(bin_img.shape)
        print(img.shape)
        print()
        print("\nOriginal img:\n" + "="*50)
        pprint.pprint(bin_img)

        print("\nProcess of table updating:")

    # first pass - assign values to pixels
    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1]):
            if img[x, y] == 0:
                continue
            elif img[x, y] == 1:
                # check neighbours from mask
                mask = [img[x, y-1], img[x-1, y-1], img[x-1, y], img[x-1, y+1]]

                # if all neighbours are zero, assign counter value to current pixel
                max_value = max(mask)
                if max_value == 0:
                    img[x, y] = counter
                    counters_table.update_table([counter])
                    counter += 1
                # else assign minimum value from neighbours
                else:
                    img[x, y] = max_value
                    not_zero = set([int(x) for x in mask if x != 0])
                    counters_table.update_table(list(not_zero), debug=debug)


    # return to original size
    img = img[1:, 1:-1]

    if debug:
        print("\nFisrt pass:\n" + "="*50)
        pprint.pprint(img)

    # second pass - assign the highest value to all connected pixels
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] == 0:
                continue
            else:
                # assign the highest value to the pixel
                img[x, y] = counters_table.get_value_by(img[x, y], by_fun=max)
    
    if debug:
        print("\nSecond pass:\n" + "="*50)
        pprint.pprint(img)

    return img.copy()


def get_centroids(img):
    """
    Get centroids of the separate areas in the image. Also returns the area size.
    """
    M, N = img.shape
    centroids = []
    # create meshgrid for x and y coordinates
    y_coords, x_coords = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")

    # per every unique value in image:
    for target_value in np.unique(img):
        # skip zeros
        if target_value == 0:
            continue
            
        # get positions of the target value in the image
        mask = (img == target_value)
        m00 = np.sum(mask)  # total number of target values
        m10 = np.sum(x_coords * mask)  # sum of x coordinates
        m01 = np.sum(y_coords * mask)  # sum of y coordinates

        xt = m10 / m00 
        yt = m01 / m00

        centroids.append((int(xt), int(yt), m00))  # save centroid coordinates and area size
    return centroids


def open_image(image, kernel):
    """
    Perform morphological opening on the image using the given kernel.
    """
    return cv2.dilate(cv2.erode(image, kernel), kernel)


def top_hat_transform(image, kernel):
    """
    Perform top-hat transformation on the image using the given kernel.
    """
    opened_img = open_image(image, kernel)
    return cv2.subtract(image, opened_img)


if __name__ == "__main__":
    cvdir = 'cv09-segmentation/'
    rgb = cv2.imread(cvdir + 'cv09_rice.bmp', cv2.IMREAD_COLOR_RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((15, 15), np.uint8)  # kernel for morphological operations

    # original segmentation
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    threshold = 130
    segmented_img = (gray > threshold).astype(np.uint8) * 255

    # top-hat transformation semgentation
    top_hat = top_hat_transform(gray, kernel)
    hist_top_hat = cv2.calcHist([top_hat], [0], None, [256], [0, 256])
    segmented_top_hat = (top_hat > 50).astype(np.uint8)

    # coloring areas and calculation number of rice grains
    processed_img = colouring_areas(segmented_top_hat, debug=False)
    # centroids of the areas
    centroids = get_centroids(processed_img)

    RICE_SIZE_THRESHOLD = 90  # value of area size to differ rice grains
    total = 0
    for x, y, area in centroids:
        if area <= RICE_SIZE_THRESHOLD:
            total += 1

    total = len(centroids) - total

    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.plot(hist_gray)
    plt.title('Original Image')
    plt.subplot(2, 2, 2)
    plt.plot(hist_top_hat)
    plt.title('Top-Hat Image')
    plt.subplot(2, 2, 3)
    plt.imshow(segmented_img, cmap='gray')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(segmented_top_hat, cmap='gray')
    plt.title('Segmented Top-Hat')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(segmented_top_hat, cmap="gray")
    ax.set_title(f'Rice Grains: {total}')
    for centroid in centroids:
        ax.plot(centroid[0], centroid[1], 'ro', markersize=5)
    ax.axis('off')
    plt.tight_layout()
    plt.show()



