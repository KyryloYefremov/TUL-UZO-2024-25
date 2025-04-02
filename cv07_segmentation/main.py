import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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


if __name__ == "__main__":
    cvdir = 'cv07-segmentation/'
    rgb = cv2.imread(cvdir + "cv07_segmentace.bmp", cv2.IMREAD_COLOR_RGB)

    ### 1. Segment coins based on histogram thresholding
    B, G, R = rgb[:,:,0].astype(float), rgb[:,:,1].astype(float), rgb[:,:,2].astype(float)
    g = (G * 255) / (R + G + B + 1e-6)

    # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # plt.plot(hist)
    # plt.show()
    threshold = 100  # get from histogram
    binary_img = (g < threshold).astype(np.uint8)
    segmented_img = binary_img * rgb[:,:,0]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb)
    ax[0].set_title("Original")

    ax[1].imshow(segmented_img, cmap="gray")
    ax[1].set_title("Segmented coins")

    plt.show()

    ### 2. - 5. ###
    # gray = cv2.imread(cvdir + "cv07_barveni.bmp", cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)    

    # colouring areas in the image
    processed_img = colouring_areas(binary_img, False)

    # centroids of the areas
    centroids = get_centroids(processed_img)

    COIN_SIZE_THRESHOLD = 4000  # value of area size to differ 1 and 5 czech crowns
    total = 0
    for x, y, area in centroids:
        if area < COIN_SIZE_THRESHOLD:
            v = 1
        else:
            v = 5
        total += v
        print(f"Centroid: ({x}, {y}), Crown value: {v}")
    print(f"Total sum: {total} CZK")

    # plot the image with centroids
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(segmented_img, cmap="gray")
    ax.set_title("Processed image")
    for centroid in centroids:
        ax.plot(centroid[0], centroid[1], 'ro', markersize=5)
    plt.show()

