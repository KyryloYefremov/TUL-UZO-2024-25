import cv2
import numpy as np
import matplotlib.pyplot as plt
import pprint

from table import Table


def erode(img, kernel):
    """Erode the image using the given kernel."""
    return cv2.erode(img, kernel)

def dilate(img, kernel):
    """Dilate the image using the given kernel."""
    return cv2.dilate(img, kernel)

def open_img(img, kernel):
    """Open the image using the given kernel."""
    return dilate(erode(img, kernel), kernel)

def close_img(img, kernel):
    """Close the image using the given kernel."""
    return erode(dilate(img, kernel), kernel)

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


def segmentation(rgb, threshold=80):
    """
    Segment the image using the given threshold.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist[:threshold] = 1
    hist[threshold:] = 0
    return hist[gray]


def plot_result(rgb, binary_img, segmented_img, centroids):
    """"
    Plot the results of the segmentation.
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("Original img")
    ax[0, 1].imshow(binary_img, cmap="gray")
    ax[0, 1].set_title("Processed img")
    ax[1, 0].imshow(segmented_img, cmap="gray")
    ax[1, 0].set_title("Segmented img")
    ax[1, 1].imshow(rgb)
    ax[1, 1].set_title("Centroids")
    for x, y, _ in centroids:
        ax[1, 1].plot(x, y, 'yx', markersize=10)
    plt.tight_layout()
    plt.show()


##################################################################################################
if __name__ == "__main__":
    cvdir = 'cv08-segmentation/'
    imgs = ["cv08_im1.bmp", "cv08_im2.bmp"]

    rgb1 = cv2.imread(cvdir + imgs[0], cv2.IMREAD_COLOR_RGB)
    rgb2 = cv2.imread(cvdir + imgs[1], cv2.IMREAD_COLOR_RGB)

    ### 1. Segmentation
    threshold = 80
    binary_img1 = segmentation(rgb1, threshold)

    R, G, B = rgb2[:,:,0].astype(float), rgb2[:,:,1].astype(float), rgb2[:,:,2].astype(float)
    r = (R * 255) / (R + G + B + 1e-6)
    binary_img2 = (r > 90).astype(np.uint8)
    
    ### 2. Filtering (only for img1)
    kernel = np.ones((5, 5), np.uint8)
    binary_img1 = binary_img1.astype(np.uint8)
    segmented_img1 = open_img(binary_img1, kernel)

    ### 3. Apply coloring_areas
    processed_img1 = colouring_areas(segmented_img1, debug=False)
    processed_img2 = colouring_areas(binary_img2, debug=False)

    ### 4. Get centroids
    centroids1 = get_centroids(processed_img1)
    centroids2 = get_centroids(processed_img2)
    
    ### 5. Plot results
    plot_result(rgb1, binary_img1, segmented_img1, centroids1)
    plot_result(rgb2, binary_img2, binary_img2, centroids2)

    