import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_distances(input_hist, other_hists):
        # calculate distances between input img histogram and others
        distances = {}
        for f, hist in other_hists.items():
            dist = np.linalg.norm(input_hist - hist)
            distances[f] = dist

        return sorted(distances.items(), key=lambda x: x[1])  # sort distances by dist value


if __name__ == "__main__":
    d = 'cv01/images/'
    img_files = [
        d+im for im in ['im01.jpg','im02.jpg','im03.jpg','im04.jpg','im05.jpg','im06.jpg','im07.jpg','im08.jpg','im09.jpg',]
    ]
    
    # read all images and calc their histograms
    hists = {}
    imgs = {}
    for f in img_files:
        img = cv2.imread(f, cv2.IMREAD_COLOR_RGB)
        hist = np.histogram(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).flatten(), bins=256, range=(0, 256))[0]
        hists[f] = hist
        imgs[f] = img

    # for every img in dir:
    fig, axes = plt.subplots(len(img_files), len(img_files), figsize=(15, 5))
    for j, f in enumerate(img_files):
        input_img = imgs.get(f)
        input_hist = np.histogram(cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY).flatten(), bins=256, range=(0, 256))[0]
        
        # calculate distances between input histogram and other histogram
        sorted_distances = calculate_distances(input_hist, hists)
        
        # show results:
        # by rows: first img - input, others - from min to max distance
        for i, (f, _) in enumerate(sorted_distances):
            img = imgs.get(f)
            axes[j, i].imshow(img)
            axes[j, i].axis('off')

    plt.show()