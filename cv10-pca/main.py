import numpy as np
import matplotlib.pyplot as plt
import cv2


class PCA:

    def __init__(self, imgs):
        # convert to GRAY and flatten the images
        self.imgs = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten() for img in imgs])

    def train(self):
        # create matrix from known images
        Wp = self.imgs.T
        # print(f'Wp shape: {np.array(Wp).shape}')

        # calculate mean vector from Wp matrix rows
        self.mean = np.mean(Wp, axis=1)
        # print(f'mean shape: {np.array(mean).shape}')

        # subtract mean from Wp matrix rows
        W = Wp - self.mean[:, np.newaxis]
        # print(f'W shape: {np.array(W).shape}')

        # calculate covariance matrix
        C = W.T @ W
        # print(f'C shape: {np.array(C).shape}')

        # calculate eigenvalues and eigenvectors
        eigen_val, eigen_vec = np.linalg.eig(C)
        # print(f'eigen_val shape: {np.array(eigen_val).shape}')

        # sort eigenvalues and eigenvectors
        idx = np.argsort(eigen_val)[::-1]
        Ep = eigen_vec[:, idx]
        # print(f'Ep shape: {np.array(Ep).shape}')

        # calculate eigenspace
        self.E = W @ Ep
        # print(f'E shape: {np.array(E).shape}')
        self.PI = self.E.T @ W

    def test(self, img):
        # convert to GRAY and flatten the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
        # subtract mean from the image
        W = img - self.mean
        # project the image into the eigenspace
        self.PI_img = self.E.T @ W
        print(f'PI_img shape: {np.array(self.PI_img).shape}')

    def classify(self):
        # calculate the distance using Euklidean distance between the test image and the known images
        distances = []
        for i in range(len(self.imgs)):
            dist = np.linalg.norm(self.PI_img - self.PI[:, i])
            distances.append(dist)
        
        # find minimum distance
        min_index = distances.index(min(distances))
        # return the most similar image
        return min_index


if __name__ == "__main__":
    data_dir = 'cv10-pca/data/'
    # Load the images
    imgs = []
    for i in range(1, 4):
        for j in range(1, 4):
            img = cv2.imread(f'{data_dir}p{i}{j}.bmp', cv2.IMREAD_COLOR_RGB)
            imgs.append(img)

    pca_classifier = PCA(imgs)
    pca_classifier.train()
    # apply PCA to the test image
    test_img = cv2.imread(f'{data_dir}unknown.bmp', cv2.IMREAD_COLOR_RGB)
    pca_classifier.test(test_img)
    img_index = pca_classifier.classify()

    # show input image and the most similar image
    plt.subplot(1, 2, 1)
    plt.imshow(test_img)
    plt.title('Unknown Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(imgs[img_index])
    plt.title('Similar Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


