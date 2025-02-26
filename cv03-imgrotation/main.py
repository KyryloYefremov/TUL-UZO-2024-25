import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    plt.ion()
    clear = lambda: os.system('clear')
    clear()
    plt.close('all')

    img_path = '/Users/kirillefremov/development/PycharmProjects/TUL-UZO-2024-25/cv03-imgrotation/cv03_robot.bmp'
    origin_img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

    # calc
    angle = float(input("Enter degree: "))
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # calc the size of new transformed img
    H, W = origin_img.shape[:2]
    new_W = int(abs(W * cos_theta) + abs(H * sin_theta))
    new_H = int(abs(W * sin_theta) + abs(H * cos_theta))
    # init new img array with white pixels
    rotated_img = np.zeros((new_W, new_H, origin_img.shape[2]), dtype=np.uint8) + 255

    # center of original and new images
    c_x_old, c_y_old = W // 2, H // 2
    c_x_new, c_y_new = new_W // 2, new_H // 2

    for y2 in range(new_H):
        for x2 in range(new_W):
            # inv transformation to the original coords
            x1 = (x2 - c_x_new) * cos_theta + (y2 - c_y_new) * sin_theta + c_x_old
            y1 = -(x2 - c_x_new) * sin_theta + (y2 - c_y_new) * cos_theta + c_y_old

            # interpolation
            x1, y1 = int(round(x1)), int(round(y1))

            if 0 <= x1 < W and 0 <= y1 < H:
                rotated_img[y2, x2] = origin_img[y1, x1]

    cv2.imshow('img', origin_img)
    cv2.imshow('rotated', rotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()