import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def calculate_centroid(back_proj):
    """
    Calculates the center of mass of the back-projection image.
    """
    M, N = back_proj.shape
    x, y = np.meshgrid(np.arange(N), np.arange(M))
    sum_bp = np.sum(back_proj)

    xt = np.sum(x * back_proj) / sum_bp
    yt = np.sum(y * back_proj) / sum_bp
    return xt, yt


def track_object(bgr, prev_xy=None):
    """
    Tracks the object in the frame using back-projection and center of mass.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    back_projection = hist[hue]

    # if we proccess other frames
    if prev_xy is not None:
        x1, y1, x2, y2 = prev_xy
        back_projection = back_projection[y1:y2, x1:x2]
        dx, dy = calculate_centroid(back_projection)
        return int(x1 + dx), int(y1 + dy) 
    
    # if we process the first frame
    return calculate_centroid(back_projection)


if __name__ == "__main__":
    # init program
    plt.ion()
    clear = lambda: os.system('clear')
    clear()
    plt.close('all')
    
    # read data
    cv_dir = os.getcwd() + '/cv02-camshift/'
    cap = cv2.VideoCapture(cv_dir + 'cv02_hrnecek.mp4')
    img = cv2.imread(cv_dir + 'cv02_vzor_hrnecek.bmp')

    # get starting coords
    y1, x1, _ = np.floor_divide(img.shape, 2)
    y2, x2, _ = np.floor_divide(img.shape, 2)

    # cacl possib hist
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv_img[:, :, 0]
    hist = np.histogram(hue, 180, (0, 180))[0]
    hist = hist / np.max(hist)

    prev_coords = None
    # process video
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        # get the new predicted position of obj in curr frame
        xt, yt = track_object(bgr, prev_coords)

        # updates coords of tracked obj
        new_x1, new_y1 = abs(int(xt - x1)), abs(int(yt - y1))
        new_x2, new_y2 = abs(int(xt + x2)), abs(int(yt + y2))
        prev_coords = (new_x1, new_y1, new_x2, new_y2)

        # plot tracked obj 
        cv2.rectangle(bgr, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
        cv2.imshow('Image', bgr)

        key = 0xFF & cv2.waitKey(30)
        if key == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()