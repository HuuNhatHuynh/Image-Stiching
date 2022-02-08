import numpy as np
import cv2 as cv
from skimage import filters

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def ComputeHomographyMatrix(PointSet1, PointSet2):
    m = len(PointSet1)
    A = np.zeros((2 * m, 9))
    for i in range(m):
        A[2 * i, :] = [PointSet1[i][0], PointSet1[i][1], 1, 0, 0, 0, -PointSet2[i][0] * PointSet1[i][0], -PointSet2[i][0] * PointSet1[i][1], -PointSet2[i][0]]
        A[2 * i + 1, :] = [0, 0, 0, PointSet1[i][0], PointSet1[i][1], 1, -PointSet2[i][1] * PointSet1[i][0], -PointSet2[i][1] * PointSet1[i][1], -PointSet2[i][1]]
    _, _, V = np.linalg.svd(A)
    H = np.reshape(V[-1, :], (3, 3))
    return H

def HarrisDetector(img, window = 5, alpha = 0.05, threshold = 2000000):
    Ix = filters.sobel_v(img)
    Iy = filters.sobel_h(img)
    ker = cv.getGaussianKernel(window, sigma = 1)
    Ixx = cv.filter2D(Ix ** 2, -1, ker)
    Ixy = cv.filter2D(Ix * Iy, -1, ker)
    Iyy = cv.filter2D(Iy ** 2, -1, ker)
    response = (Ixx * Iyy - Ixy ** 2) - alpha * (Ixx + Iyy) ** 2
    response[response < threshold] = 0
    return response

