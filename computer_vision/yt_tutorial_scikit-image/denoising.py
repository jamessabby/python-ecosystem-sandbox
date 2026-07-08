import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

""" DENOISING FUNCTIONS"""

img = cv2.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/noisy.jpg", 1)
kernel = np.ones((3,3), np.float32)/9      # 25 1's so we don't alter the energy of the image so the brightness remains constant

filt_2D = cv2.filter2D(img, -1, kernel) # Sharp details (like noise or grain) are "averaged out" with their neighbors, creating a blurring effect.
blur = cv2.blur(img, (3,3))
gaussian_blur  = cv2.GaussianBlur(img, (3,3), 1)
median_blur = cv2.medianBlur(img, 3)
bilateral_blur = cv2.bilateralFilter(img, 9, 75, 75)

cv2.imshow("Original", img)
# cv2.imshow("2D custom filter", filt_2D)
cv2.imshow("blur", blur)
# cv2.imshow("Guassian Blur", gaussian_blur)
cv2.imshow("Median filter", median_blur)
cv2.imshow("Bilateral filter", bilateral_blur)

"""EDGE DETECTION FUNCTIONS"""

# neuron_img = cv2.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/neuron.jpg", 0)
# canny_edge = cv2.Canny(img, 100, 200)

# cv2.imshow("Original", neuron_img)
# cv2.imshow("Canny", canny_edge)

cv2.waitKey(0)
cv2.destroyAllWindows()



