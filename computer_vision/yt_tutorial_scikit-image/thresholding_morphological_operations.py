import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/noisy.jpg", 0)    # 0 is grayscale

# plt.hist(img.flat, bins=100, range=((0, 255)))


"""Erosion + Dilation method"""
    # ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# kernel = np.ones((3,3), np.uint8)
# erosion = cv2.erode(th, kernel, iterations=1)
# dilation = cv2.dilate(erosion, kernel, iterations=1)
# opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)  # erosion + dilation shortcut

"""Median Filter method"""
median = cv2.medianBlur(img, 3) # 3 is the kernel size
ret, th = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imshow("Original image", img)

# cv2.imshow("Otsu image", th)
# cv2.imshow("Erosion image", erosion)
# cv2.imshow("Erosion + Dilation image", dilation)
# cv2.imshow("Opened Image", opening)

cv2.imshow("Thresholded Image", th)
cv2.imshow("Median Image", median)

"""
NOTE: You can median or non-local means filter first and if it didn't improved then you can use 
morphological operations like erosion + dilation method 
"""

cv2.waitKey(0)
cv2.destroyAllWindows()

