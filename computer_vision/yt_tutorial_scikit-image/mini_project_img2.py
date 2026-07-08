import cv2
import numpy as np
from matplotlib import pyplot as plt

"""1.	Image Importing and Grayscale Conversion"""

img = cv2.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/img4_objects.jpg", 0)

"""Resizing the image"""

# new width
target_with = 500

# ratio = target width / original width
ratio = target_with / img.shape[1]

# new height
target_height = int(img.shape[0] * ratio)

dim = (target_with, target_height)

img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

"""2.   Brightness and Contrast Enhancement"""

# brightness
brighter_img = cv2.convertScaleAbs(img, alpha=1, beta=50)
darker_img = cv2.convertScaleAbs(img, alpha=1, beta=-50)

# contrast
contrast_img = cv2.convertScaleAbs(img, alpha=2, beta=0)

# cv2.imshow("Original Image",img)
# cv2.imshow("Brighter Image", brighter_img)
# cv2.imshow("Darker Image", darker_img)
# cv2.imshow("Contrast Image", contrast_img)

"""3.   Noise Reduction"""

median_img = cv2.medianBlur(img, 3)

# cv2.imshow("Original Image",img)
# cv2.imshow("Median Image", median_img)

"""4.  Edge Filtering"""

from skimage.filters import sobel, scharr, prewitt

sobel_edge = sobel(img)
scharr_edge = scharr(img)
prewitt_edge = prewitt(img)

sobel_norm = cv2.normalize(sobel_edge, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

scharr_norm = cv2.normalize(scharr_edge, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

prewitt_norm = cv2.normalize(prewitt_edge, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# cv2.imshow("Original Image",img)
# cv2.imshow("Sobel Filter", sobel_norm)
# cv2.imshow("Scharr Filter", scharr_norm)
# cv2.imshow("Prewitt Filter", prewitt_norm)

"""5.    Edge Detection"""

canny_edge = cv2.Canny(img, 50, 100)

# cv2.imshow("Original Image",img)
# cv2.imshow("Canny Edge", canny_edge)

"""6.    Thresholding"""

ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# cv2.imshow("Original Image",img)
# cv2.imshow("Threshold Image", th)

"""7.   Morphological Operation"""

kernel = np.ones((3,3), np.uint8)

# erosion
erosion = cv2.erode(th, kernel, iterations=1)

# erosion + dilation
dilation = cv2.dilate(erosion, kernel, iterations=1)

cv2.imshow("Original Image",img)
cv2.imshow("Erosion image", erosion)
cv2.imshow("Erosion + Dilation Image", dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()