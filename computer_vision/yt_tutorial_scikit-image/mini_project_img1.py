import cv2
import numpy as np
from matplotlib import pyplot as plt

"""1. Image Importing and Grayscale Conversion"""

img = cv2.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/img1_low_contrast.jpg", 0)

"""2. Brightness and Contrast Enhancement"""

# Increase Brightness
brighter_img = cv2.convertScaleAbs(img, alpha=1, beta=40)

# Decrease Brightness
darker_img = cv2.convertScaleAbs(img, alpha=1, beta=-20)

# Increase Contrast
contrast_img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

# cv2.imshow("Original image", img)
# cv2.imshow("Brighter image", brighter_img)
# cv2.imshow("Darker image", darker_img)
# cv2.imshow("Adjusted Contrast", contrast_img)


"""3. Noise Reduction"""

median = cv2.medianBlur(img, 3)

# cv2.imshow("Original image", img)
# cv2.imshow("Median Image", median)

"""4. Edge Filtering"""

from skimage.filters import roberts, sobel, prewitt
    
    roberts_edge = roberts(img)
    sobel_edge = sobel(img)
    prewitt_edge = prewitt(img)

roberts_norm = cv2.normalize(roberts_edge, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
roberts_norm = roberts_norm.astype(np.uint8)

sobel_norm = cv2.normalize(sobel_edge, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
sobel_norm = sobel_norm.astype(np.uint8)

prewitt_norm = cv2.normalize(prewitt_edge, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
prewitt_norm = prewitt_norm.astype(np.uint8)

# cv2.imshow("Roberts", roberts_norm)
# cv2.imshow("Sobel", sobel_norm)
# cv2.imshow("Prewitt", prewitt_norm)

"""5. Edge Detection"""

"""
cv2.Canny(src, threshold1, threshold2)

threshold1 = weak edges, pixels below this are discarded but only kept if connected to strong edges
threshold2 = strong edges, Pixels with gradient magnitude above this value are definitely edges.

NOTE: Pixels between threshold1 and threshold2 are kept only if they are connected to a strong edge.
"""

canny_edge = cv2.Canny(median, 50, 100) # used the already blurred median image, not the original

# cv2.imshow("Original image", img)
# cv2.imshow("Canny Edge Detector", canny_edge)

"""6. Thresholding"""

ret, th = cv2.threshold(contrast_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
kernel = np.ones((3,3), np.uint8)

# cv2.imshow("Original image", img)
# cv2.imshow("Thresholded Image", th)

"""7. Morphological Operations"""

# Erosion
erosion = cv2.erode(th, kernel, iterations=1)

# Dilation
dilated = cv2.dilate(erosion, kernel, iterations=2)

# cv2.imshow("Original image", img)
# cv2.imshow("Erosion Image", erosion)
# cv2.imshow("Erosion + Dilated Image", dilated)

cv2.waitKey(0)
cv2.destroyAllWindows()