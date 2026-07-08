# EDGE DETECTOR

from skimage import io

from matplotlib import pyplot as plt

img = io.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/angry-doc.jpg", as_gray = True)

from skimage.feature import canny

edge_canny = canny(img, sigma=1)    # Sigma controls how much the image is blurred before edge detection.
# plt.imshow(edge_canny)

# Deconvolution - requires original image and a point spread function and it calculates the deconvolution, it just sharpen
#                  the image
#               - is the process of undoing blur in an image.

from skimage import restoration
import numpy as np

# Point Spread Function (PSF) - describes how a single point of light gets blurred by an imaging system.
#                             - PSF models the blur that deconvolution tries to undo.
psf = np.ones((3,3)) / 9    # Normalizing the 3x3 matrix with 9 1's so the image energy remains the same
#                           1/9  1/9  1/9
#                           1/9  1/9  1/9
#                           1/9  1/9  1/9


# Deconvolution removes blur, Wiener deconvolution removes blur while controlling noise.
deconvolved, _ = restoration.unsupervised_wiener(img, psf)


plt.imsave("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/deconvolved.jpg", deconvolved, cmap='gray')