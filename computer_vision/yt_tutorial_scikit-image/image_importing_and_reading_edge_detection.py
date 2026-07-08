# Importing and reading images

from skimage import io
from matplotlib import pyplot as plt

img = io.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/angry-doc.jpg", as_gray=True)

# EDGE DETECTION

from skimage.transform import rescale, resize, downscale_local_mean

rescaled_img = rescale(img, 1.0/4.0, anti_aliasing=True)    # same pic, fewer pixels
resized_img = resize(img, (50, 50))                  # fixed output size, proportions not guaranteed. 
downscaled_img = downscale_local_mean(img, (4,3)) # Downscaling = averaging pixels → blocky, pixelated look.


plt.imshow(downscaled_img)   







