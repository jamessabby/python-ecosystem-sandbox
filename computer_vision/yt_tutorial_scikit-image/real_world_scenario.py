    import matplotlib.pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np

# Quantify the are of a clean image
# Entropy measures texture inside the disk, and a bigger disk means a bigger local area.

img = io.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/scratch.jpg", as_gray=True)
entropy_img = entropy(img, disk(3))
plt.imshow(img, cmap='gray')
plt.axis('off')

# from skimage.filters import try_all_threshold

# fig, ax = try_all_threshold(entropy_img, figsize=(10,8), verbose=False)
# plt.show()


# Thresholding = Turning a grayscale image into a black-and-white image by choosing a cutoff value.

# What happens during thresholding

# 1. You start with a grayscale image (values like 0.0 → 1.0).

# 2. You pick a threshold value.

# 3. Then for every pixel:

#       If pixel value ≥ threshold → white (1)

#       If pixel value < threshold → black (0)

# In your entropy case

# You threshold entropy values, not brightness:

#       High entropy → textured → white

#       Low entropy → smooth → black

# So thresholding here means:

#       “Separate textured regions from smooth regions.”

from skimage.filters import threshold_otsu

thresh = threshold_otsu(entropy_img) 
binary = entropy_img <= thresh
plt.imshow(binary, cmap='gray')

# True pixels / All pixels (true and false combined)
print("The percent bright pixels is: ", (np.sum(binary==1)*100)/(np.sum(binary==1)+(np.sum(binary==0))))

# IMPORTANT NOTE: Quantifying what you're seeing is the essence of any image processing