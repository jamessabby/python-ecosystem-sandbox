from skimage import io
import matplotlib.pyplot as plt

"""1. Load and display image"""

img = io.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/woman-img.jpg", as_gray=True)

# plt.imshow(img, cmap='gray')
# plt.axis("off")
# plt.show()

"""2. Edge detection"""

# from skimage.filters import sobel

# edges = sobel(img)

# plt.imshow(edges, cmap='gray')
# plt.axis("off")
# plt.show()

"""3. Compare original vs result"""

# fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# ax[0].imshow(img, cmap='gray')
# ax[0].set_title("Original")
# ax[0].axis("off")

# ax[1].imshow(edges, cmap='gray')
# ax[1].set_title("Sobel")
# ax[1].axis("off")

# plt.show()

"""4. Change one parameter, sigma"""

# from skimage.feature import canny

# NOTE:
# Small sigma (e.g. 0.5–1)
# Less blur, more fine details, more noise edges

# Medium sigma (e.g. 1–2)
# Balanced smoothing, clean edges

# Large sigma (e.g. 3+)
# Strong blur, only major edges remain

# edges = canny(img, sigma=3)

# plt.imshow(edges, cmap='gray')
# plt.axis("off")
# plt.show()

"""5. Add one more feature """

# from skimage.transform import rescale

# small = rescale(img, 0.5)

# plt.imshow(small, cmap='gray')
# plt.axis("off")
# plt.show()

"""
Task:
Compare Sobel and Canny edge detection on the same image
"""

from skimage.feature import canny
from skimage.filters import sobel

fig, ax = plt.subplots(1,2, figsize=(8,4))

sobel_img = sobel(img)
canny_img = canny(img)

ax[0].imshow(sobel_img, cmap='gray')
ax[0].set_title("Sobel Image")
ax[0].axis("off")

ax[1].imshow(canny_img, cmap='gray')
ax[1].set_title("Canny Image")
ax[1].axis("off")

plt.show()



