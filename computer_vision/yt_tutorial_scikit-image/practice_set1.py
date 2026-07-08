"""
Course: Image Processing
Coverage: Week 2–3
Tools: Python, scikit-image, matplotlib
IDE: Spyder
"""

"""
Problem 1 — Image Loading and Visualization

Objective: Understand how images are loaded and displayed in Python.

Instructions:

1. Load an image from disk using skimage.io.imread.

2. Convert the image to grayscale during loading.

3. Display the image using plt.imshow.

4. Remove axis ticks and labels.

5 Print the image shape and data type.

Expected Outcome:

- You should see a grayscale image.

- You should be able to explain:

    -> What the shape represents

    -> What the data type represents
"""
# Answer: 

# from skimage import io
# from matplotlib import pyplot as plt

# img = io.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/scratch.jpg", as_gray=True)

# plt.imshow(img, cmap='gray')
# plt.axis('off')
# print("Image shape: " + str(img.shape) + '\n' + "Data type: " + str(img.dtype))

"""
Problem 2 — Brightness Adjustment

Objective: Learn how pixel intensity affects image appearance.

Instructions:

1. Create a brighter version of the image by increasing pixel values.

2. Create a darker version of the image by decreasing pixel values.

3. Ensure pixel values remain within a valid range.

4. Display the original, brighter, and darker images side-by-side.

Expected Outcome:

- Visible brightness differences.

- No visual artifacts caused by overflow or underflow
"""

# Answer:
    
# from skimage import io, exposure
# from matplotlib import pyplot as plt

# img = io.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/scratch.jpg", as_gray=True)
# brighter_img = exposure.rescale_intensity(img, in_range=(0, 0.7), out_range=(0,1))
# darker_img = exposure.rescale_intensity(img, in_range=(0, 1), out_range=(0, 0.06))
    

# fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8,8))

# ax = axes.ravel()

# ax[0].imshow(img, cmap='gray')
# ax[0].set_title("Original")

# ax[1].imshow(brighter_img, cmap='gray')
# ax[1].set_title("Brighter version")

# ax[2].imshow(darker_img, cmap='gray')
# ax[2].set_title("Darker version")

# for a in ax:
#     a.axis('off')
    
# plt.show()


"""

"""

