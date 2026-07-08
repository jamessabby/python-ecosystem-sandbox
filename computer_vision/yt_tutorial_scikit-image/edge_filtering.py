from skimage import io

from matplotlib import pyplot as plt

img = io.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/angry-doc.jpg", as_gray=True)


# EDGE FILTERING

from skimage.filters import roberts, sobel, scharr, prewitt

edge_roberts = roberts(img)
edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)

# # Creates a 2×2 grid of plots, all sharing the same axis (x or y) scale.
# # figsize = 8 inches wide × 8 inches tall (how big the image appears)
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, 
                         figsize=(8, 8))



ax = axes.ravel()       # Flattens the 2D axes array into a 1D list for easy indexing.
                        # before ravel(), ax =
                        #                   [[ax00, ax01],
                        #                   [ax10, ax11]]
                        
                        # after ravel(),  ax = [ax00, ax01, ax10, ax11]


ax[0].imshow(img, cmap=plt.cm.gray)     # Displays the original image in grayscale.
ax[0].set_title('Original Image')       # Sets the title for the first plot.

ax[1].imshow(edge_roberts, cmap=plt.cm.gray)
ax[1].set_title('Roberts edge detection')

ax[2].imshow(edge_sobel, cmap=plt.cm.gray)
ax[2].set_title('Sobel')

ax[3].imshow(edge_scharr, cmap=plt.cm.gray)
ax[3].set_title('Scharr')

    
for a in ax:
    a.axis('off')       # Removes axis ticks and borders from all plots.
    
    
plt.tight_layout()      # Automatically adjusts spacing to prevent overlap. Less empty padding, subplots are closer
plt.show()