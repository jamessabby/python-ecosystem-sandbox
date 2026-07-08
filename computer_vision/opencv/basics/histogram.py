import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('photos/cats.jpg')
cv.imshow('Image', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray scale', gray)

mask = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

masked = cv.bitwise_and(img, img, mask=mask)

# grayscale histogram
# gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256]) # image is a list 

# plt.figure()
# plt.title('Grayscale histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.show()

plt.figure()
plt.title('Color histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
  hist = cv.calcHist([img], [i], mask, [256], [0, 256])
  plt.plot(hist, color=col)
  plt.xlim([0, 256])


plt.show()
cv.waitKey(0)