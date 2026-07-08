import cv2 as cv
import numpy as np

img = cv.imread('photos/cat.jpeg')
cv.imshow('Cat image', img)
print(img.shape)

cropped_img = img[60:250, 250:400]
cv.imshow('Cropped image', cropped_img)

cv.waitKey(0)