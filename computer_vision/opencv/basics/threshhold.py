import cv2 as cv

img = cv.imread('photos/cats.jpg')
cv.imshow('Image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# simple thresholding
# pixel > 150 = white
# pixel <= 150 = black 
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
#cv.imshow('Simple threshold', thresh)

# inverse thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
#cv.imshow('Inverse threshold', thresh)

# adaptive thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive threshold', adaptive_thresh)


cv.waitKey(0)