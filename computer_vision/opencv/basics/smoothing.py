import cv2 as cv

img = cv.imread('photos/cats.jpg')
cv.imshow('Cats', img)

# ---- BLURS ----- 
# Averaging
avg = cv.blur(img, (7,7))
cv.imshow('Averaging Blur', avg)

# Gaussian 
gauss = cv.GaussianBlur(img, (7,7), 0) # parameters: image, tuple kernel size, and standard deviation
cv.imshow('Gaussian Blur', gauss)

# Median
median = cv.medianBlur(img, 3) # integer kernel size
cv.imshow('Median Blur', median)

# Bilateral 
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral Blur', bilateral)

cv.waitKey(0)