import cv2 as cv

img = cv.imread('photos/makati.jpg')
cv.imshow('Makati', img)

# converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

# blur
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
#cv.imshow('Blurred Image', blur)

# edge cascade
canny = cv.Canny(blur, 125, 175)
#cv.imshow('Canny edges', canny)

# dilating the images
dilated = cv.dilate(canny, (7,7), iterations=3)
#cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (3,3), iterations=1)
#cv.imshow('Eroded', eroded)

# resize
resized = cv.resize(img, (600, 500), interpolation=cv.INTER_LINEAR)
#cv.imshow('Resized', resized)

# cropping - images are arrays so we can do array slicing
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)


cv.waitKey(0)