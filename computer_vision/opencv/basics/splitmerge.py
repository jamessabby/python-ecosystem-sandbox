import cv2 as cv
import numpy as np

img = cv.imread('photos/bgr_sample.jpg')
cv.imshow('image', img)

b,g,r = cv.split(img)

#cv.imshow('blue', b)
#cv.imshow('green', g)
#cv.imshow('red', r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b,g,r])
#cv.imshow('Merged', merged)

# individual separated color channels
blank = np.zeros(img.shape[:2], dtype='uint8')

blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank, blank, r])

cv.imshow('separated_blue', blue)
cv.imshow('separated_green', green)
cv.imshow('separated_red', red)


cv.waitKey(0)