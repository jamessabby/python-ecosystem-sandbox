import cv2 as cv
import numpy as np

img = cv.imread('photos/makati.jpg')
cv.imshow('Original', img)

# 1. Translation
def translate(img, x, y):
  transMat = np.float32([[1,0,x], [0,1,y]])
  dimensions = (img.shape[1], img.shape[0])  # height, width 
  return cv.warpAffine(img, transMat, dimensions)

# -x = left
# -y = up
# x = right
# y = down

translated = translate(img, 100, 100) # shift to right and down
#cv.imshow('Translated', translated) 

translated2 = translate(img, -100, -100) # shift to right and down
#cv.imshow('Translated2', translated2) 

# 2. Rotation

def rotate(img, angle, rotPoint=None):
  (height,width) = img.shape[:2]
  
  if rotPoint is None:
    rotPoint = (width//2, height//2)
  
  rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)
  dimensions = (width, height)
  
  return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
#cv.imshow('Rotated', rotated)

rotated_rotated = rotate(rotated, -45)
#cv.imshow('Rotated rotated', rotated_rotated)

# flip
flip = cv.flip(img, -1) # 0 for vertical, 1 for horizontal, -1 flip both vertical and horizontal
#cv.imshow('Flipped', flip)

# crop
cropped = img[200:400, 300:400]
cv.imshow('cropped', cropped)

cv.waitKey(0)