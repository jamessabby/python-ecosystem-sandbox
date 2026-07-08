import cv2 as cv
import numpy as np

img = cv.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/freecodecamp/images/cat.jpg")

blank = np.zeros((500, 500, 3), dtype='uint8')

# 1. Paint the image color
# blank[:] = 0,0, 255

# 2. Draw a rectangle
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1)

cv.imshow('Rectangle', blank)

# 3. Draw a circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255),thickness=-1)
cv.imshow('Circle', blank) 

# 4. Draw a line
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness=3)
cv.imshow('Line', blank) 


cv.waitKey(0)
cv.destroyAllWindows()