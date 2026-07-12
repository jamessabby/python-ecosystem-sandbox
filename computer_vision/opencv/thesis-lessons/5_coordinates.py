import cv2 as cv
import numpy as np

img = cv.imread("computer_vision/opencv/photos/cat.jpeg")

if img is None:
    print("Image failed to load.")
    exit()

# coordinate boundaries 
cat_face = {'x': 200, 'y': 70, 'w': 210, 'h': 170}

# numpy slicing
x, y, w, h = cat_face['x'], cat_face['y'], cat_face['w'], cat_face['h']
roi_crop = img[y:y+h, x:x+w]

print(f"Original image shape: {img.shape}")
print(f"Cropped image shape: {roi_crop.shape}")

cv.imshow("Original image", img)
cv.imshow("Cropped image", roi_crop)

cv.waitKey(0)
cv.destroyAllWindows()