import cv2

imagePath = "C:/Users/James/OneDrive/Documents/Code/tutorials/python/scikit-image/images/input_image.jpg"

img= cv2.imread(imagePath)

# print(img.shape) # (height, width, channels)

# NOTE: Note: OpenCV uses BGR (Blue, Green, Red) instead of the standard RGB.
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print(gray_image.shape)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
    
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x,   y), (x + w, y + h), (0, 255, 0), 4 )
    
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')