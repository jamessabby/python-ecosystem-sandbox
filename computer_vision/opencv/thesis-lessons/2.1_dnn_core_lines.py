import cv2 as cv
import numpy as np

# load the ai model and text label
net = cv.dnn.readNetFromONNX("models/mobilenetv2-7.onnx")

# with = lets you open a file and close automatically afer you use
# f = variable that lets you use the contents inside the block
with open("models/imagenet_classes.txt") as f:
    categories = [line.strip() for line in f.readlines()]   # creates a list fills it ex: ['cat', 'dog', 'towel', ...]

img  = cv.imread("computer_vision/opencv/photos/cat.jpeg")

if img is None:
    print("Error: Could not load the image. Check your filename or path!")
    exit()

# 4 core lines

# 1. pack the data inside the suitcase

# scalefactor = from 0-255 to 0-1 values only
# 224x224 is the designed shape for mobilenetV2
# swapRB = BGR to RGB format of color channels
blob = cv.dnn.blobFromImage(img, scalefactor=(1.0/255.0), size=(224, 224), swapRB=True, crop=True)

# 2. Hand the suitcase to AI's entry gate
net.setInput(blob)

# 3. execute the underlying mathematical computation 
output = net.forward()

# 4. Get the index location of the highest decimal score
winner_index = np.argmax(output)

predicted_label = categories[winner_index]
print(f"AI prediction: {predicted_label.upper()}") 
print(f"Winner index position: {winner_index}")
