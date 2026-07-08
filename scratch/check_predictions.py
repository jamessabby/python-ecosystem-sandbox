import cv2 as cv
import numpy as np

# Load model and labels
net = cv.dnn.readNetFromONNX("models/mobilenetv2-7.onnx")

with open("models/imagenet_classes.txt") as f:
    categories = [line.strip() for line in f.readlines()]

img = cv.imread("computer_vision/opencv/photos/cat.jpeg")
if img is None:
    print("Error: Could not load the image.")
    exit()

# Preprocess image with crop=True
blob = cv.dnn.blobFromImage(img, scalefactor=(1.0/255.0), size=(224, 224), swapRB=True, crop=True)
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
blob = (blob - mean) / std

net.setInput(blob)
output = net.forward()

# Print top 5 indices and predictions
top5_indices = np.argsort(output[0])[-5:][::-1]
print("Top 5 predictions with crop=True:")
for idx in top5_indices:
    print(f"Index: {idx}, Label: {categories[idx]}, Score: {output[0][idx]:.4f}")
