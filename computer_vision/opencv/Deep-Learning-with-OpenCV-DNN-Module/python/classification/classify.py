import cv2
import numpy as np
import os

# __file__ - absolute path to the script (i.e. file:///c:/Users/James/OneDrive/Documents/Code/tutorials/opencv/Deep-Learning-with-OpenCV-DNN-Module/python/classification/classify.py)
# os.path.abspath() - returns the absolute path to the file
# os.path.dirname() - returns the directory name of the file
# os.chdir() - changes the current working directory to the specified path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ImageNet is a dataset of images and their corresponding class labels.
# use when: : "I need to read data from a file on my hard drive" or 
#               "I need to save data out to a file on my hard drive,"
# Automatic Cleanup: The exact microsecond your code leaves that indented block, 
# Python forcefully destroys the f variable and cleanly snaps the connection to your hard drive shut.
with open('../../input/classification_classes_ILSVRC2012.txt', 'r') as f:
    # f.read(): Reads the entire text file into a single, massive string of characters in your laptop's RAM.
    # .split('\n'): This is the data-transformation step. 
    # The text file contains 1,000 object names, with each name sitting on a brand-new line. In computer systems, 
    # a line break is represented by a hidden character sequence called \n (newline). The .split('\n') 
    # function chops up that single massive text string everywhere a line break occurs.
    image_net_names = f.read().split('\n')
# final class names (just the first word of the many ImageNet names for one image)
#split() method splits the string at each comma and returns a list of strings
# The first element of the list is the class name
class_names = [name.split(',')[0] for name in image_net_names]

# load the neural network model
# cv2.dnn.readNet() - loads the neural network model from the specified file paths
# model - path to the model file
# config - path to the configuration file
# framework - name of the framework (e.g., 'TensorFlow', 'Caffe', 'PyTorch')
model = cv2.dnn.readNet(model='../../input/DenseNet_121.caffemodel', 
                      config='../../input/DenseNet_121.prototxt', 
                      framework='Caffe')


# load the image from disk
image = cv2.imread('../../input/image_1.jpg')

#cv2.dnn.blobFromImage() - creates a blob from an image
# image - image to create a blob from
# scalefactor - scaling factor to apply to the pixel values
# size - size of the blob (width, height)
# mean - mean values to subtract from the pixel values
# Shape: (1, 3, 224, 224) -> 1 image, 3 color maps, 224x224 pixels each
blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), 
                             mean=(104, 117, 123))
# set the input blob for the neural network
model.setInput(blob)
# forward pass image blog through the model
outputs = model.forward()


final_outputs = outputs[0]
# make all the outputs 1D
# Shape: (1000, 1) -> 1,000 rows, 1 column
# [
#   [-2.34],  # Index 0: (Class: goldfish)
#   [ 0.12],  # Index 1: (Class: great white shark)
#   ...
#   [14.85],  # Index 673: (Class: computer mouse) -> The highest value!
#   ...
#   [-5.11]   # Index 999: (Class: toilet paper)
# ]
final_outputs = final_outputs.reshape(1000, 1)

# get the class label
label_id = np.argmax(final_outputs)

# convert the output scores to softmax probabilities
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))

# get the final highest probability
final_prob = np.max(probs) * 100.

# map the max confidence to the class label names
out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}"

# put the class name text on top of the image
cv2.putText(image, out_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
            2)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.imwrite('../../outputs/result_image.jpg', image)
