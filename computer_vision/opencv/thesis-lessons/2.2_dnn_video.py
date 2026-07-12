import cv2 as cv
import numpy as np

net = cv.dnn.readNetFromONNX("models/mobilenetv2-7.onnx")

with open("models/imagenet_classes.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]

cap = cv.VideoCapture("computer_vision/opencv/videos/dog.mp4")

font = cv.FONT_HERSHEY_COMPLEX
font_scale = 1
font_color = (0, 255, 0) #  BGR
thickness = 2

width = 640
height = 480
new_dimension = (width, height)

while cap.isOpened():
    ret, frame = cap.read()

    rescaled_frame = cv.resize(frame, new_dimension, interpolation=cv.INTER_AREA)
    
    if not ret:
        print("Video stream finished.")
        break

    blob = cv.dnn.blobFromImage(rescaled_frame, scalefactor=1.0/255.0, size=(244,244), swapRB=True, crop=False)

    net.setInput(blob)

    output = net.forward()

    winning_index = np.argmax(output)

    prediction_text = categories[winning_index].upper()

    cv.putText(rescaled_frame, f"{prediction_text}", (30, 50), font, font_scale, font_color, thickness)  

    cv.imshow("Live Deep Learning Feed", rescaled_frame)

    if cv.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()