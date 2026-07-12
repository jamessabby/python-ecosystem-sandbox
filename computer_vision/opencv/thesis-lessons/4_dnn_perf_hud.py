import cv2 as cv
import numpy as np
import time 

net = cv.dnn.readNetFromONNX("models/mobilenetv2-7.onnx")
width = 640
height = 480
new_dimension = (width, height)

with open("models/imagenet_classes.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]

cap = cv.VideoCapture("computer_vision/opencv/videos/dog.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    rescaled_frame = cv.resize(frame, new_dimension, interpolation=cv.INTER_AREA)


    if not ret:
        print("Video live stream has ended.")
        break

    # get the actual dimensions of frame dynamicallly
    height, width, _ = rescaled_frame.shape      # underscore means there's a data (ex: channel) going to be returned there but will not get used

    # start the timer for this frame
    start_time = time.time()    

    blob = cv.dnn.blobFromImage(rescaled_frame, scalefactor=1.0/255.0, size=(224, 244), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()
    winner_index = np.argmax(output)
    prediction_label = categories[winner_index].upper

    # start time and calculate the fps
    end_time = time.time()
    time_taken = end_time - start_time
    fps = 1.0 / time_taken

    # Drawing the graphics HUD and bounding box

    # 1. Draw a simulated target box right at the center of the frame
    # calculate the screen coordinates
    box_x1, box_y1 = int(width * 0.25), int(height * 0.25)              # top-left corner
    box_x2, box_y2 = int(width * 0.75), int(height * 0.75)              # bottom-right corner

    # draw a bounding box rectangle
    cv.rectangle(rescaled_frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 0), thickness=3)
    
    # draw the background layout bar for text readabity at the top
    cv.rectangle(rescaled_frame, (0, 0), (width, 60), (40, 40, 40), -1)          # the -1 fills the rectangle solid

    # print the live AI prediction on the top left
    cv.putText(rescaled_frame, f"FPS: {fps:.2f}", (width - 160, 40), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 165, 255), 2)

    cv.imshow("Live Video Feed", rescaled_frame)

    if cv.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


# This FPS is not the video's original playback speed; it is your computer's 
# processing speed. In computer vision applications (like autonomous driving or 
# security cameras), developers track this HUD (Heads-Up Display) metric to know if their AI model 
# is fast enough to process live video feeds in real-time without lagging behind.