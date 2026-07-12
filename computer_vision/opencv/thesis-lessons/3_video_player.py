import cv2 as cv

cap = cv.VideoCapture("computer_vision/opencv/videos/dog.mp4")

new_width = 640
new_height = 480
new_dimension = (new_width, new_height)

if not cap.isOpened():
    print("Video ended or failed to load")
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Video streamed finish")
        break

    rescaled_frame = cv.resize(frame, new_dimension, interpolation=cv.INTER_AREA)

    cv.imshow("Video sample", rescaled_frame)
    if cv.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
