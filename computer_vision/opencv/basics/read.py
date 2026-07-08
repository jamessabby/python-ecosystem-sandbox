import cv2 as cv

# reading images
# img = cv.imread('photos/cat.jpg')

# cv.imshow('Cat', img)

# cv.waitKey(0) # waits infinite time for keyboard to be pressed


# reading videos
capture = cv.VideoCapture('videos/dog.mp4')

while True:
  isTrue, frame = capture.read()  # grabe the video frame by frame 
  cv.imshow('Video', frame)       # display each frame 

  if cv.waitKey(20) & 0xFF==ord('d'): # if letter 'd' is pressed, end loop 
    break
  
capture.release()
cv.destroyAllWindows()