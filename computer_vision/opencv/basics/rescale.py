import cv2 as cv

# works on images, videos and live video
def rescaleFrame(frame, scale=0.75):
  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)
  dimensions = (width, height)
  
  return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# works only for live video (external camera, wesbcam)
def changeRes(width, height):
  capture.set(3, width)
  capture.set(3, height)

# resized image
img = cv.imread('photos/cat_large.jpg')
resized_image = rescaleFrame(img, scale=0.25)
cv.imshow('Image', img)
cv.imshow("Resized image", resized_image)

# reading videos
capture = cv.VideoCapture('videos/dog.mp4')

while True:
  isTrue, frame = capture.read()
  frame_resized = rescaleFrame(frame, scale=  0.25)
  cv.imshow('Video', frame)
  cv.imshow('Video resized', frame_resized)
  
  if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()
cv.waitKey(0)