import cv2 as cv


# Works for images, videos and live videos
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Works for live videos only
def changeRes(width,height):
    cv.set(3, width)
    cv.set(3, height)
    


img = cv.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/freecodecamp/images/cat.jpg")

capture = cv.VideoCapture("C:/Users/James/OneDrive/Documents/Code/tutorials/python/freecodecamp/videos\dice.mp4")

rescaled_img = rescaleFrame(img)

# cv.imshow('Original', img)
# cv.imshow('Resized', rescaled_img)

# cv.waitKey(0)
# cv.destroyAllWindows()

# while True: 
#     isTrue, frame = capture.read()
    
#     frame_resized = rescaleFrame(frame, scale=.2)
    
#     cv.imshow('Video', frame)
#     cv.imshow('Rescaled video', frame_resized)
    
#     if cv.waitKey(20) & 0xFF==ord('q'):
#         break
    
# capture.release()
# cv.destroyAllWindows()

