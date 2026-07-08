import cv2 as cv

"""Reading Images"""

img = cv.imread("C:/Users/James/OneDrive/Documents/Code/tutorials/python/freecodecamp/images/cat.jpg")

# cv.imshow("Cat", img)

# cv.waitKey(0)
# cv.destroyAllWindows()

"""Reading Videos"""

capture = cv.VideoCapture(0)   # 0 means default camera

while True: 
    isTrue, frame = capture.read()
    cv.imshow('Me', frame)
    
    if cv.waitKey(20) & 0xFF==ord('q'):
        break
    
capture.release()
cv.destroyAllWindows()