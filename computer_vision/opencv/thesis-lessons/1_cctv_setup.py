import cv2 as cv

# camera details
username = "SpotItCamera"
password = "spotittapo232"
camera_ip = "192.168.18.11"

# Port 554 = universal, internationally standardized network port for RTSP (Real-Time Streaming Protocol).
# when there's a port 554 in python opencv code, it pulls the raw video data stream
# stream1 = gets the high quality video channel, stream2 is the low quality one
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:554/stream1"

cap = cv.VideoCapture(rtsp_url)

# continuous video loop
# handles the error when camera fails
while True:
    # ret = returns boolean value indicating success (True) or failure (False) in reading the frame
    # frame = the image itself
    ret,frame = cap.read()

    # if ret is false, break
    if ret is not True:
        print("Connection failed! Check your IP, username, password, or the local WiFi network")
        break

    # scale down the preview window to 50% to fit in the screen
    # (0, 0) means “I’m not giving an exact width and height. Instead, use the scale factors fx and fy to compute the new size.” 
    # fx and fy = 0.5 means resize the height and width 50% from the original
    small_frame = cv.resize(frame, (0,0), fx = 0.5, fy = 0.5)

    cv.imshow("Live feed from Tapo", small_frame)

    # check if q was pressed to stop the video
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
print("Stream connection has closed cleanly")