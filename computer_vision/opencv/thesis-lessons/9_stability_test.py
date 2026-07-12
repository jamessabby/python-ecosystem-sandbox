import cv2 as cv
import numpy as np

SCENE_MOTION_LIMIT = 0.15
THRESHOLD = 10

# load the orignal ref image
ref_img = cv.imread("computer_vision/opencv/photos/cat.jpeg")
ref_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
ref_blur = cv.GaussianBlur(ref_gray, (21,21), 0)

# mock disrupted frame since we dont have the cctv camera yet
disrupted_img = ref_img.copy()
h, w, _ = disrupted_img.shape
cv.rectangle(disrupted_img, (50, 50), (w - 50, h- 50), (255, 255, 255), -1)

# process the disrupted frame through your pipeline steps
live_gray = cv.cvtColor(disrupted_img, cv.COLOR_BGR2GRAY)
live_blur = cv.GaussianBlur(live_gray, (21, 21), 0)

# subtracts the entire live frame from your pristine baseline laboratory image
full_dif = cv.absdiff(ref_blur, live_blur)

# takes your grayscale difference map (full_diff) and converts it into a clean, black-and-white image mask called a Binary Image.
# cv2.threshold = looks at every pixel in the difference map (0 - no change, 1+ - drastic change)
# THRESHOLD:
#       pixel value < 10 no change => black
#       pixel value > 10 change => white
# _ = cv2.threshold always returns 2 pieces of data (optimal threshold value used and actual output image matrix), we already 
# know threshold = 10 so throwaway the first parameter 
_, full_thr = cv.threshold(full_dif, THRESHOLD, 255, cv.THRESH_BINARY)

total_pixels = w * h
changed_pixels = cv.countNonZero(full_thr)
motion_pct = changed_pixels / total_pixels

print("="*60)
print(f"Total room matrix pixels: {total_pixels}")
print(f"Mutated white pixels counted: {changed_pixels}")
print(f"Computed global motion ratio: {motion_pct:.2f}")

if motion_pct > SCENE_MOTION_LIMIT:
    print("🚨 [STATUS] STATE UNSTABLE! Object tracking skipped to prevent false alarms.")
else:
    print("🟢 [STATUS] STATE STABLE. Proceeding to target ROI zone isolation.")
print("="*60)

cv.imshow("Original reference image", ref_img)
cv.imshow("Disrupted image", disrupted_img)
cv.imshow("Simulated Disruption (Matrix Change)", full_thr)
cv.waitKey(0)
cv.destroyAllWindows()