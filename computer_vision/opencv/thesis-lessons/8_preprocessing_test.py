import cv2 as cv

img = cv.imread("computer_vision/opencv/photos/cat.jpeg")

if img is None:
    print("File cannot be found.")
    exit()

print("="*50)
print(f"📥 Raw Image Shape: {img.shape}")

rotated_img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
print(f"🔄 Rotated Image Shape: {rotated_img.shape}")

scale = 0.70
new_width = int(rotated_img.shape[1] * scale)
new_height = int(rotated_img.shape[0] * scale)
rescaled_img = cv.resize(rotated_img, (new_width, new_height), interpolation=cv.INTER_AREA)

print(f"Rescaled image shape: {rescaled_img.shape}")
print("="*50)

cv.imshow("Raw image", img)
cv.imshow("Rotated image", rotated_img)
cv.imshow("Rescaled image", rescaled_img)

cv.waitKey(0)
cv.destroyAllWindows()