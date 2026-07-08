import cv2
import matplotlib.pyplot as plt

# 1. Load the XML file using OpenCV's classifier
# Use the same path you fixed earlier
trained_file = r"C:\Users\James\OneDrive\Documents\Code\tutorials\python\scikit-image\xml_files\haarcascade_eye.xml"
detector = cv2.CascadeClassifier(trained_file)

# 2. Load the image
# Note: Astronaut is built into skimage, so we can still grab it from there
from skimage import data
img = data.astronaut()

# 3. Convert to Grayscale
# OpenCV performs much better (and faster) on grayscale images
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 4. Detect objects
# We adjust minSize to (10, 10) so it doesn't skip the tiny eyes!
detected = detector.detectMultiScale(
    gray_img, 
    scaleFactor=1.1, 
    minNeighbors=5, 
    minSize=(10, 10)
)

# 5. Draw the results and show
for (x, y, w, h) in detected:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

plt.imshow(img)
plt.axis('off')
plt.show()