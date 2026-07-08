import urllib.request
import os

# Create a clean 'models' folder right inside your current directory
os.makedirs('models', exist_ok=True)

print("Downloading MobileNetV2 model architecture (ONNX format)...")
model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
urllib.request.urlretrieve(model_url, "models/mobilenetv2-7.onnx")

print("Downloading ImageNet class labels text file...")
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(labels_url, "models/imagenet_classes.txt")

print("Done! Check your sidebar—your 'models' folder is fully loaded.")