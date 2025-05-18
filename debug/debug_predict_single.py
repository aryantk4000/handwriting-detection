# debug/debug_predict_single.py

import cv2
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Paths
image_path = 'D:/Handwriting/dataset/train/0/img001-001.png'  # ✅ Pick one known training image
model_path = 'models/handwritten_mobilenetv2.keras'
classes_path = 'models/classes.json'
output_path = 'debug/debug_input.png'

# Load model and class mapping
model = load_model(model_path)
with open(classes_path, 'r') as f:
    class_names = json.load(f)

# Load and preprocess image like in predict_sentence
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess
# Invert, threshold, pad, resize to 224x224, normalize to 0-1, convert to RGB
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find bounding box and pad to square
x, y, w, h = cv2.boundingRect(thresh)
char = thresh[y:y+h, x:x+w]

# Pad to square
size = max(w, h)
padded = np.ones((size, size), dtype=np.uint8) * 0  # Black background
x_offset = (size - w) // 2
y_offset = (size - h) // 2
padded[y_offset:y_offset+h, x_offset:x_offset+w] = char

# Resize and normalize
resized = cv2.resize(padded, (224, 224))
rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
norm = rgb.astype('float32') / 255.0
input_img = np.expand_dims(norm, axis=0)

# Predict
pred = model.predict(input_img)[0]
pred_index = np.argmax(pred)
pred_class = class_names[pred_index]
confidence = pred[pred_index] * 100

print(f"Predicted: {pred_class} ({confidence:.2f}%)")

# Save debug image
cv2.imwrite(output_path, rgb)
print(f"Saved preprocessed image to: {output_path}")

# Show original and processed image
plt.subplot(1, 2, 1)
plt.title("Original Training Image")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Preprocessed for Model")
plt.imshow(resized, cmap='gray')

plt.tight_layout()
plt.show()
