import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

from utils.segment_utils import segment_characters  # Your segmentation function

# Paths - adjust as needed
MODEL_PATH = 'D:/Handwriting/models/handwritten_mobilenetv2.keras'
CLASSES_JSON = 'D:/Handwriting/models/classes.json'
SENTENCE_IMAGE_PATH = './0.png'  # Your test image path
OUTPUT_DIR = './segment/segmented_chars_test'

# Create output folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model and classes
model = load_model(MODEL_PATH)
with open(CLASSES_JSON, 'r') as f:
    class_names = json.load(f)

def preprocess_char_img(char_img):
    # Input: grayscale numpy array (H, W)
    # Output: (1, 224, 224, 3) normalized tensor
    img_rgb = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized.astype('float32') / 255.0
    return np.expand_dims(img_normalized, axis=0)

def main():
    img = cv2.imread(SENTENCE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found at {SENTENCE_IMAGE_PATH}")
        return
    
    char_images = segment_characters(img)
    print(f"Segmented {len(char_images)} characters.")
    
    for i, char_img in enumerate(char_images):
        input_tensor = preprocess_char_img(char_img)
        preds = model.predict(input_tensor)
        pred_index = np.argmax(preds[0])
        pred_label = class_names[pred_index]
        pred_confidence = preds[0][pred_index] * 100
        
        print(f"Char {i+1}: Predicted '{pred_label}' with confidence {pred_confidence:.2f}%")
        
        # Save segmented char image (scaled back to 0-255)
        save_img = (char_img * 255).astype('uint8')
        save_path = os.path.join(OUTPUT_DIR, f"char_{i+1}_{pred_label}.png")
        cv2.imwrite(save_path, save_img)
        print(f"  Saved segmented char image to: {save_path}")

if __name__ == "__main__":
    main()
