import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.segment_utils import segment_characters

# Paths
MODEL_PATH = 'D:/Handwriting/models/handwritten_mobilenetv2.keras'
CLASSES_JSON = 'D:/Handwriting/models/classes.json'
SENTENCE_IMAGE_PATH = 'EFG.png'

# Load model and class names
mobilenetv2_model = load_model(MODEL_PATH)
with open(CLASSES_JSON, 'r') as f:
    class_names = json.load(f)

# Define target size for MobileNetV2 input
TARGET_SIZE = (224, 224)

def preprocess_char_img(char_img_raw_grayscale):
    """
    Preprocesses a single segmented character image (raw grayscale from segment_characters).
    This function performs:
    1. Inverting colors (if segment_characters output is white on black, and training is black on white).
    2. Thresholding to ensure clean black character on white background.
    3. Finding bounding box of the character.
    4. Padding to a square image, maintaining aspect ratio.
    5. Resizing to the target_size (224, 224) for MobileNetV2.
    6. Converting to RGB and normalizing to 0-1.
    """
    inverted_img = cv2.bitwise_not(char_img_raw_grayscale)
    _, thresh = cv2.threshold(inverted_img, 127, 255, cv2.THRESH_BINARY)

    temp_for_bbox = cv2.bitwise_not(thresh)
    x, y, w, h = cv2.boundingRect(temp_for_bbox)

    if w == 0 or h == 0:
        print(f"Warning: Bounding box for character was empty. Returning blank image.")
        blank_img = np.ones(TARGET_SIZE, dtype=np.uint8) * 255
        return np.expand_dims(cv2.cvtColor(blank_img, cv2.COLOR_GRAY2RGB).astype('float32') / 255.0, axis=0)

    char = thresh[y:y+h, x:x+w]

    size = max(w, h)
    padded = np.ones((size, size), dtype=np.uint8) * 255
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = char

    resized = cv2.resize(padded, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    normalized_img = rgb_img.astype('float32') / 255.0
    input_img = np.expand_dims(normalized_img, axis=0)

    return input_img

def main():
    os.makedirs('segment', exist_ok=True)

    img = cv2.imread(SENTENCE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found at {SENTENCE_IMAGE_PATH}")
        return

    char_images_raw = segment_characters(img)
    print(f"Segmented {len(char_images_raw)} characters.")

    if char_images_raw:
        plt.figure(figsize=(10, 4))
        for i, char_img in enumerate(char_images_raw):
            plt.subplot(1, len(char_images_raw), i + 1)
            plt.imshow(char_img, cmap='gray')
            plt.axis('off')
            plt.title(f'Raw Char {i + 1}')
        plt.tight_layout()
        plt.savefig('segment/segmentation_raw_result.png')
        plt.close()
    else:
        print("No characters segmented.")

    predicted_sentence = ""
    processed_char_visuals = []

    for i, char_img_raw in enumerate(char_images_raw):
        char_input_for_model = preprocess_char_img(char_img_raw)

        if char_input_for_model.shape == (1, TARGET_SIZE[0], TARGET_SIZE[1], 3):
            processed_char_visuals.append((char_input_for_model[0] * 255).astype(np.uint8))
        else:
            print(f"Skipping visualization for character {i+1} due to preprocessing issue.")

        preds = mobilenetv2_model.predict(char_input_for_model)
        pred_index = np.argmax(preds[0])
        pred_char = class_names[pred_index]

        predicted_sentence += pred_char

        print(f"\n--- Predictions for Char {i+1} ---")
        print(f"Top prediction: {pred_char} (Confidence: {preds[0][pred_index]*100:.2f}%)")

        # Sort predictions by confidence for better readability
        sorted_indices = np.argsort(preds[0])[::-1] # Get indices that would sort in descending order
        
        # Print top N predictions, or all if N is large
        num_to_print = min(5, len(class_names)) # Print top 5, or all if less than 5 classes
        print(f"Other top {num_to_print-1} predictions:")
        for k in range(1, num_to_print): # Start from 1 to skip the top prediction already printed
            idx = sorted_indices[k]
            print(f"  {class_names[idx]}: {preds[0][idx]*100:.2f}%")

        # Or, if you want ALL classes, uncomment this:
        # print("All class confidences:")
        # for idx, confidence in enumerate(preds[0]):
        #     print(f"  {class_names[idx]}: {confidence*100:.2f}%")


    print("\nPredicted Sentence:")
    print(predicted_sentence)

    if processed_char_visuals:
        plt.figure(figsize=(10, 4))
        for i, processed_img in enumerate(processed_char_visuals):
            plt.subplot(1, len(processed_char_visuals), i + 1)
            plt.imshow(processed_img, cmap='gray')
            plt.axis('off')
            plt.title(f'Processed Char {i + 1}')
        plt.tight_layout()
        plt.savefig('segment/segmentation_processed_result.png')
        plt.close()
    else:
        print("No processed images to display.")


if __name__ == "__main__":
    main()