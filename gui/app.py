import os
import sys

# --- START: Crucial Path Adjustments for 'utils' and 'models' in parent directory ---
# Get the directory where the current script (app.py) is located (e.g., D:\Handwriting\gui)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (e.g., D:\Handwriting)
# This assumes app.py is in D:\Handwriting\gui and utils/models are in D:\Handwriting
project_root = os.path.join(current_script_dir, '..')

# Add the project root directory to Python's system path
sys.path.append(project_root)
# --- END: Crucial Path Adjustments ---


import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
import json
import cv2  # For preprocessing
from PIL import Image

# Import segment_characters and cv2_img_to_base64 from segment_utils
from utils.segment_utils import segment_characters, cv2_img_to_base64


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and classes
try:
    # Use os.path.join for robust path handling
    model_path = os.path.join(project_root, 'models', 'handwritten_mobilenetv2.keras')
    classes_path = os.path.join(project_root, 'models', 'classes.json')

    model = load_model(model_path)
    with open(classes_path, 'r') as f:
        class_names = json.load(f)
    print("Models and class names loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load model or classes.json. Ensure paths are correct and files exist. Error: {e}")
    model = None
    class_names = []


# Define target size for MobileNetV2 input
TARGET_SIZE = (224, 224)

# np_img_to_base64 is kept here as it's only used for plotting/visualizing numpy arrays
# that might not be raw OpenCV images
def np_img_to_base64(img_np):
    # Convert numpy image to base64 PNG string for HTML <img>
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        img_np = (img_np * 255).astype(np.uint8)
    if len(img_np.shape) == 2:  # grayscale
        pil_img = Image.fromarray(img_np)
    else: # RGB/BGR
        if img_np.shape[2] == 3:
            pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(img_np)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64_str

# IMPORTANT: preprocess_single_char_img needs to use cv2_img_to_base64 from segment_utils
def preprocess_single_char_img(char_img_raw_grayscale):
    """
    Preprocesses a single segmented character image (raw grayscale from segment_characters)
    and returns intermediate visualization steps.
    """
    intermediate_steps = {}

    # The raw segment already has black characters on a white background.
    # If your model was trained on black characters on a white background,
    # we should NOT invert it initially.
    img_working = char_img_raw_grayscale.copy()

    # Still generate an inverted image for visualization, but don't use it for the pipeline
    inverted_img_for_display = cv2.bitwise_not(char_img_raw_grayscale)
    intermediate_steps['inverted_img_b64'] = cv2_img_to_base64(inverted_img_for_display)


    # 2. Thresholding
    # Use Otsu's thresholding for the char, but only if it's not a blank image
    if np.min(img_working) == np.max(img_working):
        # If uniform, create a blank image (white background, no char)
        thresh = np.ones_like(img_working) * 255
    else:
        # Use THRESH_BINARY (not THRESH_BINARY_INV) to get black text on white background
        _, thresh = cv2.threshold(img_working, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    intermediate_steps['thresholded_img_b64'] = cv2_img_to_base64(thresh)

    # Find bounding box of the character (for cropping)
    # Ensure there are non-zero pixels to find a contour (i.e., not entirely white/black)
    if np.sum(thresh == 0) == 0: # Check if the thresholded image is entirely white (no black pixels)
        print("WARNING: Thresholded character image is entirely white (no black pixels). Returning blank.")
        blank_img = np.ones(TARGET_SIZE, dtype=np.uint8) * 255 # White blank image
        return {
            "processed_visual": cv2_img_to_base64(blank_img),
            "model_input": np.expand_dims(cv2.cvtColor(blank_img, cv2.COLOR_GRAY2RGB).astype('float32') / 255.0, axis=0),
            "intermediate_steps": {
                'inverted_img_b64': intermediate_steps['inverted_img_b64'],
                'thresholded_img_b64': cv2_img_to_base64(blank_img),
                'padded_img_b64': cv2_img_to_base64(blank_img),
                'resized_img_b64': cv2_img_to_base64(blank_img)
            }
        }
    
    # Use findContours to get the main character contour and its bounding box
    # Using RETR_EXTERNAL and CHAIN_APPROX_SIMPLE to get the outer bounding box
    # Find contours on the inverted thresholded image to detect black pixels correctly if needed
    # Or, if characters are black on white, find contours directly on 'thresh'
    # For black text on white background, contours are found around black areas.
    # Let's re-invert thresh to find contours as white objects on black background, then revert
    contours_for_bbox, _ = cv2.findContours(cv2.bitwise_not(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = 0, 0, thresh.shape[1], thresh.shape[0] # Default to full image if no contours found
    if contours_for_bbox:
        # Get the largest contour (most likely the character)
        largest_contour = max(contours_for_bbox, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)


    if w == 0 or h == 0:
        # Return blank images if no character found to prevent errors
        print("WARNING: Bounding box for character is zero-sized. Returning blank.")
        blank_img = np.ones(TARGET_SIZE, dtype=np.uint8) * 255
        return {
            "processed_visual": cv2_img_to_base64(blank_img),
            "model_input": np.expand_dims(cv2.cvtColor(blank_img, cv2.COLOR_GRAY2RGB).astype('float32') / 255.0, axis=0),
            "intermediate_steps": {
                'inverted_img_b64': intermediate_steps['inverted_img_b64'],
                'thresholded_img_b64': intermediate_steps['thresholded_img_b64'], # Keep original thresholded for display
                'padded_img_b64': cv2_img_to_base64(blank_img),
                'resized_img_b64': cv2_img_to_base64(blank_img)
            }
        }

    # Crop the character from the thresholded image (black on white)
    char = thresh[y:y+h, x:x+w]

    # 3. Padding to square
    size = max(w, h)
    padded = np.ones((size, size), dtype=np.uint8) * 255 # White background
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = char
    intermediate_steps['padded_img_b64'] = cv2_img_to_base64(padded)


    # 4. Resize to target_size (224x224)
    resized = cv2.resize(padded, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    intermediate_steps['resized_img_b64'] = cv2_img_to_base64(resized)


    # 5. Convert to 3 channels (RGB) for MobileNetV2
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # 6. Normalize pixel values to 0-1
    normalized_img = rgb_img.astype('float32') / 255.0

    # For visualization, return the normalized image scaled back to 0-255 uint8
    # Since we aim for black characters on white background visually and for the model.
    # The `processed_visual` should reflect this:
    processed_visual = (normalized_img[:, :, 0] * 255).astype(np.uint8)

    # For model input, add batch dimension
    model_input = np.expand_dims(normalized_img, axis=0)

    return {
        "processed_visual": cv2_img_to_base64(processed_visual), # This is the final visual before model input
        "model_input": model_input,
        "intermediate_steps": intermediate_steps # Return all intermediate steps
    }


def plot_to_base64(fig):
    # Takes a matplotlib figure object
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig) # Close the figure to free memory
    return img_base64

def get_top_predictions_data(pred_probs, class_names, top_n=5):
    """Returns top N predictions as a list of dictionaries."""
    top_indices = np.argsort(pred_probs)[-top_n:][::-1]
    top_data = []
    for i in top_indices:
        # Ensure the index is within the bounds of class_names
        if i < len(class_names):
            top_data.append({
                "label": class_names[i],
                "confidence": float(pred_probs[i]) * 100
            })
        else:
            print(f"Warning: Predicted index {i} out of bounds for class_names (len={len(class_names)}). Skipping.")
            top_data.append({
                "label": "[UNKNOWN]",
                "confidence": 0.0
            })
    return top_data

def plot_prediction_bar(pred_probs, class_names, top_n=10):
    fig, ax = plt.subplots(figsize=(8, 5))
    top_indices = np.argsort(pred_probs)[-top_n:][::-1]
    # Filter out indices that are out of bounds for class_names
    valid_top_indices = [i for i in top_indices if i < len(class_names)]

    top_probs = pred_probs[valid_top_indices]
    top_labels = [class_names[i] for i in valid_top_indices]

    sns.barplot(x=top_probs, y=top_labels, palette='viridis', ax=ax)
    ax.set_xlabel('Probability')
    ax.set_title('Top Predictions')
    plt.tight_layout()

    return plot_to_base64(fig)

def plot_confusion_like(pred_probs, class_names, top_n=5):
    fig, ax = plt.subplots(figsize=(6, 5)) # Slightly larger for readability
    top_indices = np.argsort(pred_probs)[-top_n:][::-1]
    # Filter out indices that are out of bounds for class_names
    valid_top_indices = [i for i in top_indices if i < len(class_names)]

    labels = [class_names[i] for i in valid_top_indices]

    # Create a diagonal matrix where diagonal elements are the probabilities
    matrix_values = pred_probs[valid_top_indices]
    matrix = np.diag(matrix_values) # Create a diagonal matrix

    # Set up annot_kws to limit decimal places
    sns.heatmap(matrix, annot=True, xticklabels=labels, yticklabels=labels,
                cmap='Blues', fmt='.2f', linewidths=.5, linecolor='black', ax=ax,
                annot_kws={"size": 10}) # Adjust annotation font size

    ax.set_title('Predicted Probability Matrix (Top {})'.format(len(valid_top_indices)))
    ax.set_ylabel('Predicted Class')
    ax.set_xlabel('Predicted Class')
    plt.tight_layout()

    return plot_to_base64(fig)


# --- Function to draw bounding boxes and text on an image ---
def draw_predictions_on_image(image_bgr, char_data_list, original_scale_factor=1):
    """
    Draws bounding boxes and predicted labels on the original image.
    image_bgr: The original full sentence image in BGR format (from cv2.imread without grayscale).
    char_data_list: List of dictionaries, each containing 'bbox' and 'predicted_label'.
    original_scale_factor: If the char_data_list bboxes are for a scaled down image, scale them up.
    """
    img_with_boxes = image_bgr.copy() # Work on a copy

    # Define colors and font
    box_color = (0, 255, 0)  # Green BGR
    text_color = (0, 0, 255) # Red BGR
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    line_thickness = 2

    for char_data in char_data_list:
        if 'bbox' in char_data: # Ensure bbox is present
            x, y, w, h = char_data['bbox']
            # Scale bbox coordinates if image was scaled down before segmentation (not currently done in app.py)
            x = int(x * original_scale_factor)
            y = int(y * original_scale_factor)
            w = int(w * original_scale_factor)
            h = int(h * original_scale_factor)

            label = char_data.get('predicted_label', '?')

            # Draw rectangle
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), box_color, line_thickness)

            # Put text label
            # Adjust text position to be just above the box
            text_x = x
            text_y = y - 10 if y - 10 > 10 else y + h + 20 # Avoid going off top, or place below
            cv2.putText(img_with_boxes, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img_with_boxes


@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize variables to None or empty for the initial GET request
    filename = None
    original_full_sentence_b64 = None
    full_sentence_with_preds_b64 = None
    predicted_sentence = ""
    individual_char_data = []
    intermediate_segmentation_steps = {}
    error_message = None # New variable for displaying errors

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            error_message = "No file selected."
            print(error_message) # Debug print
        else:
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            # Read original image in color for drawing and grayscale for processing
            original_img_bgr = cv2.imread(filepath)
            original_img_grayscale = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # IMPORTANT: Check if the images were loaded successfully
            if original_img_bgr is None or original_img_grayscale is None:
                error_message = f"Could not load image from '{filename}'. It might be corrupted or not a valid image format. Please try another image."
                print(f"ERROR: {error_message}")
                return render_template('index.html', error_message=error_message)

            # Convert original image to base64 for display
            original_full_sentence_b64 = cv2_img_to_base64(original_img_bgr)

            # Ensure model and class_names are loaded before proceeding
            if model is None or not class_names:
                error_message = "Model or class definitions not loaded. Please check server logs for errors during startup."
                print(f"ERROR: {error_message}")
                return render_template('index.html', error_message=error_message)


            # Segment characters and get bounding boxes AND intermediate segmentation steps
            # segment_characters now returns a tuple: (list_of_char_segments, dict_of_steps)
            segmented_data_list, intermediate_segmentation_steps = segment_characters(original_img_grayscale)


            char_bboxes_for_overlay = [] # To collect bboxes and labels for the overlay image
            max_chars_to_display = 20 # Limit to avoid overwhelming the browser

            if not segmented_data_list: # Check if segmentation returned no characters
                print("DEBUG: No characters were segmented from the image. Predicted sentence will be empty.")
                predicted_sentence = "[No characters detected]" # Provide a default message
            else:
                for i, (char_img_raw, bbox) in enumerate(segmented_data_list): # Unpack the tuple (image, bbox)
                    if i >= max_chars_to_display:
                        print(f"Warning: Limiting processing to first {max_chars_to_display} characters for display.")
                        break

                    raw_segment_b64 = cv2_img_to_base64(char_img_raw)
                    processed_char_outputs = preprocess_single_char_img(char_img_raw)
                    processed_visual_b64 = processed_char_outputs["processed_visual"]
                    model_input_for_char = processed_char_outputs["model_input"]
                    intermediate_preprocessing_steps = processed_char_outputs["intermediate_steps"]

                    # --- Prediction handling: Ensure model input is correct ---
                    if model_input_for_char is None or model_input_for_char.size == 0 or model_input_for_char.shape[-1] != 3:
                        print(f"Warning: Character {i+1} has invalid model input shape or content. Skipping prediction.")
                        pred_label = "[Invalid Char]"
                        pred_confidence = 0.0
                        top_predictions_list = []
                        barplot_b64 = None
                        confusion_matrix_b64 = None
                    else:
                        try:
                            preds = model.predict(model_input_for_char, verbose=0)
                            pred_probs = preds[0]

                            top_predictions_list = get_top_predictions_data(pred_probs, class_names, top_n=5)

                            pred_label = top_predictions_list[0]['label'] if top_predictions_list else "[UNKNOWN]"
                            pred_confidence = top_predictions_list[0]['confidence'] if top_predictions_list else 0.0

                            barplot_b64 = plot_prediction_bar(pred_probs, class_names, top_n=10)
                            confusion_matrix_b64 = plot_confusion_like(pred_probs, class_names, top_n=5)
                        except Exception as e:
                            print(f"ERROR: Prediction failed for character {i+1}: {e}")
                            pred_label = "[Error Pred]"
                            pred_confidence = 0.0
                            top_predictions_list = []
                            barplot_b64 = None
                            confusion_matrix_b64 = None
                    # --- End Prediction handling ---

                    predicted_sentence += pred_label

                    individual_char_data.append({
                        "char_num": i + 1,
                        "raw_segment_img": raw_segment_b64,
                        "processed_char_img": processed_visual_b64,
                        "predicted_label": pred_label,
                        "confidence": f"{pred_confidence:.2f}",
                        "top_predictions": top_predictions_list,
                        "barplot_img": barplot_b64,
                        "confusion_matrix_img": confusion_matrix_b64,
                        "intermediate_preprocessing": intermediate_preprocessing_steps,
                        "bbox": bbox # Store bbox for overlay visualization
                    })

                    # Collect bbox and label for the full sentence overlay
                    char_bboxes_for_overlay.append({
                        "bbox": bbox,
                        "predicted_label": pred_label
                    })

            # Generate the full sentence image with bounding boxes and predictions
            if original_img_bgr is not None: # Only draw if original image was loaded successfully
                full_sentence_with_preds_b64 = cv2_img_to_base64(
                    draw_predictions_on_image(original_img_bgr, char_bboxes_for_overlay)
                )
            else:
                full_sentence_with_preds_b64 = None # No overlay if original image failed

            # Debugging prints
            print(f"\n--- DEBUGGING RENDER DATA ---")
            print(f"Filename: '{filename}'")
            print(f"Predicted Sentence: '{predicted_sentence}' (Length: {len(predicted_sentence)})")
            print(f"Number of individual character data entries: {len(individual_char_data)}")
            print(f"Segmentation steps keys: {intermediate_segmentation_steps.keys() if intermediate_segmentation_steps else 'None'}")
            print(f"--- END DEBUGGING RENDER DATA ---\n")


    return render_template('index.html',
                           filename=filename,
                           original_full_sentence_img=original_full_sentence_b64,
                           full_sentence_with_preds_img=full_sentence_with_preds_b64,
                           predicted_sentence=predicted_sentence,
                           individual_char_data=individual_char_data,
                           intermediate_segmentation_steps=intermediate_segmentation_steps,
                           error_message=error_message # Pass error message to template
                           )

if __name__ == '__main__':
    app.run(debug=True)

