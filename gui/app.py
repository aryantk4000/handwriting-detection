import os
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

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and classes
model = load_model('D:/Handwriting/models/handwritten_mobilenetv2.keras')
with open('D:/Handwriting/models/classes.json', 'r') as f:
    class_names = json.load(f)

def np_img_to_base64(img_np):
    # Convert numpy image to base64 PNG string for HTML <img>
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)  # If normalized float, scale back
    if len(img_np.shape) == 2:  # grayscale
        pil_img = Image.fromarray(img_np)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64_str

def cv2_img_to_base64(img):
    # img assumed to be numpy uint8 in BGR or grayscale
    if len(img.shape) == 2:
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64_str

def preprocess_steps(img_path):
    """
    Perform all preprocessing steps, return dictionary of base64 images for display
    and numpy array for model input.
    """

    # Load original image as OpenCV BGR and PIL for different steps
    original_pil = Image.open(img_path).convert("RGB")
    original_cv = cv2.imread(img_path)

    # Resize the original PIL image to 224x224 for model input (final step)
    resized_pil = original_pil.resize((224, 224))
    resized_np = np.array(resized_pil)

    # Grayscale (OpenCV)
    gray = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)

    # Blurred
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholded (Otsu's binarization)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Padded - make image square by padding shorter side with black pixels
    h, w = thresholded.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = thresholded

    # Normalized resized image for model input (224x224x3)
    # Convert padded (grayscale) to 3-channel by stacking
    padded_3ch = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    normalized = cv2.resize(padded_3ch, (224, 224))
    normalized = normalized.astype('float32') / 255.0

    # Helper to convert PIL image to base64 string
    def pil_to_base64(pil_img):
        buff = BytesIO()
        pil_img.save(buff, format='PNG')
        return base64.b64encode(buff.getvalue()).decode('utf-8')

    images = {
        "original": pil_to_base64(original_pil),
        "grayscale": cv2_img_to_base64(gray),
        "blurred": cv2_img_to_base64(blurred),
        "thresholded": cv2_img_to_base64(thresholded),
        "padded": cv2_img_to_base64(padded),
        "resized": pil_to_base64(resized_pil),
        "normalized": cv2_img_to_base64((normalized * 255).astype(np.uint8)),
    }

    # Prepare the image for model prediction: normalized array with batch dimension
    model_input = np.expand_dims(normalized, axis=0)  # Shape (1, 224, 224, 3)

    return images, model_input

def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return img_base64

def plot_prediction_bar(pred_probs, class_names, top_n=10):
    plt.figure(figsize=(8, 5))
    top_indices = np.argsort(pred_probs)[-top_n:][::-1]
    top_probs = pred_probs[top_indices]
    top_labels = [class_names[i] for i in top_indices]

    sns.barplot(x=top_probs, y=top_labels, palette='viridis')
    plt.xlabel('Probability')
    plt.title('Top Predictions')
    plt.tight_layout()

    return plot_to_base64()

def plot_confusion_like(pred_probs, class_names, top_n=5):
    plt.figure(figsize=(5, 4))
    top_indices = np.argsort(pred_probs)[-top_n:][::-1]
    labels = [class_names[i] for i in top_indices]
    matrix = np.zeros((top_n, top_n))
    np.fill_diagonal(matrix, pred_probs[top_indices])

    sns.heatmap(matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues', fmt='.2f')
    plt.title('Confusion-like Matrix (Top Predictions)')
    plt.ylabel('True (simulated)')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    return plot_to_base64()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            # Get all preprocessing step images and model input
            prep_images, model_input = preprocess_steps(filepath)

            # Predict using model
            preds = model.predict(model_input)
            pred_index = np.argmax(preds[0])
            if pred_index >= len(class_names):
                return f"Error: predicted index {pred_index} out of range for class_names length {len(class_names)}"
            pred_label = class_names[pred_index]
            pred_confidence = float(preds[0][pred_index]) * 100

            # Generate plots
            barplot_base64 = plot_prediction_bar(preds[0], class_names)
            conf_matrix_base64 = plot_confusion_like(preds[0], class_names)

            return render_template('index.html',
                                   filename=filename,
                                   pred_label=pred_label,
                                   pred_confidence=f"{pred_confidence:.2f}",
                                   barplot_img=barplot_base64,
                                   conf_matrix_img=conf_matrix_base64,
                                   # Pass all preprocessing images
                                   original_img=prep_images["original"],
                                   grayscale_img=prep_images["grayscale"],
                                   blurred_img=prep_images["blurred"],
                                   thresholded_img=prep_images["thresholded"],
                                   padded_img=prep_images["padded"],
                                   resized_img=prep_images["resized"],
                                   normalized_img=prep_images["normalized"]
                                   )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
