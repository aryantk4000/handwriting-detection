import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image # Make sure PIL is imported if you use Image.new in cv2_img_to_base64


# Helper function to convert OpenCV image (NumPy array) to base64 string
def cv2_img_to_base64(img):
    # img assumed to be numpy uint8 in BGR or grayscale
    if img is None:
        # Return a small transparent PNG if image is None, to avoid broken image icon
        img_placeholder = Image.new('RGBA', (1, 1), (0, 0, 0, 0)) # 1x1 transparent PNG
        buff = BytesIO()
        img_placeholder.save(buff, format="PNG")
        base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        return base64_str

    if len(img.shape) == 2: # Grayscale
        pil_img = Image.fromarray(img)
    else: # RGB/BGR
        # Ensure image is 3 channels before converting to RGB, or handle RGBA if needed
        if img.shape[2] == 4: # RGBA
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        else: # BGR
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64_str


def segment_characters(image):
    """
    Segments characters from a grayscale image and captures intermediate steps.
    Returns a tuple: (list of (character_image, bbox_coordinates), dict of intermediate_segmentation_steps_b64)
    """
    intermediate_segmentation_steps = {}
    print("DEBUG: Entering segment_characters function.")
    print(f"DEBUG: Input image shape: {image.shape}, dtype: {image.dtype}")


    # 1. Grayscaling (if not already) and capture
    if len(image.shape) == 3:
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_img = image.copy() # Already grayscale

    intermediate_segmentation_steps['grayscale_img_b64'] = cv2_img_to_base64(grayscale_img)
    print("DEBUG: Grayscale image processed.")

    # 2. Binarization (Otsu) and capture
    # Adding a check for the min/max values in grayscale_img
    min_val, max_val = np.min(grayscale_img), np.max(grayscale_img)
    print(f"DEBUG: Grayscale image pixel value range: {min_val}-{max_val}")
    if min_val == max_val:
        print("WARNING: Grayscale image has uniform pixel values. Binarization might fail. Creating blank binary image.")
        # Handle uniform image (e.g., completely black or white)
        binary_img = np.zeros_like(grayscale_img) if min_val < 128 else np.ones_like(grayscale_img) * 255
    else:
        # Use THRESH_BINARY_INV as we expect dark characters on a light background
        _, binary_img = cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    intermediate_segmentation_steps['binary_img_b64'] = cv2_img_to_base64(binary_img)
    print("DEBUG: Binarized image processed.")


    # 3. Find contours and capture visualization
    # Using RETR_EXTERNAL to get only outer contours (e.g., for full words or characters)
    # If characters are connected, RETR_LIST or RETR_TREE might be needed, but start with EXTERNAL
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"DEBUG: Found {len(contours)} initial contours.")

    # Create a blank image to draw contours on for visualization
    contours_visual_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2BGR) # Start with grayscale but make it 3-channel
    cv2.drawContours(contours_visual_img, contours, -1, (0, 255, 0), 2) # Draw contours in green (BGR) with thickness 2
    intermediate_segmentation_steps['contours_img_b64'] = cv2_img_to_base64(contours_visual_img)


    character_segments = []
    image_height, image_width = image.shape[:2] # Get dimensions to normalize thresholds later
    print(f"DEBUG: Image dimensions: {image_width}x{image_height}")

    # Define dynamic thresholds based on image size and observations from debug output
    # Relaxing max_char_height significantly to allow tall contours
    min_char_width = image_width * 0.005 # e.g., 0.5% of image width
    min_char_height = image_height * 0.01 # e.g., 1% of image height
    max_char_width = image_width * 0.75   # Increased from 0.5 to 0.75 to allow wider characters/words
    max_char_height = image_height * 0.95 # **THIS IS THE CRITICAL CHANGE, from 0.5 to 0.95**

    print(f"DEBUG: Contour filtering thresholds: min_w={min_char_width:.2f}, min_h={min_char_height:.2f}, max_w={max_char_width:.2f}, max_h={max_char_height:.2f}")


    for i, contour in enumerate(contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        print(f"DEBUG: Contour {i+1} bbox: (x={x}, y={y}, w={w}, h={h})")


        # Filter out very small or very large (non-character) contours
        # These thresholds are now more permissive for height
        if w < min_char_width or h < min_char_height or w > max_char_width or h > max_char_height:
            print(f"DEBUG: Filtering out contour {i+1} (bbox: {w}x{h}) - outside size limits.")
            continue

        # Extract the character image using the bounding box from the original grayscale image
        char_img = grayscale_img[y:y+h, x:x+w]
        character_segments.append((char_img, (x, y, w, h))) # Return char_img and its bbox

    print(f"DEBUG: Found {len(character_segments)} characters after filtering.")

    # Sort characters from left to right (important for sentence order)
    character_segments.sort(key=lambda item: item[1][0])
    print("DEBUG: Characters sorted by x-coordinate.")

    return character_segments, intermediate_segmentation_steps # Return both segments and steps