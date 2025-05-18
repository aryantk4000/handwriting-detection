import cv2
import numpy as np

def segment_characters(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    bounding_boxes = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 5 and h > 10:  # basic size filter
            char_img = thresh[y:y+h, x:x+w]
            chars.append(char_img)
            bounding_boxes.append((x, y, w, h))

    # Sort characters left to right based on x
    sorted_chars = [x for _, x in sorted(zip(bounding_boxes, chars), key=lambda b: b[0][0])]

    return sorted_chars
