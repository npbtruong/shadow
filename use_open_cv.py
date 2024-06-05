import cv2
import numpy as np
import os

def process_image(image_path, output_dir):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    min_area = 0.05 * image.shape[0] * image.shape[1]
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract panels
    for i, contour in enumerate(contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract panel from the image
        panel = image[y:y+h, x:x+w]

        # Save the panel
        cv2.imwrite(os.path.join(output_dir, f'panel_{i}.png'), panel)

# Get a list of all files in the directory
image_dir = 'imgs'
output_dir = 'panels'
image_files = os.listdir(image_dir)

# Process each image file
for image_file in image_files:
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_file)
        process_image(image_path, output_dir)
