import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def visualize_detections(image_np, draw_results):
    """
    Visualizes bounding box detections on an image with improved styling.

    Args:
        image_np (np.array): The input image as a NumPy array (OpenCV format, BGR channel order).
        draw_results (list): A list containing three arrays:
                             [boxes_array, scores_array, labels_array].
                             - boxes_array (list): List of bounding boxes, e.g., [[x1,y1,x2,y2], ...].
                                                   Note: 'boxes' should be in [x_min, y_min, x_max, y_max] format.
                             - scores_array (list): List of confidence scores, e.g., [s1, s2, ...].
                             - labels_array (list): List of labels, e.g., ["label1", "label2", ...].
    Returns:
        np.array: The image with visualized detections, as a NumPy array (OpenCV format).
    """

    # Convert the OpenCV image (NumPy array, BGR) to a PIL Image (RGB) for text drawing.
    # PIL offers better font rendering capabilities.
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Define a vibrant color palette for bounding boxes and text backgrounds.
    # These colors will cycle through for different detections.
    colors = [
        (255, 99, 71),   # Tomato
        (60, 179, 113),  # MediumSeaGreen
        (65, 105, 225),  # RoyalBlue
        (255, 215, 0),   # Gold
        (186, 85, 211),  # MediumOrchid
        (0, 206, 209),   # DarkTurquoise
        (255, 140, 0),   # DarkOrange
        (124, 252, 0),   # LawnGreen
        (255, 105, 180), # HotPink
        (75, 0, 130)     # Indigo
    ]

    # Try to load a common TrueType font (like Arial) for better text quality.
    # Fallback to a default PIL font if 'arial.ttf' is not found.
    font = ImageFont.load_default()

    # Unpack the boxes, scores, and labels directly from the draw_results list
    # Assuming draw_results is always [boxes_array, scores_array, labels_array]
    if len(draw_results) != 3:
        print("Error: draw_results must contain exactly three arrays: boxes, scores, and labels.")
        return image_np # Return original image if format is incorrect

    boxes, scores, labels = draw_results

    # Process each individual detection
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Ensure box coordinates are integers for drawing
        x, y, x2, y2 = [int(round(coord, 0)) for coord in box.tolist()]
        score_item = round(score.item(), 3) # Round score for display

        print(f"Detected {label} with confidence {score_item} at location {[x, y, x2, y2]}")

        # Select a color from the palette, cycling through them
        current_color = colors[i % len(colors)]
        text_fill_color = (255, 255, 255) # White text for good contrast on colored backgrounds

        # Draw the bounding box on the PIL image using ImageDraw
        # This ensures the rectangle is drawn on the same image object as the text.
        draw.rectangle([(x, y), (x2, y2)], outline=current_color, width=2) # Thickness 2

        # Prepare the text string including label and score
        display_text = f"{label} ({score_item:.2f})"

        # Calculate text size using PIL's font to determine background rectangle dimensions
        # Using textbbox for modern Pillow versions
        # textbbox returns (left, top, right, bottom) of the text bounding box
        left, top, right, bottom = draw.textbbox((0, 0), display_text, font=font)
        text_width = right - left
        text_height = bottom - top

        # Determine text position to prevent overflow.
        # Default position is slightly above the bounding box.
        text_x = x
        text_y = y - text_height - 5 # 5 pixels padding above text

        # If the text would go above the image boundary (y < 0),
        # place it just inside the top of the bounding box instead.
        if text_y < 0:
            text_y = y + 5 # 5 pixels padding below the top edge of the box

        # Draw a filled rectangle as a background for the text for better readability.
        # The background rectangle uses the same color as the bounding box.
        # Add a small padding around the text.
        bg_x1 = text_x
        bg_y1 = text_y
        bg_x2 = text_x + text_width + 8 # Add padding to width
        bg_y2 = text_y + text_height + 8 # Add padding to height

        # Ensure the background rectangle does not go beyond image boundaries
        bg_x2 = min(bg_x2, image_pil.width)
        bg_y2 = min(bg_y2, image_pil.height)

        draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], fill=current_color)

        # Draw the text on the PIL image
        draw.text((text_x + 4, text_y + 4), display_text, font=font, fill=text_fill_color) # Add padding to text position

    # Convert the modified PIL image (RGB) back to OpenCV format (BGR)
    final_image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return final_image_np
