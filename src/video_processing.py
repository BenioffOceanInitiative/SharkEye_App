import cv2

def ar_resize(width, height, imgsz_h):
    imgsz_w = imgsz_h * width / height
    return [736, 1280]


def convert_bbox_center_to_corners(x_center, y_center, width, height):
    # Calculate half-width and half-height
    half_width = width / 2
    half_height = height / 2

    # Calculate top-left and bottom-right coordinates
    top_left_x = int(x_center - half_width)
    top_left_y = int(y_center - half_height)
    bottom_right_x = int(x_center + half_width)
    bottom_right_y = int(y_center + half_height)

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def draw_bounding_box(image, bbox, object_id, confidence):
    x, y, width, height = bbox

    # Convert float coordinates to integers
    x, y, width, height = int(x), int(y), int(width), int(height)

    xtl, ytl, xbr, ybr = convert_bbox_center_to_corners(x, y, width, height)

    # Draw bounding box on the image
    color = (0, 255, 0)  # Green color for the bounding box
    thickness = 2
    cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, thickness)

    # Display object information
    text = f"Object ID: {object_id}, Confidence: {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = (xbr - 30, ybr + 30)
    cv2.putText(image, text, text_position, font, font_scale, color, font_thickness)
    return image 


def save_image_with_bbox(image, filename, path):
    filepath = path + '/' + filename + '.jpeg'
    cv2.imwrite(filepath, image)
