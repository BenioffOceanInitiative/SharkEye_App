import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import logging
import time
import math
from segment_anything import sam_model_registry, SamPredictor 

try:
    from segmentation.segmentation_utility import show_mask, show_box
except Exception as e:
    from segmentation_utility import show_mask, show_box

from pathlib import Path
from skimage import measure
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.spatial import ConvexHull
from skimage.measure import label

ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 2688, 1512
ASPECT_RATIO = ORIGINAL_WIDTH / ORIGINAL_HEIGHT
DRONE_ALTITUDE_M = 40
FOV_RADIANS = 1.274090354 # From estimate of 73 degrees 

def convert_yolo(bbox_path):
    """ Return list of bounding boxes coordinates """
    with open(bbox_path, "r") as f:
        bboxes = [] 
        for line in f:
            bbox, x, y, w, h = [float(item) for item in line.split()]
            # Scale bbox
            x *= ORIGINAL_WIDTH
            y *= ORIGINAL_HEIGHT
            w *= ORIGINAL_WIDTH
            h *= ORIGINAL_HEIGHT
            
            x1, y1 = (int(x - w/2), int(y - h/2))
            x2, y2 = (int(x + w/2), int(y + h/2))

            bboxes.append([x1, y1, x2, y2])
        return bboxes

def load_image(image_path):
    """ Load image using CV2 """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def crop_image(image, area):
    x1, y1, x2, y2 = area
    return image[y1:y2, x1:x2]

def run_prediction(image, bbox, draw_mask = False, cropped=False):
    start_time = time.time()

    sam_checkpoint = "src/segmentation/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    if cropped:
        h, w, _ = image.shape
        point_h = int(h / 2)
        point_w = int(w /2 )
        
        input_point = np.array([[point_h, point_w]])
        input_label = np.array([1])

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
            )

    else:
        input_box = np.array(bbox)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
            )
    
    if draw_mask:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        plt.show()
    
    end_time = time.time()
    print("Time taken to compute prediction:", end_time - start_time)
    return masks

def draw_mask(mask, image):
    mask = mask.astype(np.uint8)

    # Choose color
    color = np.array([30, 144, 255], dtype=np.uint8)

    # Create a 3-channel color mask
    color_mask = np.zeros_like(image, dtype=np.uint8)
    for i in range(3):
        color_mask[:, :, i] = mask * color[i]

    # Blend original image with color mask
    overlayed = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

    # Convert to BGR for OpenCV display if image is in RGB
    overlayed_bgr = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
    return overlayed_bgr

def largest_region(mask):
    labeled = label(mask)
    regions, counts = np.unique(labeled, return_counts=True)
    counts[regions == 0] = 0  # background
    largest = regions[np.argmax(counts)]
    return labeled == largest

def find_pixel_length(mask, draw_line=False, viz_name=None):
    """ Takes in a segmentation mask in the form of a boolean numpy array and returns the length of
    the longest line within the mask. If draw_line is True, will display the mask and the calculated line"""
    start_time = time.time()

    mask = np.squeeze(mask) # Adjust dimensions
    cleaned_mask = largest_region(mask)
    points = np.argwhere(cleaned_mask)

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    max_dist = 0
    best_pair = None

    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            d = euclidean(hull_points[i], hull_points[j])
            if d > max_dist:
                max_dist = d
                best_pair = (hull_points[i], hull_points[j])
                    
    longest_line = best_pair
    max_length = max_dist

    # Output
    end_time = time.time()
    print("Longest line segment endpoints:", longest_line)
    print("Length of longest line:", max_length)
    print("Time taken to compute:", end_time - start_time)

    # Visualization
    if draw_line:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mask, cmap='gray')  # Show the mask
        if longest_line is not None:
            (y1, x1), (y2, x2) = longest_line
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1, label='Longest Line')
            ax.scatter([x1, x2], [y1, y2], c='blue', s=3)  # Endpoints
            ax.legend()

        ax.set_title('Longest Line on Mask')
        ax.axis('off')
        plt.savefig(f'./{viz_name}.jpg')
        # plt.show()
    
    return max_length

def calculate_shark_length_from_pixel(shark_pixel_length):
    """Calculate shark length in feet based on pixel_length"""
    long_side = (2 * ASPECT_RATIO * DRONE_ALTITUDE_M * math.tan(FOV_RADIANS / 2))/ np.sqrt(1 + ASPECT_RATIO ** 2) 
    pixel_size_m = long_side / ORIGINAL_WIDTH

    length_m = shark_pixel_length * pixel_size_m 
    return length_m * 3.28084  # Convert meters to feet

if __name__ == "__main__":
    
    if True:
    # if not os.path.exists('./src/segmentation/mask.npy'):
        # sam_checkpoint = "./sam_vit_h_4b8939.pth"
        # model_type = "vit_h"

        # Select image
        image = load_image('./src/segmentation/data/DJI_0091_Trim_1_images_frame113.jpg')
        bboxes = convert_yolo('./src/segmentation/data/DJI_0091_Trim_1_annotations_frame113.txt')[0]

        print("Running prediction on cropped image")
        cropped = crop_image(image, bboxes)
        prediction = run_prediction(image, cropped, cropped=True)
        print(calculate_shark_length_from_pixel(find_pixel_length(prediction, draw_line=True)))

        print(" Running prediction on image with bounding boxes")
        prediction = run_prediction(image, bboxes)
        print(calculate_shark_length_from_pixel(find_pixel_length(prediction, draw_line=True)))

        np.save('./mask.npy', prediction)
    
    else:
        image = load_image('./src/segmentation/data/DJI_0091_Trim_1_images_frame113.jpg')
        print(image)
        mask = np.load('./src/segmentation/mask.npy')

        draw_mask(mask, image)
        # mask = 
        print(calculate_shark_length_from_pixel(find_pixel_length(mask, draw_line=True)))