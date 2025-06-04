import sys
import os
import cv2
import pandas as pd
import albumentations
import random
import matplotlib.pyplot as plt

from pathlib import Path

def load_image(image_path):
    """ Load image using CV2 """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_yolo_labels(bbox_path):
    with open(bbox_path, "r") as f:
        bboxes = [] 
        labels = []
        for line in f:
            label, x, y, w, h = [float(item) for item in line.split()]
            bboxes.append([x, y, w, h])
            labels.append(int(label))
        
        return {
            'bboxes': bboxes,
            'labels': labels
        }
    
def reconstruct_yolo_labels(bbox_dict: dict):
    results = []

    for label in bbox_dict['labels']:
        bbox = bbox_dict['bboxes'][label]
        bbox.insert(label, 0)

def get_area(bbox_path):
    """ Return area of """
    with open(bbox_path, "r") as f:
        bboxes = {} 
        for line in f:
            bbox, _, _, w, h = [float(item) for item in line.split()]
            bboxes[bbox] = w*h
        return(bboxes)
  
def count_detections(bbox_path):
    with open(bbox_path, "r") as f:
        print(len([line for line in f]))

def find_images_in_path(path: Path, n: int = 0):
    base_path = path
    annotations = base_path / "annotations"
    images = base_path / "images"

    data = []
    for anno_file in annotations.glob("*.txt"):
        areas = get_area(anno_file)
        if len(areas) == 1:
            image_file = images / (anno_file.stem + ".jpg")
            data.append((image_file, areas[0]))

    # Create dataframe of image paths with areas
    df = pd.DataFrame(data, columns=["frame", "areas"]).sort_values("areas", ascending=False)
    
    # Return dataframe with specified number of entries
    if n > 0:
        df = df[:n]
    return df 


def crop_image_random_with_bbox(image_path, bbox_path, image_output: Path, anno_output: Path, scale_by: float = 0.5):
    image = load_image(image_path)  # RGB np.array
    H, W, _ = image.shape

    bbox_dict = read_yolo_labels(bbox_path)
    bboxes = bbox_dict['bboxes']
    labels = bbox_dict['labels']

    # Convert YOLO bboxes to pixel format
    abs_bboxes = []
    for bx, by, bw, bh in bboxes:
        x_center = bx * W
        y_center = by * H
        width = bw * W
        height = bh * H
        abs_bboxes.append([x_center, y_center, width, height])

    crop_h = int(H * scale_by)
    crop_w = int(W * scale_by)

    # Step 1: randomly choose a bbox to center around
    chosen_idx = random.randint(0, len(abs_bboxes) - 1)
    x_center, y_center, box_w, box_h = abs_bboxes[chosen_idx]

    # Step 2: randomly jitter the crop window around this bbox
    min_x = max(int(x_center - crop_w + box_w / 2), 0)
    max_x = min(int(x_center - box_w / 2), W - crop_w)
    min_y = max(int(y_center - crop_h + box_h / 2), 0)
    max_y = min(int(y_center - box_h / 2), H - crop_h)

    if min_x > max_x or min_y > max_y:
        raise ValueError("Cannot find valid crop region that includes the bbox fully.")
    
    # Corners of crop
    x1 = random.randint(min_x, max_x)
    y1 = random.randint(min_y, max_y)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Crop the image
    cropped_image = image[y1:y2, x1:x2]

    # Step 3: convert all bboxes to relative to crop
    new_bboxes = []
    new_labels = []
    for (bx, by, bw, bh), label in zip(abs_bboxes, labels):
        # Pixel coords of original bbox 
        left = bx - bw / 2
        right = bx + bw / 2
        top = by - bh / 2
        bottom = by + bh / 2

        # Check if bbox is at least partially inside the crop
        if left >= x1 and right <= x2 and top >= y1 and bottom <= y2:
            # Adjust coordinates relative to crop
            new_cx = (bx - x1) / crop_w
            new_cy = (by - y1) / crop_h
            new_bw = bw / crop_w
            new_bh = bh / crop_h
            new_bboxes.append([new_cx, new_cy, new_bw, new_bh])
            new_labels.append(label)

    if not new_bboxes:
        print("No boxes survived the crop. You may want to retry or add logic to ensure at least one survives.")
        return None

    # Save image with color fix
    image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    
    if not os.path.exists(image_output.parent):
        os.makedirs(image_output.parent)
    cv2.imwrite(str(image_output), image_bgr)

    if not os.path.exists(anno_output.parent):
        os.makedirs(anno_output.parent)
    with open(anno_output, "w") as f:
        for label, bbox in zip(new_labels, new_bboxes):
            line = f"{label} " + " ".join("{:.7f}".format(x) for x in bbox)
            f.write(line + "\n")

    return {
        'image': cropped_image,
        'bboxes': new_bboxes,
        'labels': new_labels
    }

def reconstruct_labels(bboxes: list, labels: list):
    results = []

    for label in labels:
        bbox = bboxes[label]
        bbox.insert(label, 0)
        results.append(bbox)

    return results

def draw_yolo_bboxes(image_path, anno_path, class_names=None, color=(0, 255, 0), thickness=2):
    # Load image (in BGR for OpenCV)
    image = cv2.imread(str(image_path))
    H, W = image.shape[:2]

    # Read YOLO annotations
    with open(anno_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # Skip malformed lines

        class_id, x_center, y_center, width, height = map(float, parts)
        x_center *= W
        y_center *= H
        width *= W
        height *= H

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Optional: label with class name or ID
        label = class_names[int(class_id)] if class_names else str(int(class_id))

    # Display image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.title("YOLO Bounding Boxes")
    plt.show()

def main(base_path, n: int):
    cropped_count = 0
    all_anno_paths = []

    # Collect all annotation paths across all subdirectories
    for video in base_path.iterdir():
        if video.is_dir():
            annotations = video / "annotations"
            anno_list = list(annotations.glob("*.txt"))
            all_anno_paths.extend(anno_list[::5])  # Sample every 5th to reduce volume

    # Shuffle the list randomly
    random.shuffle(all_anno_paths)

    # Loop through shuffled list and crop until n images are done
    for anno_path in all_anno_paths:
        if cropped_count >= n:
            return  # Stop once n images are cropped

        video = anno_path.parents[1]
        images = video / "images"
        image_path = images / (anno_path.stem + ".jpg")

        if not image_path.exists():
            continue

        areas = get_area(anno_path)
        if len(areas) != 1:
            continue

        image_output = Path("cropped_for_cv_game/images") / (anno_path.stem + ".jpg")
        anno_output = Path("cropped_for_cv_game/annotations") / (anno_path.stem + ".txt")

        result = crop_image_random_with_bbox(
            image_path,
            bbox_path=anno_path,
            image_output=image_output,
            anno_output=anno_output
        )

        if result is not None:
            cropped_count += 1

if __name__ == "__main__":
    path2020 = Path("sharkeye2/all_data/yolo_image_format/2020")
    main(base_path=path2020, n = 30)
    


    # Testing 

    # image_path = Path("./2020/DJI_0091_Trim_1/images/frame100.jpg")
    # anno_path = Path("./2020/DJI_0091_Trim_1/annotations/frame100.txt")
    # cropped_img = Path("./Cropped/2020/DJI_0091_Trim_1/images/frame100.jpg")
    # cropped_anno = Path("./Cropped/2020/DJI_0091_Trim_1/annotations/frame100.txt")
    # crop_image_random_with_bbox(image_path, bbox_path=anno_path, image_output=cropped_img, anno_output=cropped_anno)
    # draw_yolo_bboxes(image_path=("./Cropped/2020/DJI_0091_Trim_1/images/frame100.jpg"), anno_path=("./Cropped/2020/DJI_0091_Trim_1/annotations/frame100.txt"), class_names=['0'])
    # main(base_path)
    
