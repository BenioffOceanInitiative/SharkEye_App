import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import time
import math
import threading
from segment_anything import sam_model_registry, SamPredictor
from utility import resource_path, select_torch_device
from log_config import get_logger

logger = get_logger("sharkeye.segment")
from pathlib import Path

try:
    from segmentation.segmentation_utility import show_mask, show_box
except Exception as e:
    from segmentation_utility import show_mask, show_box

from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull
from skimage.measure import label

ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 2688, 1512
ASPECT_RATIO = ORIGINAL_WIDTH / ORIGINAL_HEIGHT
DRONE_ALTITUDE_M = 40
FOV_RADIANS = 1.274090354  # From estimate of 73 degrees

DEFAULT_SAM_CHECKPOINT = Path("model_weights/sam_vit_b_01ec64.pth")

_MODEL_TYPES = {
    "sam_vit_h_4b8939.pth": "vit_h",
    "sam_vit_l_0b3195.pth": "vit_l",
    "sam_vit_b_01ec64.pth": "vit_b",
}


def _get_device() -> torch.device:
    return select_torch_device()



def _empty_torch_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _load_sam_predictor(checkpoint_path: Path) -> SamPredictor:
    """Load SAM weights from disk and return a ready-to-use predictor."""
    sam_checkpoint = resource_path(checkpoint_path)
    model_type = _MODEL_TYPES[checkpoint_path.name]
    device = _get_device()

    sam = sam_model_registry[model_type](checkpoint=None)

    # PyTorch 2.6 compatibility: explicitly allow pickle ONLY for trusted weights.
    ckpt = torch.load(sam_checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    missing, unexpected = sam.load_state_dict(ckpt, strict=False)
    if missing:
        logger.info(f"[SAM] Missing keys: {len(missing)} (often OK for buffers)")
    if unexpected:
        logger.info(f"[SAM] Unexpected keys: {len(unexpected)} (ignored)")

    sam.to(device=device)
    return SamPredictor(sam)


class SamPredictorCache:
    """Reuse a single SamPredictor instance for repeated segmentation calls."""

    _predictor: SamPredictor | None = None
    _checkpoint_key: str | None = None
    # Guards the load path so a background warmup thread and the processing thread
    # can't double-load the ~375MB checkpoint if they call in concurrently.
    _lock = threading.Lock()

    @classmethod
    def get_predictor(cls, checkpoint_path: Path = DEFAULT_SAM_CHECKPOINT) -> SamPredictor:
        sam_checkpoint = resource_path(checkpoint_path)
        if cls._predictor is not None and cls._checkpoint_key == sam_checkpoint:
            return cls._predictor

        with cls._lock:
            # Re-check inside the lock: another thread may have loaded it while we waited.
            if cls._predictor is not None and cls._checkpoint_key == sam_checkpoint:
                return cls._predictor
            cls.release()
            logger.info(f"[SAM] Loading model from {checkpoint_path.name}")
            cls._predictor = _load_sam_predictor(checkpoint_path)
            cls._checkpoint_key = sam_checkpoint
            return cls._predictor

    @classmethod
    def release(cls) -> None:
        if cls._predictor is not None:
            del cls._predictor
            cls._predictor = None
            cls._checkpoint_key = None
            _empty_torch_cache()


def get_sam_predictor(checkpoint_path: Path = DEFAULT_SAM_CHECKPOINT) -> SamPredictor:
    """Return a cached SamPredictor, loading weights on first use."""
    return SamPredictorCache.get_predictor(checkpoint_path)


def release_sam_model() -> None:
    """Release the cached SAM model and clear accelerator memory."""
    SamPredictorCache.release()


def load_sam_model(checkpoint_path: Path = DEFAULT_SAM_CHECKPOINT) -> SamPredictor:
    """Backward-compatible alias for get_sam_predictor."""
    return get_sam_predictor(checkpoint_path)


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


def run_prediction(
    image,
    bbox,
    checkpoint_path: Path = DEFAULT_SAM_CHECKPOINT,
    draw_mask=False,
    cropped=False,
    show_time=False,
):
    start_time = time.time()

    predictor = get_sam_predictor(checkpoint_path)
    predictor.set_image(image)

    if cropped:
        h, w, _ = image.shape
        point_h = int(h / 2)
        point_w = int(w / 2)

        input_point = np.array([[point_h, point_w]])
        input_label = np.array([1])

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        input_box = None
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
        if input_box is not None:
            show_box(input_box, plt.gca())
        plt.axis('off')
        plt.show()

    end_time = time.time()
    if show_time:
        logger.debug("Time taken to compute prediction: %.3fs", end_time - start_time)
    return masks


def draw_mask(mask, image):
    mask = np.squeeze(mask).astype(np.uint8)

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
    mask = np.squeeze(mask)  # Adjust dimensions
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


def calculate_shark_length_from_pixel(
    shark_pixel_length,
    original_width=ORIGINAL_WIDTH,
    original_height=ORIGINAL_HEIGHT,
    drone_altitude=DRONE_ALTITUDE_M,
    fov_radians=FOV_RADIANS,
):
    """Calculate shark length in feet based on pixel_length"""
    aspect_ratio = original_width / original_height
    long_side = (2 * aspect_ratio * drone_altitude * math.tan(fov_radians / 2)) / np.sqrt(1 + aspect_ratio ** 2)
    pixel_size_m = long_side / original_width

    length_m = shark_pixel_length * pixel_size_m
    return length_m * 3.28084  # Convert meters to feet


if __name__ == "__main__":

    if True:
        # Select image
        image = load_image('./src/segmentation/data/DJI_0091_Trim_1_images_frame113.jpg')
        bboxes = convert_yolo('./src/segmentation/data/DJI_0091_Trim_1_annotations_frame113.txt')[0]

        logger.debug("Running prediction on cropped image")
        cropped = crop_image(image, bboxes)
        prediction = run_prediction(image, cropped, cropped=True)
        logger.debug(calculate_shark_length_from_pixel(find_pixel_length(prediction, draw_line=True)))

        logger.debug(" Running prediction on image with bounding boxes")
        prediction = run_prediction(image, bboxes)
        logger.debug(calculate_shark_length_from_pixel(find_pixel_length(prediction, draw_line=True)))

        np.save('./mask.npy', prediction)
        release_sam_model()

    else:
        image = load_image('./src/segmentation/data/DJI_0091_Trim_1_images_frame113.jpg')
        logger.debug(image)
        mask = np.load('./src/segmentation/mask.npy')

        draw_mask(mask, image)
        logger.debug(calculate_shark_length_from_pixel(find_pixel_length(mask, draw_line=True)))
