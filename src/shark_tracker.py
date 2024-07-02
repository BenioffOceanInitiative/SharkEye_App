import numpy as np

class SharkTracker:
    def __init__(self, track_id: int, initial_detection: np.ndarray, max_missed_detections: int = 5, min_detected_frames: int = 3):
        self.id = track_id
        self.last_detection = initial_detection
        self.missed_detections = 0
        self.detected_frames = 1
        self.max_missed_detections = max_missed_detections
        self.min_detected_frames = min_detected_frames
        self.is_active = True
        self.is_valid = False
        self.best_frame_with_box = None
        self.best_frame = None
        self.best_confidence = 0.0

    def update(self, detection: np.ndarray = None, frame: np.ndarray = None, frame_with_box: np.ndarray = None, confidence: float = 0.0):
        """
        Update the tracker with a new detection or mark it as missed.
        
        :param detection: New detection coordinates [x, y, w, h]
        :param frame: Current video frame without bounding boxes
        :param frame_with_boxes: Current video frame with bounding boxes
        :param confidence: Detection confidence
        """
        if detection is not None:
            self.last_detection = detection
            self.missed_detections = 0
            self.detected_frames += 1
            if frame is not None and frame_with_box is not None:
                if self.best_frame_with_box is None or confidence > self.best_confidence:
                    print(f"Updating best frame for track {self.id}. Best confidence: {self.best_confidence}, Detected confidence: {confidence}")
                    self.best_confidence = confidence
                    self.best_frame = frame.copy()
                    self.best_frame_with_box = frame_with_box.copy()
        else:
            self.missed_detections += 1

        self.is_active = self.missed_detections <= self.max_missed_detections
        self.is_valid = self.detected_frames >= self.min_detected_frames
        print(f"Track {self.id} update - Conf: {self.best_confidence}, Is active: {self.is_active}, Is valid: {self.is_valid}, Detected frames: {self.detected_frames}, Missed detections: {self.missed_detections}")

    def predict_next_position(self) -> np.ndarray:
        """
        Predict the next position of the shark. Currently returns the last known position.
        """
        return self.last_detection