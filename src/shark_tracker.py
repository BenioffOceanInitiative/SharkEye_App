import numpy as np

class SharkTracker:
    """
    A class to track individual sharks across video frames.
    
    This class maintains the state of a tracked shark, including its position,
    detection history, and best detected frame.
    """

    def __init__(self, track_id: int, initial_detection: np.ndarray, max_missed_detections: int = 5, min_detected_frames: int = 3):
        """
        Initialize a new SharkTracker instance.

        :param track_id: Unique identifier for this tracker
        :param initial_detection: Initial detection coordinates [x, y, w, h, confidence]
        :param max_missed_detections: Maximum number of consecutive frames a shark can be missing before the track is considered inactive
        :param min_detected_frames: Minimum number of frames a shark must be detected to be considered a valid track
        """
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
        
        :param detection: New detection coordinates [x, y, w, h, confidence]
        :param frame: Current video frame without bounding boxes
        :param frame_with_box: Current video frame with bounding boxes
        :param confidence: Detection confidence
        """
        if detection is not None:
            self._update_with_detection(detection, frame, frame_with_box, confidence)
        else:
            self._update_missed_detection()

        self._update_status()

    def _update_with_detection(self, detection: np.ndarray, frame: np.ndarray, frame_with_box: np.ndarray, confidence: float):
        """
        Update the tracker with a new detection.

        :param detection: New detection coordinates [x, y, w, h, confidence]
        :param frame: Current video frame without bounding boxes
        :param frame_with_box: Current video frame with bounding boxes
        :param confidence: Detection confidence
        """
        self.last_detection = detection
        self.missed_detections = 0
        self.detected_frames += 1
        self._update_best_frame(frame, frame_with_box, confidence)

    def _update_missed_detection(self):
        """
        Update the tracker when no detection is found in the current frame.
        """
        self.missed_detections += 1

    def _update_status(self):
        """
        Update the active and valid status of the tracker.
        """
        self.is_active = self.missed_detections <= self.max_missed_detections
        self.is_valid = self.detected_frames >= self.min_detected_frames

    def _update_best_frame(self, frame: np.ndarray, frame_with_box: np.ndarray, confidence: float):
        """
        Update the best frame if the current detection has higher confidence.

        :param frame: Current video frame without bounding boxes
        :param frame_with_box: Current video frame with bounding boxes
        :param confidence: Detection confidence
        """
        if frame is not None and frame_with_box is not None:
            if self.best_frame_with_box is None or confidence > self.best_confidence:
                self.best_confidence = confidence
                self.best_frame = frame.copy()
                self.best_frame_with_box = frame_with_box.copy()

    def predict_next_position(self) -> np.ndarray:
        """
        Predict the next position of the shark. Currently returns the last known position.

        :return: Predicted next position [x, y, w, h, confidence]
        """
        return self.last_detection