import numpy as np

class SharkTracker:
    """
    A class to track individual sharks across video frames.
    
    This class maintains the state of a tracked shark, including its position,
    detection history, best detected frame, and maximum estimated length.
    """

    def __init__(self, track_id: int, initial_detection: np.ndarray, max_missed_detections: int = 5, min_detected_frames: int = 3, frame_number: int = 0):
        """
        Initialize a new SharkTracker instance.

        :param track_id: Unique identifier for this tracker
        :param initial_detection: Initial detection coordinates [x, y, w, h, confidence, length]
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
        self.max_length = initial_detection[5]
        self.last_estimated_length = initial_detection[5]
        self.frame_number = frame_number

    def update(self, detection: np.ndarray = None, frame: np.ndarray = None, frame_with_box: np.ndarray = None, confidence: float = 0.0, length: float = 0.0, frame_number: int = 0):
        if detection is not None:
            self.last_detection = detection
            self.missed_detections = 0
            self.detected_frames += 1
            self._update_best_frame(frame, frame_with_box, confidence, frame_number)
            self._update_max_length(length)
            self.last_estimated_length = length
        else:
            self._update_missed_detection()

        self._update_status()

    def _update_missed_detection(self):
        """
        Update the tracker when no detection is found in the current frame.
        """
        self.missed_detections += 1

    def _update_best_frame(self, frame: np.ndarray, frame_with_box: np.ndarray, confidence: float, frame_number: int):
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
                self.frame_number = frame_number
                
    def _update_max_length(self, length: float):
        """
        Update the maximum length if the current detection is longer.

        :param length: Estimated shark length
        """
        if length > self.max_length:
            self.max_length = length

    def _update_status(self):
        """
        Update the active and valid status of the tracker.
        """
        self.is_active = self.missed_detections <= self.max_missed_detections
        self.is_valid = self.detected_frames >= self.min_detected_frames

    def predict_next_position(self) -> np.ndarray:
        """
        Predict the next position of the shark. Currently returns the last known position.

        :return: Predicted next position [x, y, w, h, confidence, length]
        """
        return self.last_detection

    @property
    def current_length(self):
        """
        Returns the current best estimate of the shark's length (max length).

        :return: Maximum detected length of the shark
        """
        return self.max_length

    @property
    def last_length(self):
        """
        Returns the last estimated length of the shark.

        :return: Last estimated length of the shark
        """
        return self.last_estimated_length