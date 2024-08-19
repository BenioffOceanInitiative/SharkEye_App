import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import datetime
import time
from typing import List, Tuple, Callable
import logging
import traceback
import cv2
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import torch
import pandas as pd
from ultralytics import YOLO

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from shark_tracker import SharkTracker
from config import CONFIDENCE_THRESHOLD, VIDEO_STRIDE, MAX_MISSED_DETECTIONS, MIN_DETECTED_FRAMES, MODEL_PATH

class SharkDetector(QObject):
    update_progress = pyqtSignal(int)
    update_frame = pyqtSignal(QPixmap)
    processing_finished = pyqtSignal(int, float)
    all_videos_processed = pyqtSignal(int, float)
    error_occurred = pyqtSignal(str)
    current_video_changed = pyqtSignal(str)
    

    def __init__(self):
        super().__init__()
        logging.info("Initializing SharkDetector Class")
        self.model = None
        self.device = self._get_device()
        self.is_cancelled = False
        self.shark_trackers: List[SharkTracker] = []
        self.unique_track_ids = set()
        self.max_distance = 0
        self.max_missed_detections = MAX_MISSED_DETECTIONS
        self.min_detected_frames = MIN_DETECTED_FRAMES

    def _get_device(self) -> str:
        """Determine the appropriate device (CUDA, MPS, or CPU) for model execution."""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def _load_model(self):
        """Load the YOLO model and configure it based on the available device."""
        try:
            logging.info("Model loading...")
            self.model = YOLO(MODEL_PATH)
            logging.info(f"Model is for device: {self.device}")
            if self.device == 'cpu':
                self.model.to(self.device).float()
                logging.info("Model loaded in full precision (float32) mode")
            else:
                self.model.to(self.device).half()
                logging.info("Model loaded in half precision (float16) mode")
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logging.error(error_msg)
            self.error_occurred.emit(error_msg)
            raise
        
    def _unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            self._clear_gpu_memory()
            logging.info("Model unloaded and GPU memory cleared")
            
    def process_videos(self, video_paths, progress_callback, frame_callback):
        logging.info(f"Processing {len(video_paths)} videos")
        total_unique_detections = 0
        total_processing_time = 0

        # Create a timestamped folder for this detection run
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", timestamp)
        os.makedirs(results_dir, exist_ok=True)
        
        self.frame_data = []
        for i, video_path in enumerate(video_paths):
            if self.is_cancelled:
                break

            current_video = i + 1
            total_videos = len(video_paths)

            logging.info(f"\nProcessing video {current_video}/{total_videos}: {video_path}")
            try:
                self._load_model()
                
                unique_detections, processing_time = self.process_single_video(
                    video_path, 
                    results_dir, 
                    lambda video_progress: progress_callback(current_video, total_videos, video_progress),
                    frame_callback
                )
                total_unique_detections += unique_detections
                total_processing_time += processing_time
                logging.info(f"Video processed: {unique_detections} detections in {processing_time:.2f} seconds")
                self.processing_finished.emit(unique_detections, processing_time)
            except Exception as e:
                logging.error(f"Error processing video {video_path}: {str(e)}")
                self.error_occurred.emit(f"Error processing video {video_path}: {str(e)}")
            finally:
                self._unload_model()
         
        self.frame_data = pd.DataFrame(self.frame_data)
        frame_data_path = os.path.join(results_dir, "results.csv")
        self.frame_data.to_csv(frame_data_path, mode='a')

        self.all_videos_processed.emit(total_unique_detections, total_processing_time)
        return total_unique_detections, total_processing_time, results_dir
    
    def process_single_video(self, video_path, results_dir, progress_callback, frame_callback):
        """Process a single video for shark detection."""
        try:
            logging.info(f"Starting to process video: {video_path}")
            self.unique_track_ids = set()

            frames_dir = os.path.join(results_dir, "frames")
            bounding_boxes_dir = os.path.join(results_dir, "bounding_boxes")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(bounding_boxes_dir, exist_ok=True)

            start_time = time.time()
            
            total_frames = self._get_total_frames(video_path)
            total_processed_frames = total_frames // VIDEO_STRIDE

            results = self.model.track(source=video_path, imgsz=[736,1280], conf=CONFIDENCE_THRESHOLD, 
                                    verbose=False, persist=True, vid_stride=VIDEO_STRIDE, 
                                    stream=True, show=False, classes=[0])
            
            for i, result in enumerate(results):
                if self.is_cancelled:
                    break

                try:                    
                    original_frame = result.orig_img
                    
                    frame_with_boxes = original_frame.copy()
                    
                    self.max_distance = max(original_frame.shape) * 0.25
                    detections = self._get_detections(result)
                    
                    frame_with_boxes = self._draw_tracks(frame_with_boxes, detections)
                    
                    self._update_tracks(detections, original_frame, frame_with_boxes, frame_number = (i + 1) * VIDEO_STRIDE)
                    
                    # Convert the frame to QPixmap and update the UI
                    height, width, channel = frame_with_boxes.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(frame_with_boxes.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(q_img)
                    frame_callback(pixmap)
                    frame_progress = int((i / total_frames) * 100)
                    progress_callback(frame_progress)
                    
                    self.unique_track_ids.update(track.id for track in self.shark_trackers if track.is_valid)
                
                except Exception as e:
                    logging.error(f"Error processing video {video_path} on frame {i}: {str(e)}")
                    continue
                
                progress = int((i + 1) / total_processed_frames * 100)
                progress_callback(progress)

            self._save_best_frames(results_dir, video_path)

            processing_time = time.time() - start_time
            return len(self.unique_track_ids), processing_time
        
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            logging.error(error_msg)
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
            return 0, 0.0

    def _get_total_frames(self, video_path: str) -> int:
        """Get the total number of frames in a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Error opening video file")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        logging.info(f"Video file opened. Total frames: {total_frames}")
        return total_frames

    def _get_detections(self, result):
        """Extract shark detections from the YOLO model result."""
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                if box.cls.cpu().numpy()[0] == 0:  # Assuming 0 is the class for sharks
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    conf = box.conf.item()
                    
                    # Estimate shark length (assuming altitude of 40m)
                    length = self.calculate_shark_length(w, h)
                    
                    detections.append(np.array([x, y, w, h, conf, length]))
        return detections
    
    def _update_tracks(self, detections: List[np.ndarray], frame: np.ndarray, frame_with_boxes: np.ndarray, frame_number: int):
        if not self.shark_trackers:
            self._create_new_tracks(detections)
            return

        matched_track_indices = set()
        for i, detection in enumerate(detections):
            closest_track = self._find_closest_track(detection, matched_track_indices)
            if closest_track:
                j, track = closest_track
                self._update_existing_track(track, detection, frame, frame_with_boxes, frame_number)
                matched_track_indices.add(j)
            else:
                self._create_new_track(detection)

        self._update_unmatched_tracks(matched_track_indices)

    def _create_new_tracks(self, detections):
        for detection in detections:
            new_track = SharkTracker(len(self.shark_trackers), detection, self.max_missed_detections, self.min_detected_frames)
            self.shark_trackers.append(new_track)
        print(f"Created {len(self.shark_trackers)} new tracks")

    def _find_closest_track(self, detection, matched_track_indices):
        closest_track = None
        min_distance = float('inf')
        for j, track in enumerate(self.shark_trackers):
            if j in matched_track_indices:
                continue
            distance = np.linalg.norm(detection[:2] - track.last_detection[:2])
            if distance < min_distance and distance < self.max_distance:
                min_distance = distance
                closest_track = (j, track)
        return closest_track

    def _update_existing_track(self, track, detection, frame, frame_with_boxes, frame_number):
        track.update(detection, frame, frame_with_boxes, detection[4], detection[5], frame_number)

    def _create_new_track(self, detection):
        new_track = SharkTracker(len(self.shark_trackers), detection, self.max_missed_detections, self.min_detected_frames)
        self.shark_trackers.append(new_track)

    def _update_unmatched_tracks(self, matched_track_indices):
        for j, track in enumerate(self.shark_trackers):
            if j not in matched_track_indices:
                track.update()
            
    def _match_detections_to_tracks(self, detections: List[np.ndarray], predicted_locations: List[np.ndarray]) -> dict:
        """Match new detections to existing tracks using the Hungarian algorithm."""
        if not detections or not predicted_locations:
            return {}

        cost_matrix = np.array([[np.linalg.norm(d[:2] - p[:2]) for p in predicted_locations] for d in detections])
        rows, cols = linear_sum_assignment(cost_matrix)
        
        return {row: col for row, col in zip(rows, cols) if cost_matrix[row, col] <= self.max_distance}
    
    def _draw_tracks(self, frame: np.ndarray, current_detections: List[np.ndarray]) -> np.ndarray:
        """Draw bounding boxes and track IDs on the frame for current detections."""
        frame_height, frame_width = frame.shape[:2]

        for detection in current_detections:
            x, y, w, h = detection[:4]
            confidence, length = detection[4], detection[5]
            
            x, y, w, h = map(int, (x - w/2, y - h/2, w, h))
            x, y = max(0, min(x, frame_width - 1)), max(0, min(y, frame_height - 1))
            w, h = min(w, frame_width - x), min(h, frame_height - y)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
            # Find matching track
            matching_track = min(self.shark_trackers, key=lambda t: np.linalg.norm(t.last_detection[:2] - detection[:2]), default=None)
            
            if matching_track and np.linalg.norm(matching_track.last_detection[:2] - detection[:2]) < self.max_distance:                
                # Add shark ID and confidence to the bounding box
                label = f"ID: {matching_track.id}, Conf: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                
                # Add length information below the bounding box
                length_label = f"Max Len: {matching_track.current_length:.2f}ft, Curr Len: {matching_track.last_length:.2f}ft"
                cv2.putText(frame, length_label, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            else:
                print("No matching track found")
                # Add confidence to the bounding box for new detections
                label = f"Conf: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                
                # Add length information below the bounding box
                length_label = f"Curr Len: {length:.2f} ft"
                cv2.putText(frame, length_label, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        return frame

    def _save_best_frames(self, result_dir: str, video_path: str):
        """Save the best frames for each valid shark track."""
        saved_count = 0
        video_name = os.path.basename(video_path)
        for track in self.shark_trackers:
            if track.is_valid and track.best_frame_with_box is not None:
                base_filename = f"{os.path.splitext(video_name)[0]}_shark_{track.id}_conf_{track.best_confidence:.2f}_len_{track.max_length:.2f}_frame{track.frame_number}.jpg"
                
                frame_path = os.path.join(result_dir, "frames", base_filename)
                cv2.imwrite(frame_path, track.best_frame)
                
                bbox_path = os.path.join(result_dir, "bounding_boxes", base_filename)
                cv2.imwrite(bbox_path, track.best_frame_with_box)

                self.frame_data.append([base_filename, bbox_path, video_path])

                saved_count += 1

    def calculate_shark_length(
        self,
        bounding_box_width_pixels,
        bounding_box_height_pixels,
        altitude_m = 40,
        sensor_width_mm=13.2,
        sensor_height_mm=8.8,
        focal_length_mm=28,
        image_width_pixels=1280,
        image_height_pixels=736
    ):
        """
        Calculate the length of a shark in feet based on its bounding box in a drone image.
        
        Parameters:
        bounding_box_width_pixels (int): Width of the shark's bounding box in pixels
        bounding_box_height_pixels (int): Height of the shark's bounding box in pixels
        altitude_m (float): Altitude of the drone in meters
        sensor_width_mm (float): Width of the camera sensor in millimeters (default: 13.2)
        sensor_height_mm (float): Height of the camera sensor in millimeters (default: 8.8)
        focal_length_mm (float): Focal length of the camera in millimeters (default: 28)
        image_width_pixels (int): Width of the image in pixels (default: 1280)
        image_height_pixels (int): Height of the image in pixels (default: 736)
        
        Returns:
        float: Estimated length of the shark in feet
        """
        # Calculate GSD (Ground Sampling Distance) in meters per pixel
        gsd_w = (sensor_width_mm * altitude_m) / (focal_length_mm * image_width_pixels)
        gsd_h = (sensor_height_mm * altitude_m) / (focal_length_mm * image_height_pixels)
        
        # Use average GSD for more accuracy
        gsd = (gsd_w + gsd_h) / 2
        
        # Calculate shark dimensions in meters
        shark_width_m = bounding_box_width_pixels * gsd
        shark_height_m = bounding_box_height_pixels * gsd
        
        # Calculate diagonal length of the bounding box (assuming this is the shark's length)
        shark_length_m = math.sqrt(shark_width_m**2 + shark_height_m**2)
        
        # Convert meters to feet (1 meter = 3.28084 feet)
        shark_length_ft = shark_length_m * 3.28084
        
        return shark_length_ft

    def _update_ui(self, frame):
        """Update the UI with the current frame and progress."""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        
        if not pixmap.isNull():
            self.update_frame.emit(pixmap)

    def _clear_gpu_memory(self):
        """Clear GPU memory cache if CUDA or MPS is being used."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def cancel(self):
        """Set the cancellation flag to stop the video processing."""
        self.is_cancelled = True