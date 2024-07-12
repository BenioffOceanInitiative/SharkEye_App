import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import datetime
import time
from typing import List, Tuple
import logging
import traceback
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
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

    def process_videos(self, video_paths: List[str], output_dir: str):
        logging.info(f"Processing {len(video_paths)} videos")
        total_unique_detections = 0
        total_processing_time = 0

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        run_dir = os.path.join(output_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)

        for i, video_file in enumerate(video_paths):
            if self.is_cancelled:
                break

            logging.info(f"\nProcessing video {i+1}/{len(video_paths)}: {video_file}")
            try:
                self._load_model()
                self.current_video_changed.emit(video_file)
                self.shark_trackers.clear()
                unique_detections, processing_time = self.process_single_video(video_file, run_dir)
                total_unique_detections += unique_detections
                total_processing_time += processing_time
                logging.info(f"Video processed: {unique_detections} detections in {processing_time:.2f} seconds")
                self.processing_finished.emit(unique_detections, processing_time)  # Emit for each video
            except Exception as e:
                logging.error(f"Error processing video {video_file}: {str(e)}")
                traceback.print_exc()
                self.error_occurred.emit(f"Error processing video {video_file}: {str(e)}")
                continue

            self._clear_gpu_memory()
            self.update_progress.emit(int((i + 1) / len(video_paths) * 100))

        self.all_videos_processed.emit(total_unique_detections, total_processing_time)
        
    def process_single_video(self, video_path: str, run_dir: str) -> Tuple[int, float]:
        """Process a single video for shark detection."""
        try:
            logging.info(f"Starting to process video: {video_path}")
            self.unique_track_ids = set()

            frames_dir = os.path.join(run_dir, "frames")
            bounding_boxes_dir = os.path.join(run_dir, "bounding_boxes")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(bounding_boxes_dir, exist_ok=True)

            start_time = time.time()
            
            total_frames = self._get_total_frames(video_path)
            total_processed_frames = total_frames // VIDEO_STRIDE

            results = self.model.track(source=video_path, imgsz=1280, conf=CONFIDENCE_THRESHOLD, 
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
                    
                    self._update_tracks(detections, original_frame, frame_with_boxes)
                    self._update_ui(frame_with_boxes)
                    
                    self.unique_track_ids.update(track.id for track in self.shark_trackers if track.is_valid)
                
                except Exception as e:
                    logging.error(f"Error processing video {video_path} on frame {i}: {str(e)}")
                    continue
                
                self.update_progress.emit(int((i + 1) / total_processed_frames * 100))
            
            self._save_best_frames(run_dir, os.path.basename(video_path))
            
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
                    detections.append(np.array([x, y, w, h, conf]))
        return detections

    def _update_tracks(self, detections: List[np.ndarray], frame: np.ndarray, frame_with_boxes: np.ndarray):
        """Update existing tracks with new detections and create new tracks for unmatched detections."""
        predicted_locations = [t.predict_next_position() for t in self.shark_trackers if t.is_active]
        matched_indices = self._match_detections_to_tracks(detections, predicted_locations)
        
        for i, detection in enumerate(detections):
            if i in matched_indices:
                track_idx = matched_indices[i]
                self.shark_trackers[track_idx].update(detection, frame, frame_with_boxes, detection[4])
            else:
                new_track = SharkTracker(len(self.shark_trackers), detection, self.max_missed_detections, self.min_detected_frames)
                new_track.update(detection, frame, frame_with_boxes, detection[4])
                self.shark_trackers.append(new_track)

        # Update all tracks, incrementing missed_detections for unmatched tracks
        for track in self.shark_trackers:
            if track.id not in [self.shark_trackers[idx].id for idx in matched_indices.values()]:
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
            x, y, w, h, confidence = detection
            
            x, y, w, h = map(int, (x - w/2, y - h/2, w, h))
            x, y = max(0, min(x, frame_width - 1)), max(0, min(y, frame_height - 1))
            w, h = min(w, frame_width - x), min(h, frame_height - y)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            track = next((t for t in self.shark_trackers if np.allclose(t.last_detection[:4], detection[:4])), None)
            
            if track:
                cv2.putText(frame, f"ID: {track.id}, Conf: {confidence:.2f}", (x, max(0, y-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
        return frame
    
    def _save_best_frames(self, result_dir: str, video_name: str):
        """Save the best frames for each valid shark track."""
        saved_count = 0
        for track in self.shark_trackers:
            if track.is_valid and track.best_frame_with_box is not None:
                base_filename = f"{os.path.splitext(video_name)[0]}_shark_{track.id}_conf_{track.best_confidence:.2f}.jpg"
                
                frame_path = os.path.join(result_dir, "frames", base_filename)
                cv2.imwrite(frame_path, track.best_frame)
                
                bbox_path = os.path.join(result_dir, "bounding_boxes", base_filename)
                cv2.imwrite(bbox_path, track.best_frame_with_box)

                saved_count += 1
        
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