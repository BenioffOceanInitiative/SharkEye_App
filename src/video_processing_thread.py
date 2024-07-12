from PyQt6.QtCore import QThread, pyqtSignal
import logging

class VideoProcessingThread(QThread):
    error_occurred = pyqtSignal(str)

    def __init__(self, shark_detector, video_paths, output_dir):
        super().__init__()
        self.shark_detector = shark_detector
        self.video_paths = video_paths
        self.output_dir = output_dir
        logging.info(f"VideoProcessingThread initialized with {len(video_paths)} videos")

    def run(self):
        try:
            logging.info("Starting video processing")
            self.shark_detector.process_videos(self.video_paths, self.output_dir)
            logging.info("Video processing completed successfully")
        except Exception as e:
            logging.exception(f"Error in Video Processing Thread: {str(e)}")
            self.error_occurred.emit(str(e))