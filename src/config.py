import os
import sys

CONFIDENCE_THRESHOLD = 0.3
MAX_MISSED_DETECTIONS = 5 # frames
MIN_DETECTED_FRAMES = 2
VIDEO_STRIDE = 30
project_root = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(project_root, 'model_weights/train8-weights-best.pt')