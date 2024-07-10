import os
import sys

CONFIDENCE_THRESHOLD = 0.3
MAX_MISSED_DETECTIONS = 5 # frames
MIN_DETECTED_FRAMES = 2 
VIDEO_STRIDE = 30

model_base_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(model_base_path, 'model_weights/train6-weights-best.pt')