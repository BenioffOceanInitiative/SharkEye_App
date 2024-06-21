from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication,QTextEdit, QSizePolicy, QLabel, QVBoxLayout,QHBoxLayout, QPushButton, QFileDialog, QListWidget
from PyQt5.QtGui import QPixmap, QTextCursor, QDragEnterEvent, QDropEvent, QDragMoveEvent
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
from tracker_logic import SharkTracker
from output import output
from video_processing import ar_resize

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray) # Signal to emit frame to GUI
    append_message_signal = pyqtSignal(str)  # Custom signal for appending messages to the GUI
    finished_signal = pyqtSignal()  # Signal to indicate the thread has finished

    def __init__(self, video_dict, device):
        super().__init__()
        self.file_paths = list(video_dict.values())
        self._run_flag = True
        self.device = device
        
    def seconds_to_minutes_and_seconds(self, seconds):
        minutes, seconds = divmod(seconds, 60)
        return str(minutes) + ':' + str(round(seconds)) 

    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def run(self, imgsz=736, altitude=40):
        gpu = self.device != 'cpu'
        #load the model
        model_weights_path = self.resource_path('../model_weights/best.pt')
        model = YOLO(model_weights_path)
        #select optimal frame rate for device
        if gpu:
            desired_frame_rate = 8
        elif not gpu:
            desired_frame_rate = 4
        
        final_shark_list = []
        final_low_conf_tracks_list = []

        shark_count = 0

        for video in self.file_paths:
            cap = cv2.VideoCapture(video)
            original_frame_width = cap.get(3)

            # get h, w for model.track to resize image with
            # TODO might need to resize and reduce frame rate before inference to conserve memory
            # new_imgsz = ar_resize(cap.get(3), cap.get(4), imgsz)
            # print(new_imgsz)
            
            # find the rate to sample the video to ge tthe desired frame rate
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_sample_rate = round(fps/desired_frame_rate)

            #initiate tracker
            st = SharkTracker(altitude, desired_frame_rate)

            frame_no = 0
            while cap.isOpened():
                success = cap.grab()

                #reducing video frame rate here
                if success and frame_no % frame_sample_rate == 0:
                    _, frame = cap.retrieve()
                    if gpu:
                        #TODO test if yolo resizing works and is efficient
                        results = model.track(frame, conf=.45, device=self.device, imgsz=1280, iou=0.90, show=False, verbose=False, persist=True)
                    elif not gpu:
                        results = model.track(frame, conf=.45, imgsz=1280, iou=0.90, show=False, verbose=False, persist=True)
                    # Get the boxes ,classes and track IDs
                    annotated_img = results[0].plot()
                    self.change_pixmap_signal.emit(annotated_img)
                    boxes = results[0].boxes.xywh.cpu().tolist()
                    confidence = results[0].boxes.conf.cpu().tolist() 
                    track_ids = results[0].boxes.id
                    if track_ids == None:
                        track_ids = []
                    else:
                        track_ids= track_ids.cpu().tolist()
                        
                    timestamp = self.seconds_to_minutes_and_seconds(cap.get(cv2.CAP_PROP_POS_MSEC)/fps)

                    detections_list = zip(track_ids, boxes, confidence)
                    all_tracks = st.update_tracker(detections_list, frame, original_frame_width, timestamp)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                elif not success:
                    break

                frame_no += 1
            
            #TODO are they sorted by object id 
            for trk in all_tracks:
                if trk.confirmed:
                    final_shark_list.append(trk)
                    shark_count += 1
                else:
                    final_low_conf_tracks_list.append(trk)

        # save ann info
        # output(final_shark_list, final_low_conf_tracks_list)
        self.append_message_signal.emit(f"Total Sharks Detected: {shark_count}")
        for shark in final_shark_list:
            self.append_message_signal.emit(f"\tShark #{int(shark.id)} - {round(shark.size, 2)}ft")

    def process_video(self, file_name, model):
        cap = cv2.VideoCapture(file_name)
        try:
            while self._run_flag:
                ret, cv_img = cap.read()
                if not ret:
                    break 
                results = model.track(cv_img, conf=0.7, verbose=False, show=False, device=self.device)
                speed = results[0].speed
                annotated_img = results[0].plot()
                self.change_pixmap_signal.emit(annotated_img)
                self.append_message_signal.emit(f"Speed: {speed} in {os.path.basename(file_name)}")
        except Exception as e:
                    self.append_message_signal.emit(f"Error processing video {file_name}: {e}")
        finally:
            cap.release()
                    
    def stop(self):
        """Sets run flag to False and waits for thread to finish, then closes capture system"""
        self._run_flag = False
        self.wait()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.video_dict = {}  # Dictionary to store filename to path mapping
        self.initUI()
        
    class DraggableListWidget(QListWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setAcceptDrops(True)
            self.setDragDropMode(QListWidget.InternalMove)
            self.previous_drop_indicator = None

        def dragEnterEvent(self, event: QDragEnterEvent):
            if event.mimeData().hasUrls():
                event.accept()
            else:
                super().dragEnterEvent(event)

        def dragMoveEvent(self, event: QDragMoveEvent):
            super().dragMoveEvent(event)
            # Get the index of the item under the cursor
            target_index = self.indexAt(event.pos()).row()
            # Clear previous indicator
            self.clearDropIndicator()
            # Set new indicator if target index is valid
            if target_index != -1:
                self.previous_drop_indicator = target_index
                self.item(target_index).setBackground(Qt.yellow)
            else:
                self.previous_drop_indicator = None

        def dropEvent(self, event: QDropEvent):
            self.clearDropIndicator()
            if event.mimeData().hasUrls():
                for url in event.mimeData().urls():
                    self.addItem(url.toLocalFile())
                event.acceptProposedAction()
            else:
                super().dropEvent(event)

        def clearDropIndicator(self):
            if self.previous_drop_indicator is not None:
                item = self.item(self.previous_drop_indicator)
                if item:  # Check if the item is not None
                    item.setBackground(Qt.white)
                self.previous_drop_indicator = None
        
    def initUI(self):
        self.setWindowTitle("Sharkeye Tracking")
        self.disply_width = 1024
        self.display_height = 768
        
        self.initial_width = 1024
        self.initial_height = 768
        
        self.resize(self.initial_width, self.initial_height)
        
        self.device = 'cpu'
        self.devices = ['cpu']

        # check for cuda/mps
        if torch.cuda.is_available():
            self.devices.append('cuda')
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            print('mps')
            self.devices.append('mps')
            self.device = 'mps'
            
        self.setStyleSheet("background-color: #1d2633;")

        self.video_file_list = self.DraggableListWidget(self)
        self.video_file_list.setStyleSheet("background-color: white; color: black")
        self.video_file_list.setMaximumHeight(100)
        
        self.add_videos_button = QPushButton('Add More Videos')
        self.add_videos_button.setStyleSheet("background-color: #082f54; color: white; border-radius: 4px; width: 100px;height: 30px;")
        self.add_videos_button.clicked.connect(self.add_videos)
    
        self.remove_video_button = QPushButton('Remove Selected Video')
        self.remove_video_button.setStyleSheet("background-color: #f22613; color: white; border-radius: 4px; width: 100px;height: 30px;")
        self.remove_video_button.clicked.connect(self.remove_selected_video)

        video_file_button_layout = QHBoxLayout()
        video_file_button_layout.addWidget(self.add_videos_button)
        video_file_button_layout.addWidget(self.remove_video_button)
        
        self.start_tracker_button = QPushButton('Start Tracker')
        self.start_tracker_button.setStyleSheet("background-color: #0aa319; color: white; border-radius: 4px;width: 100px;height: 30px;")
        self.stop_tracker_button = QPushButton('Stop Tracker')
        self.stop_tracker_button.setStyleSheet("background-color: #f54242; color: white; border-radius: 4px;height: 30px;width: 100px;")

        video_button_layout = QHBoxLayout()
        video_button_layout.addWidget(self.start_tracker_button)
        video_button_layout.addWidget(self.stop_tracker_button)
        
        self.video_player = QLabel(self)
        self.video_player.resize(self.disply_width, self.display_height)
        video_player_layout = QHBoxLayout()
        video_player_layout.addStretch(1)
        video_player_layout.addWidget(self.video_player)
        video_player_layout.addStretch(1)
        
        self.log_console = QTextEdit(self)
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(400)
        self.log_console.resize(self.disply_width, int(self.display_height/8))
        self.log_console.setStyleSheet("background-color: #0d0c0c; color: white; border-radius: 4px;")

        vertical_layout = QVBoxLayout()
        vertical_layout.addLayout(video_file_button_layout)
        vertical_layout.addWidget(self.video_file_list)
        vertical_layout.addLayout(video_button_layout)
        vertical_layout.addLayout(video_player_layout)
        vertical_layout.addStretch()
        vertical_layout.addWidget(self.log_console)

        self.setLayout(vertical_layout)
        self.start_tracker_button.clicked.connect(self.start_tracker)
        self.stop_tracker_button.clicked.connect(self.stop_track)
        self.no_video_file_state()
        
    def on_thread_finished(self):
        """Slot to handle thread completion."""
        self.stop_tracker_button.hide()
        self.start_tracker_button.show()

    def no_video_file_state(self):
        self.start_tracker_button.hide()
        self.stop_tracker_button.hide()
        self.remove_video_button.hide()
        self.video_player.hide()

    def add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Videos", "", "Video Files (*.mp4 *.avi *.mov)")
        for file in files:
            filename = os.path.basename(file)
            if file not in self.video_dict.values():
                if self.is_valid_video_dimensions(file):
                    filename = os.path.basename(file)
                    self.video_dict[filename] = file
                    self.video_file_list.addItem(filename)
                else:
                    self.append_message(f'{filename} does not have valid dimensions')
            else:
                self.append_message(f'{filename} is already in this list')
        if len(self.video_dict) > 0:
            self.add_videos_button.show()
            self.remove_video_button.show()
            self.video_file_list.show()
            self.start_tracker_button.show()
    
    def is_valid_video_dimensions(self, video_path):
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        return width >= 1280 or height >= 736
    
    def remove_selected_video(self):
        listItems = self.video_file_list.selectedItems()
        if not listItems:
            QMessageBox.information(self, "No Selection", "Please select a video to remove.")
            return
        for item in listItems:
            filename = item.text()
            if filename in self.video_dict:
                del self.video_dict[filename]
            self.video_file_list.takeItem(self.video_file_list.row(item))
        if len(self.video_dict) == 0:
            self.no_video_file_state()

    def append_message(self, message):
        """use this to log messages to the GUI, not print()"""
        self.log_console.moveCursor(QTextCursor.End)
        self.log_console.insertPlainText(f'{message}\n')
        self.log_console.moveCursor(QTextCursor.End)
        
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_player.setPixmap(qt_img)

    def start_tracker(self):
        if not self.video_dict:
            self.append_message("Please upload at least one video file")
            return
        self.thread = VideoThread(self.video_dict, self.device)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.append_message_signal.connect(self.append_message)
        self.thread.finished_signal.connect(self.on_thread_finished)
        self.thread.start()
        self.video_player.show()
        self.log_console.show()
        self.start_tracker_button.hide()
        self.stop_tracker_button.show()
        
    def stop_track(self):
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.finished_signal.emit()  # Emit the signal here
        self.stop_tracker_button.hide()
        self.start_tracker_button.show()

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause