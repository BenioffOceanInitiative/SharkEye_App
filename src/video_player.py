import sys
import cv2
import os
from PyQt6.QtCore import QUrl, Qt, QTimer, QUrl, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QSlider, QLabel, QSizePolicy
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QPainter, QPixmap, QIcon
from utility import resource_path, get_video_path

class VideoPlayer(QWidget):
    request_next_video = pyqtSignal()
    request_previous_video = pyqtSignal()
    indicator_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.disply_width = 1024
        self.display_height = 768
        
        self.initial_width = 1024
        self.initial_height = 768
        
        self.current_video_index = 1
        self.total_videos = 1
        
        self.indicators = []
        
        self.fps = None
        self.total_frames = 0
        self.current_video_index = 1
        self.total_videos = 1
        
        self.frame_to_ms_dict = {}
        self.ms_to_frame_dict = {}

        self.setWindowTitle("Video Player")
        self.resize(self.initial_width, self.initial_height)

        # Video navigation layout
        nav_layout = QHBoxLayout()
        self.prev_video_button = QPushButton("<")
        self.prev_video_button.setStyleSheet("""
            QPushButton:disabled {
                background-color: gray;
                color: white;
            }
        """)
        self.prev_video_button.clicked.connect(self.request_previous_video.emit)
        
        self.video_index_label = QLabel("Video 1 of 1")
        self.video_index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_index_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.next_video_button = QPushButton(">")
        self.next_video_button.setStyleSheet("""
            QPushButton:disabled {
                background-color: gray;
                color: white;
            }
        """)
        self.next_video_button.clicked.connect(self.request_next_video.emit)
        
        nav_layout.addWidget(self.prev_video_button)
        nav_layout.addWidget(self.video_index_label)
        nav_layout.addWidget(self.next_video_button)

        # Video file name label
        self.video_name_label = QLabel()
        self.video_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        # Video Player 
        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.video_widget.setFixedSize(853, 480)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)       

        self.play_button = QPushButton()
        self.play_button.clicked.connect(self.play_video)
        self.play_button.setMaximumSize(30, 16)

        self.play_button_icon = QPixmap(resource_path("assets/images/play-button-svg.svg")).scaled(10, 9, transformMode=Qt.TransformationMode.SmoothTransformation)
        self.pause_button_icon = QPixmap(resource_path("assets/images/pause-button-svg.svg")).scaled(10, 9, transformMode=Qt.TransformationMode.SmoothTransformation)
        self.play_button.setIcon(QIcon(self.play_button_icon))

        self.seek_previous_button = QPushButton()
        self.seek_previous_button.clicked.connect(self.seek_previous)
        self.seek_previous_button.setMaximumSize(30, 16)
        previous_button = QPixmap(resource_path("assets/images/previous-button.png")).scaled(10, 10, transformMode=Qt.TransformationMode.SmoothTransformation)
        self.seek_previous_button.setIcon(QIcon(previous_button))

        self.seek_next_button = QPushButton()
        self.seek_next_button.clicked.connect(self.seek_next)
        self.seek_next_button.setMaximumSize(30, 16)
        next_button = QPixmap(resource_path("assets/images/next-button.png")).scaled(10, 10, transformMode=Qt.TransformationMode.SmoothTransformation)
        self.seek_next_button.setIcon(QIcon(next_button))
        
        # Stitched frames
        self.frame_stitch = QLabel()
        self.frame_stitch.setFixedSize(853, 480)
        self.frame_stitch.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.frame_stitch.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_stitch.hide()
        
        self.slider = CustomSlider(Qt.Orientation.Horizontal)
        self.slider.setFixedHeight(80)
        self.slider.setMinimumWidth(self.video_widget.width() - 195)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #bcbcbc;
                height: 10px;
            }
            QSlider::handle:horizontal {
                border: 1px solid #5c5c5c;
                width: 10px;
                height: 10px;
                border-radius: 5px;
            }""")
        self.slider.sliderMoved.connect(self.set_position)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.indicator_clicked.connect(self.indicator_clicked)

        self.time_label = QLabel("00:00 / 00:00")

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.seek_previous_button)
        slider_layout.addWidget(self.play_button)
        slider_layout.addWidget(self.seek_next_button)
        slider_layout.addWidget(self.slider, alignment=Qt.AlignmentFlag.AlignCenter)
        slider_layout.addWidget(self.time_label)
        slider_layout.setAlignment(Qt.AlignmentFlag.AlignCenter| Qt.AlignmentFlag.AlignTop)

        # Shark information labels
        self.shark_info_label = QLabel()
        self.shark_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.shark_info_label.setStyleSheet("font-size: 12px;")

        layout = QVBoxLayout()
        layout.addLayout(nav_layout)
        layout.addWidget(self.video_name_label)
        layout.addWidget(self.video_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.frame_stitch, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(slider_layout)
        layout.addWidget(self.shark_info_label)

        self.setLayout(layout)

        self.video_path = None

        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.mediaStatusChanged.connect(self.handle_media_status_changed)

        self.initial_position = 0
        self.last_played_position = 0
        
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_slider)
        self.timer.start()

        self.is_slider_pressed = False
        self.fps = 30
        self.media_player.positionChanged.connect(self.check_indicator_change)

    def indicator_clicked(self, position):
        self.media_player.pause() 
        self.play_button.setIcon(QIcon(self.play_button_icon))
        self.set_position(position)
        self.update_frame_display(position)

    def check_indicator_change(self, position):
        if self.indicators:
            closest_indicator = min(self.indicators, key=lambda x: abs(x - position))
            if abs(closest_indicator - position) < 100:  # Within 100ms
                print(f"Indicator changed at position: {position}")
                self.indicator_changed.emit(closest_indicator)

    def analyze_video(self, filename):
        cap = cv2.VideoCapture(filename)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = int((self.total_frames / self.fps) * 1000)  # Duration in milliseconds
        cap.release()

        self.update_duration(self.duration)
        self.clear_indicators()
            
    def set_video_index(self, index, total):
        self.current_video_index = index
        self.total_videos = total
        self.video_index_label.setText(f"Video {index} of {total}")
        self.update_navigation_buttons()
        
    def update_navigation_buttons(self):
        self.prev_video_button.setEnabled(self.current_video_index > 1)
        self.next_video_button.setEnabled(self.current_video_index < self.total_videos)
    
    def play_video(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_button.setIcon(QIcon(self.play_button_icon))
        else:
            self.play_button.setIcon(QIcon(self.pause_button_icon))
            self.media_player.play()
            self.display_video_widget()
    
    def reset_play_button(self):
        self.media_player.pause()
        self.play_button.setIcon(QIcon(self.play_button_icon))

    def seek_previous(self):
        self.media_player.pause()
        current_position = self.media_player.position()
        if self.indicators:
            previous_indicators = [i for i in self.indicators if i < current_position]
            if previous_indicators:
                new_position = max(previous_indicators)
            else:
                new_position = 0
        else:
            new_position = max(0, current_position - 5000)  # Go back 5 seconds if no indicators
        
        self.set_position(new_position)
        self.update_seek_buttons()

    def seek_next(self):
        self.media_player.pause()
        current_position = self.media_player.position()
        if self.indicators:
            next_indicators = [i for i in self.indicators if i > current_position]
            if next_indicators:
                new_position = min(next_indicators)
            else:
                new_position = self.media_player.duration()
        else:
            new_position = min(self.media_player.duration(), current_position + 5000)  # Go forward 5 seconds if no indicators
        
        self.set_position(new_position)
        self.update_seek_buttons()

    def update_seek_buttons(self):
        current_position = self.media_player.position()
        duration = self.media_player.duration()
        
        has_previous = current_position > 0 or (self.indicators and min(self.indicators) < current_position)
        has_next = current_position < duration - 100  # Allow a small margin for rounding errors
        
        self.seek_previous_button.setEnabled(bool(has_previous))
        self.seek_next_button.setEnabled(bool(has_next))

    def analyze_video(self, filename):
        cap = cv2.VideoCapture(filename)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int((self.total_frames / self.fps) * 1000)  # Duration in milliseconds
        cap.release()

        self.update_duration(duration)
        self.clear_indicators()

    def update_duration(self, duration):
        self.slider.setRange(0, duration)
        self.update_time_label()

    def update_position(self, position):
        if not self.is_slider_pressed:
            self.slider.setValue(position)
        self.update_time_label()
        self.last_played_position = position
    
    def set_position(self, position):
        if self.media_player.isSeekable():
            self.media_player.setPosition(position)
            self.slider.setValue(position)
            self.update_frame_display(position)
            self.update_time_label()
            self.update_seek_buttons()
            self.last_played_position = position
            # Ensure the slider is visually updated
            self.slider.update()
        else:
            print("Media is not seekable yet. Waiting for media to load.")
    
    def find_nearest_valid_position(self, target_position):
        step = 1000  # 1 second step
        max_attempts = 10
        for i in range(max_attempts):
            for direction in [1, -1]:
                test_position = target_position + (i * step * direction)
                if 0 <= test_position <= self.media_player.duration():
                    if self.media_player.setPosition(test_position):
                        return test_position
        return None

    def update_frame_display(self, position):
        print(f"Updating frame display for position: {position}")
        try:
            if self.indicators:
                closest_indicator = min(self.indicators, key=lambda x: abs(x - position))
                if abs(closest_indicator - position) < 100:  # Within 100ms
                    frame_number = self.ms_to_frame(closest_indicator)
                    if frame_number is not None:
                        print(f"Found frame number: {frame_number}")
                        self.stitch_bbox(frame_number)
                        self.indicator_changed.emit(closest_indicator)
                    else:
                        print(f"No frame number found for position: {closest_indicator}")
                        self.display_video_widget()
                else:
                    self.display_video_widget()
            else:
                self.display_video_widget()
        except Exception as e:
            print(f"Error updating frame display: {str(e)}")
            self.display_video_widget()

    def update_slider(self):
        if self.media_player.isSeekable() and not self.is_slider_pressed:
            current_position = self.media_player.position()
            self.slider.setValue(current_position)
            self.slider.update()  # Force a visual update

    def slider_pressed(self):
        self.is_slider_pressed = True

    def slider_released(self):
        self.is_slider_pressed = False
        self.set_position(self.slider.value())

    def update_time_label(self):
        position = self.media_player.position()
        duration = self.media_player.duration()
        position_str = self.format_time(position)
        duration_str = self.format_time(duration)
        self.time_label.setText(f"{position_str} / {duration_str}")

    def format_time(self, milliseconds):
        seconds = int(milliseconds / 1000)
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def set_interesting_points(self, frames):
        print(f"Setting interesting points: {frames}")
        self.frame_paths = frames  # This is a dict of {frame_number: full_frame_path}
        self.frame_to_ms_dict = {frame: self.frame_to_ms(frame) for frame in frames.keys()}
        self.ms_to_frame_dict = {ms: frame for frame, ms in self.frame_to_ms_dict.items()}
        self.indicators = sorted(self.frame_to_ms_dict.values())
        self.slider.set_indicators(self.indicators)
        self.slider.update()
        
        if self.indicators:
            self.initial_position = self.indicators[0]
        else:
            self.initial_position = 0
        
        print(f"Frame paths: {self.frame_paths}")
        print(f"Frame to ms dict: {self.frame_to_ms_dict}")
        print(f"MS to frame dict: {self.ms_to_frame_dict}")
        print(f"Indicators: {self.indicators}")
            
    def frame_to_ms(self, frame_number):
        return int((frame_number / self.fps) * 1000)
    
    def ms_to_frame(self, ms):
        if ms in self.ms_to_frame_dict:
            return self.ms_to_frame_dict[ms]
        else:
            # If exact match not found, find the closest frame
            closest_ms = min(self.ms_to_frame_dict.keys(), key=lambda x: abs(x - ms))
            print(f"Exact ms {ms} not found, using closest ms {closest_ms}")
            return self.ms_to_frame_dict[closest_ms]

    def handle_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self.update_slider()
            self.update_duration(self.media_player.duration())
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            print("Invalid media file")

    def clear_indicators(self):
        self.slider.set_indicators([])
        self.slider.update()

    def set_video_info(self, video_name, shark_id=None, shark_length=None, shark_confidence=None):
        self.video_name_label.setText(os.path.basename(video_name))
        
        shark_info = []
        if shark_id is not None:
            shark_info.append(f"Shark ID: {shark_id}")
        if shark_length is not None:
            shark_info.append(f"Length: {shark_length:.2f} ft")
        if shark_confidence is not None:
            shark_info.append(f"Confidence: {shark_confidence:.2f}")
            
        print(shark_info)
        
        self.shark_info_label.setText(" | ".join(shark_info))

    def set_results_dir(self, dir):
        self.results = dir
        self.update()

    def set_video_dir(self, video_name):
        video_path = get_video_path(video_name)
        if os.path.exists(video_path):
            self.video_path = video_path
            self.media_player.setSource(QUrl.fromLocalFile(video_path))
            self.analyze_video(video_path)
            self.reset_play_button()
            self.update()
            self.update_seek_buttons()
            
            # Wait for the media to be loaded before setting position
            self.media_player.mediaStatusChanged.connect(self.on_media_loaded)
        else:
            print(f"Video file not found: {video_path}")
    
    def on_media_loaded(self, status):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            # Disconnect to prevent multiple calls
            self.media_player.mediaStatusChanged.disconnect(self.on_media_loaded)
            
            # Now that media is loaded, we can set the position
            if self.indicators:
                self.set_position(self.indicators[0])
                self.indicator_changed.emit(self.indicators[0])  # Emit signal to update shark info
            else:
                self.set_position(0)
            
            # Ensure the slider is updated
            self.update_slider()
    
    def set_bbox_dir(self, dir):
        self.bbox_dir = dir
        self.update()

    def stitch_bbox(self, frame_number):
        if frame_number in self.frame_paths:
            frame_path = self.frame_paths[frame_number]
            if os.path.exists(frame_path):
                pixmap = QPixmap(frame_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(853, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.frame_stitch.setPixmap(scaled_pixmap)
                    self.video_widget.hide()
                    self.frame_stitch.show()
                else:
                    print(f"Failed to load pixmap from {frame_path}")
                    self.display_video_widget()
            else:
                print(f"Frame path not found: {frame_path}")
                self.display_video_widget()
        else:
            print(f"No frame path for frame number: {frame_number}")
            self.display_video_widget()
    
    def display_video_widget(self):
        self.video_widget.resize(self.frame_stitch.size())
        self.frame_stitch.hide()
        self.video_widget.show()

    def remove_indicator(self, frame_number):
        indicator = int(frame_number * 1000 / self.fps)
        if indicator in self.indicators:
            self.indicators.remove(indicator)
            self.slider.set_indicators(self.indicators)
            self.slider.update()
        if frame_number in self.frame_paths:
            del self.frame_paths[frame_number]

class CustomSlider(QSlider):
    indicator_clicked = pyqtSignal(int)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indicators = []
        self.frame_paths = {}  # This will store frame number to path mapping
        self.indicator_image = QPixmap(resource_path("assets/images/Shark Detected Icon.png"))
        self.indicator_icon = QIcon(self.indicator_image)
        self.indicator_width = 33
        self.indicator_height = 40

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        for indicator in self.indicators:
            x = self.position_to_x(indicator)
            self.indicator_icon.paint(painter, x - self.indicator_width // 2, -3, self.indicator_width, self.indicator_height)

    def set_indicators(self, indicators):
        self.indicators = indicators
        self.update()

    def mousePressEvent(self, event):
        clicked_position = self.x_to_position(event.position().x())
        
        # Check if the click is near an indicator
        for indicator in self.indicators:
            indicator_x = self.position_to_x(indicator)
            if abs(event.position().x() - indicator_x) <= self.indicator_width // 2:
                self.indicator_clicked.emit(indicator)
                return

        # If not near an indicator, use default behavior
        super().mousePressEvent(event)

    def position_to_x(self, position):
        return int((position - self.minimum()) / (self.maximum() - self.minimum()) * self.width())

    def x_to_position(self, x):
        return int(x / self.width() * (self.maximum() - self.minimum()) + self.minimum())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())