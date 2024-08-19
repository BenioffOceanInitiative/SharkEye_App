import sys
import cv2
import os
from PyQt6.QtCore import QUrl, Qt, QTimer, QRect
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog, QSlider, QLabel, QGraphicsView, QGraphicsScene, QSizePolicy
from PyQt6.QtMultimedia import QMediaPlayer, QVideoSink
from PyQt6.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QGuiApplication, QIcon

class CustomSlider(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indicators = []
        self.indicator_image = QPixmap("./assets/images/Shark Detected Icon.png")
        self.indicator_icon = QIcon(self.indicator_image)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        for indicator in self.indicators:
            x = round((indicator / self.maximum()) * self.width())
            self.indicator_icon.paint(painter, x - 16, 0, 33, 40)

    def set_indicators(self, indicators):
        self.indicators = indicators
        self.update()

    def mousePressEvent(self, event):
        for indicator in self.indicators:
            x = round((indicator / self.maximum()) * self.width())

            # Define the QRect for the icon
            icon_rect = QRect(x - 16, -3, 33, 40)

            # Check if the mouse click is within the icon rectangle
            if icon_rect.contains(event.pos()):
                self.parent().set_position(indicator)
        super().mousePressEvent(event)

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.disply_width = 1024
        self.display_height = 768
        
        self.initial_width = 1024
        self.initial_height = 768

        self.results = None

        self.setWindowTitle("Video Player")
        self.resize(self.initial_width, self.initial_height)

        # Video Player 
        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.video_widget.setFixedSize(853, 480)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)       

        self.play_button = QPushButton()
        self.play_button.clicked.connect(self.play_video)
        self.play_button.setMaximumSize(30, 16)
        play_button = QPixmap("./assets/images/play-button-svg.svg").scaled(10, 9, transformMode=Qt.TransformationMode.SmoothTransformation)
        self.play_button.setIcon(QIcon(play_button))

        self.seek_previous_button = QPushButton()
        self.seek_previous_button.clicked.connect(self.seek_previous)
        self.seek_previous_button.setMaximumSize(30, 16)
        previous_button = QPixmap("./assets/images/previous-button.png").scaled(10, 10, transformMode=Qt.TransformationMode.SmoothTransformation)
        self.seek_previous_button.setIcon(QIcon(previous_button))

        self.seek_next_button = QPushButton()
        self.seek_next_button.clicked.connect(self.seek_next)
        self.seek_next_button.setMaximumSize(30, 16)
        next_button = QPixmap("./assets/images/next-button.png").scaled(10, 10, transformMode=Qt.TransformationMode.SmoothTransformation)
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
                width: 10px; /* Change the width of the handle */
                height: 10px; /* Change the height of the handle */
                border-radius: 5px; /* Round the corners of the handle */
            }""")
        self.slider.sliderMoved.connect(self.set_position)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)

        self.time_label = QLabel("00:00 / 00:00")

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.seek_previous_button)
        slider_layout.addWidget(self.play_button)
        slider_layout.addWidget(self.seek_next_button)
        slider_layout.addWidget(self.slider, alignment=Qt.AlignmentFlag.AlignCenter)
        slider_layout.addWidget(self.time_label)
        slider_layout.setAlignment(Qt.AlignmentFlag.AlignCenter| Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout()
        layout.addWidget(self.video_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.frame_stitch, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(slider_layout)

        self.setLayout(layout)

        self.video_path = None

        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.positionChanged.connect(self.update_position)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_slider)
        self.timer.start()

        self.is_slider_pressed = False
        self.fps = 30
        self.total_frames = 0

    def play_video(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
            self.display_video_widget()

    def seek_previous(self):
        if not self.media_player.isPlaying():
            self.set_position(round((self.slider.value() / 1000) * 1000) - 100)

    def seek_next(self):
        if not self.media_player.isPlaying():
            self.set_position(round((self.slider.value() / 1000) * 1000) + 100)

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

    def set_position(self, position):
        self.media_player.setPosition(position)
        if position in self.indicators:
            self.stitch_bbox()
        else:
            self.display_video_widget()

    def update_slider(self):
        if self.media_player.isSeekable() and not self.is_slider_pressed:
            self.slider.setValue(self.media_player.position())

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
        self.slider.set_indicators(frames)
        self.indicators = frames
    
    def clear_indicators(self):
        self.slider.set_indicators([])
        self.slider.update()

    def set_results_dir(self, dir):
        self.results = dir
        self.update()

    def set_video_dir(self, dir):
        self.video_path = dir
        self.media_player.setSource(QUrl.fromLocalFile(dir))
        self.analyze_video(dir)
        self.update()

    def set_bbox_dir(self, dir):
        self.bbox_dir = dir
        self.update()

    def stitch_bbox(self):
        """
        Function gets called when on frame of interest AND not isPlaying
            hide video
            show frame stitch
        Bbox shouldn't be showing when playing
        
        """
        if not self.media_player.isPlaying():
            self.frame_stitch.resize(self.video_widget.size())
            frame_dir = os.path.join(self.indicators[int(self.slider.value())])
            frame = QPixmap(frame_dir).scaled(853, 480, Qt.AspectRatioMode.KeepAspectRatio)

            self.frame_stitch.setPixmap(frame)
            self.video_widget.hide()
            self.frame_stitch.show()
    
    def display_video_widget(self):
        self.video_widget.resize(self.frame_stitch.size())
        self.frame_stitch.hide()
        self.video_widget.show()

    def remove_indicator(self, frame_number):
        indicator = int(frame_number * 1000/30)
        del self.indicators[indicator]
        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())