import cv2
import time
import imutils
import numpy as np
import easyocr
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import sys
from PyQt5.QtGui import QImage, QPixmap

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detection_signal = pyqtSignal(tuple)

    def __init__(self, video_path):
        super().__init__()
        self.car_cascade = cv2.CascadeClassifier("cars.xml")
        self.video_path = video_path
        self.running = True
        self.detected_plates = {}
        self.car_detected = {}  # Store car detection status and start time

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: can't open video capture.")
            sys.exit()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            frame = self.detect_and_annotate(frame)
            self.change_pixmap_signal.emit(frame)
            time.sleep(0.03)

        cap.release()

    def detect_and_annotate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Vehicle Detected", (x + w, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            plate_text = self.detect_registration_plate(frame[y:y + h, x:x + w])
            if plate_text:
                self.handle_detection(plate_text)
                if plate_text not in self.car_detected or not self.car_detected[plate_text]['detected']:
                    self.car_detected[plate_text] = {'detected': True, 'start_time': time.time()}
                    self.detection_signal.emit((plate_text, self.car_detected[plate_text]['start_time'], None))
                else:
                    elapsed_time = time.time() - self.car_detected[plate_text]['start_time']
                    self.car_detected[plate_text]['detected'] = False
                    self.detection_signal.emit((plate_text, time.time(), elapsed_time))

        return frame

    # ... (rest of the VideoThread code)

class CarDetectionProgram(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.video_thread = VideoThread("car_video_2.mp4")
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.detection_signal.connect(self.update_detection)
        self.video_thread.start()

    def init_ui(self):
        # ... (UI setup)
        self.setWindowTitle("Car Detection Program")
        self.setStyleSheet("background-color: #B0B0B0; color: white;")
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.table_widget = QTableWidget(self)
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Registration Plate", "Detection Time", "Elapsed Time"])
        # ... (rest of the UI styling)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.table_widget, alignment=Qt.AlignCenter)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setGeometry(100, 100, 1000, 800)
        self.table_widget.setMinimumWidth(800)
        self.table_widget.setMinimumHeight(600)
        self.table_widget.setColumnWidth(0, 300)
        self.table_widget.setColumnWidth(1, 250)
        self.table_widget.setColumnWidth(2, 250)

    def update_image(self, frame):
        # ... (image update)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_image)
        self.video_label.setPixmap(pixmap)

    def update_detection(self, detection_data):
        plate_text, detection_time, elapsed_time = detection_data
        row_count = self.table_widget.rowCount()
        self.table_widget.insertRow(row_count)
        self.table_widget.setItem(row_count, 0, QTableWidgetItem(plate_text))
        self.table_widget.setItem(row_count, 1, QTableWidgetItem(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(detection_time))))
        if elapsed_time is not None:
            self.table_widget.setItem(row_count, 2, QTableWidgetItem(f"{elapsed_time:.2f} seconds"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarDetectionProgram()
    window.show()
    sys.exit(app.exec())
