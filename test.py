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

    def __init__(self, video_path, car_detection_callback):
        super().__init__()
        self.car_cascade = cv2.CascadeClassifier("cars.xml")
        self.video_path = video_path
        self.running = True
        self.car_detection_callback = car_detection_callback  # Callback for adding data to table

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: can't open video capture.")
            sys.exit()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

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
                self.car_detection_callback(plate_text)  # Call the callback to add data to table

        return frame

    def detect_registration_plate(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(grey, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is None:
            return None

        mask = np.zeros(grey.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        cropped_image = grey[x1:x2 + 1, y1:y2 + 1]

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        if result:
            text = result[0][-2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, org=(location[0][0][0], location[1][0][1] + 60),
                        fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

            return text

        return None


class CarDetectionProgram(QMainWindow):
    def __init__(self):
        super().__init__()

        self.timer_label = None
        self.car_cascade = cv2.CascadeClassifier("cars.xml")
        self.start_time = None
        self.car_detected = False
        self.registration_plate = None
        self.car_timer = {}

        self.init_ui()

        # Create VideoThread and pass callback function to update the table
        self.video_thread = VideoThread("car_video_2.mp4", self.add_detection_to_table)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer_display)

    def init_ui(self):
        self.setWindowTitle("Car Detection Program")
        self.setStyleSheet("background-color: #B0B0B0; color: white;")

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        self.table_widget = QTableWidget(self)
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Registration Plate", "Detection Time", "Timer"])

        self.table_widget.setStyleSheet("""
            QTableWidget {
                background-color: #D3D3D3;
                color: white;
                border: 1px solid #CCCCCC;
                font-size: 16px;
            }
            QTableWidget::item {
                background-color: #4D4D4D;
                color: white;
                border: 1px solid #555555;
            }
            QTableWidget::horizontalHeader {
                background-color: #333333;
                color: white;
                font-size: 18px;
            }
        """)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.table_widget, alignment=Qt.AlignCenter)

        self.timer_label = QLabel("Timer: 0.00", self)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 20px; color: white;")
        layout.addWidget(self.timer_label, alignment=Qt.AlignCenter)

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
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_image)

        self.video_label.setPixmap(pixmap)

    def update_timer_display(self):
        if self.car_detected and self.start_time:
            elapsed_time = time.time() - self.start_time
            self.timer_label.setText(f"Timer: {elapsed_time:.2f} seconds")

    def start_timer(self):
        self.start_time = time.time()
        self.car_detected = True
        self.timer_label.setText("Timer: Running...")
        self.timer.start(1000)

    def stop_timer(self):
        if self.start_time:
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            self.timer_label.setText(f"Timer stopped - {elapsed_time:.2f} seconds")
            self.timer.stop()

    def add_detection_to_table(self, plate_text):
        """Add detected plate information to the table."""
        detection_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        timer_value = self.timer_label.text()

        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)

        # Insert data into the table
        self.table_widget.setItem(row_position, 0, QTableWidgetItem(plate_text))
        self.table_widget.setItem(row_position, 1, QTableWidgetItem(detection_time))
        self.table_widget.setItem(row_position, 2, QTableWidgetItem(timer_value))

        self.table_widget.resizeColumnsToContents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarDetectionProgram()
    window.show()
    sys.exit(app.exec())
