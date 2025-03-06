import cv2
import time
import imutils
import numpy as np
import easyocr
import sqlite3
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import sys
from PyQt5.QtGui import QImage, QPixmap


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, video_path):
        super().__init__()
        self.car_cascade = cv2.CascadeClassifier("cars.xml")
        self.video_path = video_path
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: can't open video capture.")
            sys.exit()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform car detection and license plate extraction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = self.car_cascade.detectMultiScale(gray, 1.1, 9)

            for (x, y, w, h) in cars:
                plate = frame[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 51, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (51, 51, 255), -2)
                cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('car', plate)

                # Detect the registration plate
                plate_text = self.detect_registration_plate(frame[y:y + h, x:x + w])
                if plate_text:
                    if not self.car_detected:  # Start the timer when the first car is detected
                        self.start_timer()  # Start the timer if the car is detected for the first time
                    else:  # Stop the timer when the same car is detected again
                        self.stop_timer()

                    # Handle the detection of the license plate and store data
                    self.handle_detection(plate_text)

            # Emit signal to update the video frame in the GUI
            self.change_pixmap_signal.emit(frame)

            # Sleep to match the frame rate of the video
            time.sleep(0.03)  # Adjust this to match video frame rate

        cap.release()

    def detect_registration_plate(self, frame):
        # Convert the frame to grayscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filtering to remove noise while keeping edges sharp
        bfilter = cv2.bilateralFilter(grey, 11, 17, 17)

        # Apply Canny edge detection
        edged = cv2.Canny(bfilter, 30, 200)

        # Find contours in the edged image
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)

        # Sort contours by area in descending order and take top 10
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        # Iterate through contours to find a quadrilateral contour
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)  # Approximate the contour with fewer points
            if len(approx) == 4:  # A rectangle will have 4 corners
                location = approx
                break

        if location is None:
            return None  # No license plate found

        # Create a mask to isolate the region of interest (license plate)
        mask = np.zeros(grey.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)

        # Get coordinates for cropping the image to the license plate region
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        # Crop the image to the region of interest (license plate area)
        cropped_image = grey[x1:x2 + 1, y1:y2 + 1]

        # Use EasyOCR to detect the text (registration plate) in the cropped region
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        if result:
            # Extract the text from OCR result
            text = result[0][-2]

            # Draw the rectangle around the detected license plate and add the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, org=(location[0][0][0], location[1][0][1] + 60),
                        fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

            # Return the detected registration plate text
            return text

        return None

    def handle_detection(self, plate_text):
        # Handle the detection of a car's license plate here (e.g., store data, show in GUI, etc.)
        print(f"Detected plate: {plate_text}")


class CarDetectionProgram(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize variables
        self.timer_label = None
        self.car_detected = False
        self.start_time = None

        # Setup GUI
        self.init_ui()

        # Create VideoThread and connect signals
        self.video_thread = VideoThread("car_video_2.mp4")  # Replace with your video file path
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

        # Start timer to update GUI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer_display)  # Connect the timer to update the display every second

    def init_ui(self):
        # Set the main window title and background color
        self.setWindowTitle("Car Detection Program")
        self.setStyleSheet("background-color: #B0B0B0; color: white;")  # Set greyish background and white text

        # Create a label to display the video frames
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")  # Set a black background for the video label

        # Create a table to display the detection data
        self.table_widget = QTableWidget(self)
        self.table_widget.setRowCount(0)  # Initially, no rows
        self.table_widget.setColumnCount(3)  # Three columns (Plate, Detection Time, Timer)

        # Set column headers
        self.table_widget.setHorizontalHeaderLabels(["Registration Plate", "Detection Time", "Timer"])

        # Set table background, header, and item (column) styles
        self.table_widget.setStyleSheet("""
            QTableWidget {
                background-color: #D3D3D3;  # Light grey background for the table
                color: white;
                border: 1px solid #CCCCCC;  # Outer border around the entire table
                gridline-color: #CCCCCC;  # Light grey lines between rows and columns
                font-size: 16px;  # Increase font size for readability
            }
            QTableWidget::item {
                background-color: #4D4D4D;  # Dark grey background for the cells (columns)
                color: white;
                border: 1px solid #555555;  # Border around each cell (light grey)
            }
            QTableWidget::horizontalHeader {
                background-color: #333333;  # Dark grey header background
                color: white;
                font-size: 18px;  # Larger font size for header
                padding: 5px;
            }
            QTableWidget::horizontalHeader::section {
                padding-left: 10px;
                padding-right: 10px;
                border: none;
            }
            QHeaderView::section {
                background-color: #333333;  # Dark grey for the header section (column names)
                color: white;
            }
        """)

        # Center the table in the window and increase size
        layout = QVBoxLayout()
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)  # Add the video label to the layout
        layout.addWidget(self.table_widget, alignment=Qt.AlignCenter)  # Add the table widget to the layout

        # Create a label to show the timer
        self.timer_label = QLabel("Timer: 0.00", self)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 20px; color: white;")
        layout.addWidget(self.timer_label, alignment=Qt.AlignCenter)  # Add the timer label to the layout

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Set a larger window size and ensure table expands accordingly
        self.setGeometry(100, 100, 1000, 800)  # Larger window size (1000x800)

        # Make the table widget larger
        self.table_widget.setMinimumWidth(800)  # Set minimum width for table
        self.table_widget.setMinimumHeight(600)  # Set minimum height for table

        # Adjust column widths for more space
        self.table_widget.setColumnWidth(0, 300)  # Set column 1 (Plate) width to 300px
        self.table_widget.setColumnWidth(1, 250)  # Set column 2 (Detection Time) width to 250px
        self.table_widget.setColumnWidth(2, 250)  # Set column 3 (Timer)

    def update_image(self, frame):
        """Update the image displayed on the GUI."""
        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_image)

        # Update the video display label
        self.video_label.setPixmap(pixmap)

    def update_timer_display(self):
        """Update the timer label."""
        if self.car_detected and self.start_time:
            # Calculate the elapsed time
            elapsed_time = time.time() - self.start_time
            # Update the timer label with the new elapsed time (convert to seconds)
            self.timer_label.setText(f"Timer: {elapsed_time:.2f} seconds")

    def start_timer(self):
        self.start_time = time.time()  # Store the start time
        self.car_detected = True  # Mark that a car has been detected and the timer has started
        self.timer_label.setText("Timer: Running...")
        self.timer.start(1000)  # Start the timer to update every second

    def stop_timer(self):
        if self.start_time:
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            self.timer_label.setText(f"Timer stopped - {elapsed_time:.2f} seconds")
            self.timer.stop()  # Stop the timer when the car detection ends


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarDetectionProgram()
    window.show()
    sys.exit(app.exec())