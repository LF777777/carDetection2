import cv2
import numpy as np
import easyocr
import imutils
import sqlite3
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QWidget
from PyQt5.QtCore import QTimer, Qt
import sys
import time


class CarDetectionProgram(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize variables
        self.car_cascade = cv2.CascadeClassifier("cars.xml")  # Use OpenCV's pre-trained car cascade
        self.reader = easyocr.Reader(['en'])  # EasyOCR reader for OCR
        self.timer_label = None
        self.car_detected = False
        self.registration_plate = None
        self.car_timer = {}  # Dictionary to track timers for each detected car
        self.start_time = None

        # Setup Video Capture
        self.cap = cv2.VideoCapture('car_video.mp4')  # Use the path to your video file
        if not self.cap.isOpened():
            print("Error: can't open video.")
            sys.exit()

        # Create a database to store detected car registration numbers
        self.db_connection = sqlite3.connect('car_detection.db')
        self.db_cursor = self.db_connection.cursor()
        self.db_cursor.execute("""
            CREATE TABLE IF NOT EXISTS car_detection (
                registration_plate TEXT,
                detection_time REAL
            )
        """)
        self.db_connection.commit()

        # GUI Setup
        self.init_ui()

        # Start timer to update GUI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer_display)

    def init_ui(self):
        # Set the main window title and background color
        self.setWindowTitle("Car Detection Program")
        self.setStyleSheet("background-color: #B0B0B0; color: white;")  # Set greyish background and white text

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
        layout.addWidget(self.table_widget, alignment=Qt.AlignCenter)  # This centers the table

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

    def start_timer(self):
        self.start_time = time.time()
        self.car_detected = True  # Mark that the car has been detected and timer started
        self.timer_label.setText("Timer: Running...")
        self.timer.start(1000)  # Update the timer every second

    def stop_timer(self):
        if self.start_time:
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            self.timer_label.setText(f"Timer stopped - {elapsed_time:.2f} seconds")
            self.store_car_data_db(self.registration_plate, elapsed_time)  # Store with elapsed time

    def car_detection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(gray, 1.1, 4)

        if len(cars) == 0:
            return frame

        for (x, y, w, h) in cars:
            # Draw a bounding box around the detected car
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Extract the region of interest (ROI)
            roi = frame[y:y + h, x:x + w]

            # Detect the license plate in the ROI
            plate_text = self.detect_registration_plate(roi)
            if plate_text:
                self.handle_detection(plate_text)

        return frame

    def detect_registration_plate(self, frame):
        # Convert the frame to grayscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filtering to remove noise while keeping edges sharp
        bfilter = cv2.bilateralFilter(grey, 11, 17, 17)

        # Apply Canny edge detection to detect edges
        edged = cv2.Canny(bfilter, 30, 200)

        # Find contours in the edged image
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)

        # Sort contours by area in descending order and take the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:  # A rectangle will have 4 corners (license plate)
                location = approx
                break

        if location is None:
            return None  # No license plate found

        # Create a mask to isolate the license plate
        mask = np.zeros(grey.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)

        # Get coordinates for cropping the license plate area
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        # Crop to the region of interest (license plate area)
        cropped_image = grey[x1:x2 + 1, y1:y2 + 1]

        # Use EasyOCR to detect the text in the cropped region
        result = self.reader.readtext(cropped_image)

        if result:
            text = result[0][-2]

            # Draw the rectangle around the detected license plate and add the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, org=(location[0][0][0], location[1][0][1] + 60),
                        fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

            return text

        return None

    def handle_detection(self, plate_text):
        # Handle the timer and registration plate tracking logic
        current_time = time.time()

        if plate_text not in self.car_timer:
            # If the car is detected for the first time, start a new timer
            self.car_timer[plate_text] = {'start_time': current_time, 'timer_running': True}
            self.timer_label.setText(f"Timer: Running... Car Plate: {plate_text}")
            self.timer.start(1000)  # Update every second
        else:
            # If the car has been detected before, stop the timer
            if self.car_timer[plate_text]['timer_running']:
                elapsed_time = current_time - self.car_timer[plate_text]['start_time']
                self.car_timer[plate_text]['timer_running'] = False
                self.store_car_data_db(plate_text, elapsed_time)
                self.timer_label.setText(f"Timer stopped for {plate_text} - {elapsed_time:.2f} seconds")

    def store_car_data_db(self, plate_text, elapsed_time):
        self.db_cursor.execute("INSERT INTO car_detection (registration_plate, detection_time) VALUES (?, ?)",
                               (plate_text, elapsed_time))
        self.db_connection.commit()
        self.update_table_widget()  # Update the table when new data is stored

    def update_table_widget(self):
        # Fetch all data from the database and update the table
        self.db_cursor.execute("SELECT registration_plate, detection_time FROM car_detection")
        rows = self.db_cursor.fetchall()

        # Clear the existing data in the table
        self.table_widget.setRowCount(0)

        # Insert the fetched data into the table
        for row in rows:
            row_position = self.table_widget.rowCount()
            self.table_widget.insertRow(row_position)
            self.table_widget.setItem(row_position, 0, QTableWidgetItem(row[0]))  # Registration Plate
            self.table_widget.setItem(row_position, 1, QTableWidgetItem(f"{row[1]:.2f}"))  # Detection Time
            self.table_widget.setItem(row_position, 2, QTableWidgetItem(f"{time.time() - row[1]:.2f}"))  # Timer

    def capture_screen(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process the frame with car detection
            processed_frame = self.car_detection(frame)

            # Display the frame with car detection and license plate detection
            cv2.imshow("Car Detection", processed_frame)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_timer_display(self):
        # Update the timer label every second
        if self.car_detected and self.start_time:
            elapsed_time = time.time() - self.start_time
            self.timer_label.setText(f"Timer: {elapsed_time:.2f} seconds")
        else:
            self.timer_label.setText("Timer: Not Started")


# Example usage:
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarDetectionProgram()
    window.show()
    window.capture_screen()  # Start video processing
    sys.exit(app.exec_())