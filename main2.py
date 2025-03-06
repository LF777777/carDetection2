import cv2
import time
import numpy as np
import easyocr
import sqlite3
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QTimer, Qt
import sys


class CarDetectionProgram(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize variables
        self.timer_label = None
        self.car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')  # Use OpenCV's pre-trained car cascade
        self.start_time = None
        self.car_detected = False
        self.registration_plate = None
        self.car_timer = {}  # Dictionary to track timers for each detected car

        # Setup Video Capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: can't open video capture.")
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
        self.table_widget.setColumnWidth(2, 250)  # Set column 3 (Timer) width to 250px


    def detection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Vehicle Detected", (x + w, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            plate_text = self.detect_registration_plate(frame[y:y+h, x:x+w])
            if plate_text:
                self.handle_detection(plate_text)

            return frame

        return frame

    def detect_registration_plate(self, frame):
        # Use EasyOCR to read the license plate
        reader = easyocr.Reader(['en'])
        result = reader.readtext(frame)
        if result:
            # Assuming the first result is the license plate
            text = result[0][-2]
            self.registration_plate = text
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

    def update_timer_display(self):
        # Update the timer display in the GUI
        for plate, data in self.car_timer.items():
            if data['timer_running']:
                elapsed_time = time.time() - data['start_time']
                self.timer_label.setText(f"Timer: {elapsed_time:.2f} seconds for {plate}")

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

    def capture_screen(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self.detection(frame)

            cv2.imshow("Car Detection", processed_frame)

            if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.capture_screen()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarDetectionProgram()
    window.show()
    sys.exit(app.exec())