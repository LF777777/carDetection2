import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

from pathlib import Path
import pytest  # use this for testing the gui and other parts
import sys
import sqlite3
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer


class CarDetectionProgram(QMainWindow):
    def __init__(self):
        super().__init__()

        self.timer_label = None
        # self.car_cascade = Path("./cars2.xml")
        # self.cascade = cv2.CascadeClassifier(car_cascade)
        #  self.timer_gui = None
        self.start_time = 0
        self.end_time = 0
        self.car_detected = False
        self.car_position = None

        self.cap = cv2.VideoCapture(0)  # takes default camera
        #if not self.cap.isOpened():
        #    print("Error: can't open video capture.")
        #    sys.exit()
        # Creating a timer

        # self.timer_gui = QTimer(self)
        # self.timer_gui.start(1000)

        #   self.db_connection = sqlite3.connect('car_detection.db')
        #   self.db_cursor = self.db_connection.cursor()

        #   self.db_cursor.execute("CREATE TABLE car(image, registartion plate)")
        #   self.db_cursor.execute("""
        #    INSERT INTO car VALUES
        #        f"{self.timer_gui_start}"
        #   """)

    # how can I check in this code by puttign in a conditional whthner or not car is detcetde
    def detection(self, frame):
        cars = self.cascade.detectMultiScale(frame, 1.1, 4)
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            cv2.putText(frame, "vehicle detected", (x+w, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
            self.car_detected = True

            if not self.car_detected:   # if car is
                self.start_timer()
            else:
                self.stop_timer()

        return frame

    def detect_registration_plate(self):  # put in frame later in

        global approx
        img = cv2.imread("cars/image1.png")
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # recoloured image
        plt.imshow(cv2.cvtColor(grey, cv2.COLOR_BGR2RGB))  #

        # apply filtering anf find edges for localozation, the filtering alllows us remove noise form image edge detection detect edges on image

        bfilter = cv2.bilateralFilter(grey, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)
        plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # sorting algorithm

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        mask = np.zeros(grey.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = grey[x1:x2+1, y1:y2+1]

        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        # Read number plate
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        text = result[0][-2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1,
                          color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

    def capture_screen(self):
        real_time_video = cv2.VideoCapture(0)
        while real_time_video.isOpened():
            ret, frame = real_time_video.read()
            control_key = cv2.waitKey(1)
            if ret:
                vehicle_frame = self.detection(frame)
                cv2.imshow("vehicle detection", vehicle_frame)
            else:
                break

            if control_key == ord("q"):
                break

    def store_car_data_db(self):
        pass

    def init_ui(self):
        self.setWindowTitle("Car detection Test")
        self.timer_label = QLabel("Timer: Not Started", self)
        layout = QVBoxLayout()
        layout.addWidget(self.timer_label)

    def start_timer(self):
        self.start_time = time.time()
        self.car_detected = True
        self.timer_label.setText("Timer: Running.......")

    def stop_timer(self):
        if self.start_time:
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            self.timer_label.setText(f"Timer stopped - {elapsed_time: .2f} seconds")

    def update_timer_display(self):
        if self.car_detected:
            elapsed_time = self.end_time - self.start_time
            self.timer_label.setText(f"Timer running - {elapsed_time: -2f} seconds")
            self.start_time = None

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = self.cascade.detectMultiScale(gray, 1.1, 1)

        # if car detected start a timer
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if not self.car_detected:
            self.start_timer()
        else:
            if self.car_detected:
                self.stop_timer()

    def run(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Cant grab frame")
                break

            processed_frame = self.detection(frame)

            cv2.imshow("Car detection", processed_frame)

            if cv2.waitKey(1):
                break


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarDetectionProgram()
    print(window.detect_registration_plate())
    sys.exit(app.exec())
    # import cv2
    # from pathlib import Path
    # app = QApplication(sys.argv)
    # window = CarDetectionProgram()
    # window.show()
    # window.run()
    # path_cascade = Path("cars2.xml")
    # print(path_cascade.open())

    # try:
    #    car_cascade = Path("cars2.xml")
    #    cascade = cv2.CascadeClassifier(car_cascade)

    # except Exception as e:
    #    print(f"An error occurred: {e}")
    # sys.exit(app.exec())





