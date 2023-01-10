import numpy as np
import time
import cv2
import os
from calibration import Calibration

class ArucoDetector:
    def __init__(self) -> None:
        
        self.ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }
        

        self.aruco_type = "DICT_7X7_100"

        self.arucoDict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT[self.aruco_type])
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.calibration = Calibration()
        self.calibration.calibrate()
        self.ret, self.camera_matrix, self.dist_coeff, self.r_vec, self.t_vec = self.calibration.getCalibration()
        self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)
        self.cap = cv2.VideoCapture('/dev/video0')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.prev_frame_time = 0
        # used to record the time at which we processed current frame
        self.new_frame_time = 0


    def aruco_display(self, corners, ids, rejected, image):
        if len(corners) > 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(image.copy(), self.arucoDict, parameters=self.arucoParams)
            rvec, tvec, _= cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, self.camera_matrix, self.dist_coeff)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
            for corner in corners:
                cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)

            img = cv2.aruco.drawDetectedMarkers(image, corners, ids)

            

            
            total_markers = range(0, len(ids))
            image2 = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
            for i in range(len(tvec)):
                image = cv2.drawFrameAxes(image2, self.camera_matrix, self.dist_coeff, rvec[i], tvec[i], 0.5, 1)
               
        return image

        
        
    def run(self):

        
        while self.cap.isOpened():
            
            ret, img = self.cap.read()

            h, w, _ = img.shape
            
            width = 1000
            height = int(width*(h/w))
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            corners, ids, rejected = cv2.aruco.detectMarkers(img.copy(), self.arucoDict, parameters=self.arucoParams)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.calculateFPS()
            img = self.aruco_display(corners, ids, rejected, img)
            cv2.imshow('image', img) 
            time.sleep(0.05)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
        self.cap.release()
    
    def calculateFPS(self):
            self.new_frame_time = time.time()
            
            # Calculating the fps
            
            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
            fps = 1/(self.new_frame_time-self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time
            
            # converting the fps into integer
            fps = int(fps)
            
            # converting the fps to string so that we can display it on frame
            # by using putText function
            fps = str(fps)
            print(fps)

arucomachine = ArucoDetector()
arucomachine.run()