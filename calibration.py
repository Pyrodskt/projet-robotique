import cv2
import os
import numpy as np
import json


class Calibration:
    def __init__(self) -> None:
        
        # Init the Aruco list and the boad for calibration
        # partie calibration
        self.dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
        self.board = cv2.aruco.CharucoBoard((3, 3), 0.07, 0.05, self.dict_aruco)
        self.images = ['./assets/input/' + f for f in os.listdir('./assets/input') if f.endswith(".jpg")]

    def read_chessboards(self, images):
        """
        Charuco base pose estimation.
        """
        print("POSE ESTIMATION STARTS:")
        allCorners = []
        allIds = []
        decimator = 0
        objPoints = []
        imgPoints = []
        objp = np.zeros((1, 3 * 3, 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:3, 0:3].T.reshape(-1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.05)
        prev_img_shape = None
        # Calculate the distortion of the image by looking for know aruco in the picture.
        for im in images:
            print("=> Processing image {0}".format(im))
            frame = cv2.imread(im)
            print(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.aruco.detectMarkers(gray, self.dict_aruco)

            # retval, corners = cv2.findChessboardCorners(gray, (3, 3), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            # if retval == True:
                # objPoints.append(objp)
                # corners2 = cv2.cornerSubPix(gray, corners, (5,5),(-1,-1), criteria)
                # imgPoints.append(corners2)
                # frame = cv2.drawChessboardCorners(frame, (3, 3), corners2, retval)
                # si des aruco sont détectés
            res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,self.board)
            # localise le coin (0,0) du aruco
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%4==0:
                # si il respecte toute les conditions c'est le point (0,0)
                allCorners.append(res2[1])
                allIds.append(res2[2])
            # Draw localised marker    
            frame = cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])

            # Write the output images in another location
            cv2.imwrite('assets/output/' + im.split('/')[3], frame)
            
            decimator+=1

        return allCorners, allIds, decimator, gray

    def calibrate(self):
        allCorners,allIds,imsize, gray=self.read_chessboards(self.images)
        imsize = gray.shape
        print(imsize)
        # Calibration can fail for so many reasons (Because the Universe doesn't want it)
        try:
            # calculate calibration data and save them to calib.npz file
            ret, camera_matrix, dist_coeff, rvec, tvec = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,self.board,imsize,None,None, criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5))
            # Sauvegarde des paramètres dans calib.npz
            np.savez("calib.npz", ret=ret, cam_matrix=camera_matrix, dist_coef=dist_coeff, r_vec=rvec, t_vec=tvec)


        except Exception as e:
            print('Y a un truc qui a planté', e)
            pass


    def getCalibration(self):
        calib = np.load('calib.npz')
        ret = calib['ret']
        camera_matrix = calib['cam_matrix']
        dist_coeff = calib['dist_coef']
        r_vec = calib['r_vec']
        t_vec = calib['t_vec']
        return ret, camera_matrix, dist_coeff, r_vec, t_vec