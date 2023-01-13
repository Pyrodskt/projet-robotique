import cv2
import numpy as np
from calibration import Calibration
import os

def display_markers_and_axes(images, camera_matrix, dist_coeff):
  # Convert the image to grayscale
  for image in images:
    img = cv2.imread(image)
        # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect aruco markers in the image
    corners, _, _ = cv2.aruco.detectMarkers(gray, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50))
    # If markers are detected, draw them on the image and display the image
    if corners:
        img = cv2.aruco.drawDetectedMarkers(img, corners)
    # If no markers are detected, print an error message
    else:
        print("No markers detected")
    # Iterate through each object in the image
    for corner in corners:
        # Estimate the pose of the object using the detected markers
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeff)
        # Draw the axis of the object on the image
        img = cv2.drawFrameAxes(img, camera_matrix, dist_coeff, rvec, tvec, 0.05, 2)
        # Display the image
        cv2.imwrite('assets/output/' + image.split('/')[3], img)
def run():
    cap = cv2.VideoCapture('/dev/video0')
    while cap.isOpened():
        ret, img = cap.read()
            # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect aruco markers in the image
        corners, _, _ = cv2.aruco.detectMarkers(gray, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50))

        # Iterate through each object in the image
        for corner in corners:
            # Estimate the pose of the object using the detected markers
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeff)
            # Draw the axis of the object on the image
            img = cv2.drawFrameAxes(img, camera_matrix, dist_coeff, rvec, tvec, 0.05, 2)
            # Display the image
        cv2.imshow("Axes", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
            
def read_chessboards(images):
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
    # Calculate the distortion of the image by looking for know aruco in the picture.
    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray, dict_aruco)


        res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,charuco_board)
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

def calibrate_cam(images, charuco_dict):
    corners_list = []
    charuco_ids_list = []
    arucoParams = cv2.aruco.DetectorParameters()
    # Iterate through each image
    for image in images:
        img = cv2.imread(image)
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect charuco corners in the image
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, charuco_dict, parameters=arucoParams)
        # If corners are detected, add them to the list
        
    
    image_size = gray.shape[::-1]
    allCorners,allIds,imsize, gray=read_chessboards(images)
    ret, camera_matrix, dist_coeff, rvec, tvec = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,charuco_board,image_size,None,None, criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5))
    # Print the calibration results
    print("Camera matrix:", camera_matrix)
    print("Distortion coefficients:", dist_coeff)
    return camera_matrix, dist_coeff

# Read in the image
#image = cv2.imread("./assets/input/2023-01-06-134435.jpg")
# Define the objects in the image
images = ['./assets/input/' + f for f in os.listdir('./assets/input') if f.endswith(".jpg")]
dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
charuco_board = cv2.aruco.CharucoBoard((3, 3), 0.07, 0.05, dict_aruco)
charuco_ids = 3

camera_matrix, dist_coeff = calibrate_cam(images, dict_aruco)
display_markers_and_axes(images, camera_matrix, dist_coeff)
run()