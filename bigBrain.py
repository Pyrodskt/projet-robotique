import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Normalizer

from PIL import Image

def getTrainImg():

    X_train=[]
    y_train=[]
    
    w=300
    h=300

    images = ['./picture/calib/' + f for f in os.listdir('./picture/calib/') if f.endswith(".png")]
    print("\nImages: ",images[0:5])
    for im in tqdm(images):

        #img = Image.open(im).convert('1').resize((w,h)) # convert image to black/white
        img = Image.open(im).convert('L').resize((w,h)) # convert image to grayscale
        #img = Image.open(im).resize((w,h)) # convert image to grayscale

        arr = np.array(img)
        arr = arr.ravel() # flatten 2D array to 1D
        X_train.append(arr)
        y_train.append(1)

    images = ['./picture/notcalib/' + f for f in os.listdir('./picture/notcalib/') if f.endswith(".png")]
    for im in tqdm(images):
        #img = Image.open(im).convert('1').resize((w,h)) # convert image to black/white
        img = Image.open(im).convert('L').resize((w,h)) # convert image to grayscale
        arr = np.array(img)
        arr = arr.ravel() # flatten 2D array to 1D
        X_train.append(arr)
        y_train.append(0)
            
    random.shuffle(X_train)
    random.shuffle(y_train)
    return np.array(X_train),np.array(y_train)


def getTrainData():
    X_train=[]
    y_train=[]

    DetectedArucoCount=0

    images = ['./picture/calib/' + f for f in os.listdir('./picture/calib/') if f.endswith(".png")]
    print("\nImages: ",images[0:5])
    for im in images:

        frame = cv2.imread(im)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect aruco markers in the image
        corners, _, _ = cv2.aruco.detectMarkers(gray, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50))
        
        # Iterate through each object in the image
        for corner in corners:
            
            # Estimate the pose of the object using the detected markers
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.02, camera_matrix, dist_coeff)
            DetectedArucoCount+=1

            rvec=[round(val, 4) for val in rvec[0][0]]
            tvec=[round(val, 4) for val in tvec[0][0]]
            #newX=list(x for xs in zip(rvec, tvec) for x in xs)
            newX=list(x for xs in zip(tvec) for x in xs)
        X_train.append(newX)
        y_train.append(1)

    images = ['./picture/notcalib/' + f for f in os.listdir('./picture/notcalib/') if f.endswith(".png")]
    for im in images:
        frame = cv2.imread(im)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect aruco markers in the image
        corners, _, _ = cv2.aruco.detectMarkers(gray, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50))
        newX=([-10,-10,-10])#,-10,-10,-10])
        newY=-1
        # Iterate through each object in the image
        for corner in corners:            
            # Estimate the pose of the object using the detected markers
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.02, camera_matrix, dist_coeff)
            DetectedArucoCount+=1
            
            #print("\nRVEC/TVEC: ",rvec[0:5],tvec[0:5])
            

            rvec=[round(val, 4) for val in rvec[0][0]]
            tvec=[round(val, 4) for val in tvec[0][0]]
            #newX=list(x for xs in zip(rvec, tvec) for x in xs)
            newX=list(x for xs in zip(tvec) for x in xs)

            newY=0
            X_train.append(newX)
            y_train.append(newY)
            
    print("\nDetected Arucos: ",DetectedArucoCount)        
    random.shuffle(X_train)
    random.shuffle(y_train)
    return np.array(X_train),np.array(y_train)

def run():
    #cap = cv2.VideoCapture(2)
    cap = cv2.VideoCapture(0)
    X_test=[]
    while cap.isOpened():
        ret, img = cap.read()

            # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect aruco markers in the image
        corners, _, _ = cv2.aruco.detectMarkers(gray, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50))
        # If markers are detected, draw them on the image and display the image
        if corners:
            img = cv2.aruco.drawDetectedMarkers(img, corners)
        # Iterate through each object in the image
        for corner in corners:
            # Estimate the pose of the object using the detected markers
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.02, camera_matrix, dist_coeff)
            # Draw the axis of the object on the image
            #img = cv2.drawFrameAxes(img, camera_matrix, dist_coeff, rvec, tvec, 0.02, 2)
            #print(tvec[0][0],rvec[0][0])

            rvec=[round(val, 4) for val in rvec[0][0]]
            tvec=[round(val, 4) for val in tvec[0][0]]
            newX=list(x for xs in zip(rvec, tvec) for x in xs)
            X_test.append(newX)
            # Make predictions on the test set
            #prediction = mlp.predict(newX)
            #print(prediction)
	            
            '''# Define the ar cube
    		# Since we previously set a matrix size of 1x1 for the marker and we want the cube to be the same size, it is also defined with a size of 1x1x1
    		# It is important to note that the center of the marker corresponds to the origin and we must therefore move 0.5 away from the origin 
            axis = np.float32([[-0.02, -0.02, 0.01], [-0.02, 0.02, 0.01], [0.02, 0.02, 0.01], [0.02, -0.02, 0.01],
    						   [-0.01, -0.01, 0.05], [-0.01, 0.01, 0.05], [0.01, 0.01, 0.05],[0.01, -0.01, 0.05]])
    		# Now we transform the cube to the marker position and project the resulting points into 2d
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeff)
            imgpts = np.int32(imgpts).reshape(-1, 2)

    		# Now the edges of the cube are drawn thicker and stronger
            img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)
            for i, j in zip(range(4), range(4, 8)):
                img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 2)
            img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)
    		
    	# Display the result      
        img = cv2.resize(img, (960, 540))                   
        cv2.imshow("Axes", img)'''
        
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break
            
def getCalibration():
    calib = np.load('calib.npz')
    ret = calib['ret']
    camera_matrix = calib['cam_matrix']
    dist_coeff = calib['dist_coef']
    r_vec = calib['r_vec']
    t_vec = calib['t_vec']
    return ret, camera_matrix, dist_coeff, r_vec, t_vec
'''

# Define the objects in the image
dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
charuco_board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.02, dict_aruco)

#camera_matrix, dist_coeff = calibrate_cam(images, dict_aruco)
ret, camera_matrix, dist_coeff, r_vec, t_vec = getCalibration()

'''


#get data
X_train,y_train=getTrainImg()

print("\nData: \n",X_train[0:5],y_train[0:5])
print("\nType: ",type(X_train))


#Standardization
mean=X_train.mean()
X_train = (X_train - mean)/(X_train.std())

#Info
print("\nMean: ",mean)
print("\nStandardized Data: \n",X_train[0:5],y_train[0:5],"\n")


#split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    train_size=0.750, test_size=0.250,random_state=42)
'''

#TPOT CLASSIFIER
pipeline_optimizer = TPOTClassifier(generations=3, population_size=10, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')

predictions = pipeline_optimizer.predict(X_test)
cm = confusion_matrix(y_test, predictions)
print(cm)
'''
'''
#MLP GRID SEARCH CLASSIFIER
#mlp = MLPClassifier(hidden_layer_sizes=(100,200,150,100), max_iter=1000 ,random_state=42)
mlp = MLPClassifier(max_iter=100,random_state=42,verbose=True)
parameter_space = {
    'hidden_layer_sizes': [(500,520,200)],#(35,45,30),(100,200,100),
    'activation': ['tanh', 'relu'],
    'solver': ['adam','sgd'],
    'alpha': [0.0005, 0.005,0.0001],
    'learning_rate': ['constant','adaptive']}

clf = GridSearchCV(mlp, parameter_space, cv=5)
clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels
print('\nBest parameters found:\n', clf.best_params_,"\n")
print("\nScore:",clf.score(X_test, y_test))
predictions = clf.predict(X_test)
# Compute the confusion matrix
cm = confusion_matrix(y_test, predictions)
print(cm)
'''


'''
'''
#MLP GRID SEARCH CLASSIFIER
# Entraîner le classifieur MLP sur les données d'entraînement
mlp = MLPClassifier(hidden_layer_sizes=(600,610,300,200), max_iter=500 ,alpha=0.0001,random_state=42,learning_rate_init=0.001,verbose=3)
mlp.fit(X_train, y_train)


#synapsView(clf.best_estimator_)

# Make predictions on the test set
predictions=mlp.predict(X_test)
# Compute the confusion matrix
print("\nScore:",mlp.score(X_test, y_test))

cm = confusion_matrix(y_test, predictions)
print(cm)


#run()