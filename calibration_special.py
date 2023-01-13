import cv2
from cv2 import aruco
import numpy as np
import os


def loadImagesToAugment(path)->dict:
    """
    :param path: name of the folder which contains all the marker images named w.r.t marker ids.
    :return: dictionary with key as marker id and value as the image to be augmented for that marker id.
    """

    images_list=os.listdir(path=path)
    noOfImages=len(images_list)
    print("No of images found : ",noOfImages)
    # print(images_list)
    augDict={}
    #the images are named with their marker id
    for image in images_list:
        augDict[int(image.split('.')[0])]=cv2.imread(f'{path}/{image}')

    # print(augDict)
    return augDict


def findArucoMarkers(img,markerSize=6,totalMarkers=250,draw=True)->list:
    """
    :param img: a single frame/image in the live webcam stream
    :param markerSize: size of the markers
    :param markerSize: total number of markers that compose the dictionary
    :param draw: flag to signal drawing of bounding box for the markers detected in the frame
    :return: list of bounding boxes(list) and their ids(list    )
    """

    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #convert the image to a grayscale
    # arucoDict=aruco.Dictionary_get(aruco.DICT_6X6_250)
    key=getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict=aruco.getPredefinedDictionary(key)
    arucoParam=aruco.DetectorParameters()

    #rejected bounding box is returned when the id is not decoded but the entity is detected as a marker 
    bboxs,ids,rejected_bboxs=aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)  

    # print(ids)
    if(draw):
        aruco.drawDetectedMarkers(img,bboxs)
        # print(bboxs)

    return [bboxs,ids]   #bbox and ids are of type list



def augmentAruco(bbox,id,img,augment_image,draw_id_on_image=True):
    """
    :param bbox: the four coordinates of the bouding box where marker is detected
    :param id: id of the marker
    :param img: a single frame/image in the live webcam stream
    :param augment_image: the image to be augmented
    :param draw_id_on_image: flag to signal the drawing of marker id value on the detected marker
    :return: frame/image with the augmented image overlayed
    """

    top_left=     bbox[0][0][0],bbox[0][0][1]
    top_right=    bbox[0][1][0],bbox[0][1][1]
    bottom_right= bbox[0][2][0],bbox[0][2][1]
    bottom_left=  bbox[0][3][0],bbox[0][3][1]

    h,w,d=augment_image.shape
    # print(h,w)

    pts1=np.array([top_left,top_right,bottom_right,bottom_left])
    pts2=np.array([[0,0],[w,0],[w,h],[0,h]])

    #the homography is a 3×3 matrix that maps the points in one point to the corresponding point in another image 
    matrix, _ = cv2.findHomography(pts2,pts1)
    #augment the image and will the other areas with complete black
    imgOutput = cv2.warpPerspective(augment_image, matrix, (img.shape[1],img.shape[0]))
    # print(img.shape[1],img.shape[0],"screen/frame height and width")
    # print(matrix)

    #fill the marker area as complete black so that we can overlay it with imgOutput
    cv2.fillConvexPoly(img,pts1.astype(int),(0,0,0))

    #overlay
    imgOutput=img+imgOutput

    # print(top_left)
    if(draw_id_on_image):
        cv2.putText(img=imgOutput,text=str(id),org=[int(top_left[0]),int(top_left[1])],
                    fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(0,0,255),thickness=2)


    return imgOutput




def main():
    cap = cv2.VideoCapture('/dev/video2')
    augDict=cv2.imread("assets/effrayant.png")
    while(1):
        success,img=cap.read()

        arucoFound = findArucoMarkers(img)   #get the postition of the bounding boxes and their ids
        
        #Loop through each marker and augment them
        if(len(arucoFound[0])!=0):   #if the length of the bboxs list is 0, then it means that no marker is detected
            
            for bbox,id in zip(arucoFound[0],arucoFound[1]):
                
                img=augmentAruco(bbox,id,img,augDict)

                # print(bbox,id)
                # # [[[413. 203.]
                # #   [410. 233.]
                # #   [380. 231.]
                # #   [383. 201.]]] [124]
        img = cv2.resize(img, (800, 800))
        cv2.imshow("Modified",img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
        
    



if __name__=="__main__":
    main()

