import cv2
import numpy as np
import os

def detect(filename: str):
    img = cv2.imread(filename)

    #MASK
    mask = cv2.imread('./mask.jpg')
    img = cv2.bitwise_and(img, mask)

    #FILTER
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = hsv[:,:,1]
    hsv[:,:,2] = hsv[:,:,1]
    hsv[hsv < 140] = 0
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    #BLOB DETECTOR PARAMETERS
    params = cv2.SimpleBlobDetector.Params()

    params.minThreshold = 80 #80
    params.maxThreshold = 255 #255
    params.minDistBetweenBlobs = 250 #250

    params.filterByArea = True
    params.minArea = 35

    #PARAMS THAT CAN BE REMOVED
    params.filterByCircularity = False
    params.minCircularity = 0.1

    params.filterByConvexity = False
    params.minConvexity = 0.2

    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    #-----------------------------

    #DETECT BLOB KEYPOINTS
    detector = cv2.SimpleBlobDetector.create()
    detector.setParams(params)
    keypoints = detector.detect(img)

    '''
    for keypoint in keypoints:
        print("Coords: " + str(keypoint.pt) + " Size:" + str(keypoint.size))'''

    img_keypts = cv2.drawKeypoints(img, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



    cv2.imwrite("./detection_results/" + filename.split("/")[-1][:-4] + "_" + str(len(keypoints)) + ".jpg", img_keypts)
    #cv2.imshow("Image", img_keypts)
    #cv2.waitKey(0)

if __name__ == "__main__":
    folder_path = "../images"
    images_dir = os.listdir(folder_path)
    for image_file in images_dir:
        image_path = os.path.join(folder_path + '/' + image_file)
        detect(image_path)