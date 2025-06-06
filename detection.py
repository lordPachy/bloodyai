import cv2

'''
<summary> Detection
Input:
- img: a numpy array of the input image
Output: 
- coordinates: the coordinates of the centroids of the detected cells
Description:
This function takes an input image, masks out the exterior of the scope,
applies a saturation filter to it and uses SimpleBlobDetector with the set parameters
to detect and return the white blood cells in the image.
</summary>
'''


def detect(img):
    # MASK
    mask = cv2.imread('./mask.jpg')
    img = cv2.bitwise_and(img, mask)

    # FILTER
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 1]
    hsv[:, :, 2] = hsv[:, :, 1]
    hsv[hsv < 140] = 0
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # BLOB DETECTOR PARAMETERS
    params = cv2.SimpleBlobDetector.Params()

    params.minThreshold = 80  # 80
    params.minDistBetweenBlobs = 250  # 250

    params.filterByArea = True
    params.minArea = 35

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    # -----------------------------

    # DETECT BLOB KEYPOINTS
    detector = cv2.SimpleBlobDetector.create()
    detector.setParams(params)
    keypoints = detector.detect(img)

    coordinates = []
    for keypoint in keypoints:
        coordinates.append(keypoint.pt)
    return coordinates
