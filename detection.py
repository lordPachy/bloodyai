import cv2


def detect(img):
    # MASK
    mask = cv2.imread('./detection/mask.jpg')
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
