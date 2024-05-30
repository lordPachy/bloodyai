import recognition
import cv2
import detection


def detect_and_classify(filepath: str):
    # LOAD DATA
    model = recognition.prepare_ResNet18('./classification/parameters/sgd.pt')
    img = cv2.imread(filepath)
    results = dict()

    # PERFORM DETECTION AND CLASSIFICATION
    centroids = detection.detect(img)
    for coordinates in centroids:
        x_left = int(coordinates[0]) - 287
        y_top = int(coordinates[1]) - 287
        cell_crop = img[y_top:y_top + 575, x_left:x_left + 575, :]
        #print(cell_crop.shape)
        label = recognition.inference_with_ResNet18(model, recognition.labels, cell_crop)
        results[(x_left, y_top)] = label

    # SHOW RESULTS
    for cell_coords in results:
        x = cell_coords[0]
        y = cell_coords[1]
        label = results[cell_coords]
        cv2.rectangle(img, (x, y), (x + 575, y + 575), (0, 255, 0), 20)
        cv2.putText(img, label, (x, y - 30), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0), 25)
        cv2.putText(img, label, (x, y - 30), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 255, 255), 8)

    img = cv2.resize(img[600:4700, :, :], None, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
    cv2.imshow("Results", img)
    cv2.imwrite('/home/pachy/Desktop/ACSAI/bloodyai/classification/results/artifact.png', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    detect_and_classify("/home/pachy/Desktop/ACSAI/bloodyai/dataset/dl.raabindata.com/WBC/First_microscope/95-5-11-1/images/20160801_103240.jpg")
