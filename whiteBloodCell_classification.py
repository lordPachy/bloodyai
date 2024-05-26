import cv2
import detection
import recognition


def detect_and_classify(filepath: str):
    # LOAD DATA
    model = recognition.prepare_ResNet18('./classification/parameters/state_dict_model.pt')
    img = cv2.imread(filepath)
    centroids = detection.detect(img)
    results = dict()

    # PERFORM DETECTION AND CLASSIFICATION
    for coordinates in centroids:
        x_left = int(coordinates[0]) - 287
        y_top = int(coordinates[1]) - 287
        cell_crop = img[y_top:y_top + 575, x_left:x_left + 575, :]
        label = recognition.inference_with_ResNet18(model, recognition.labels, cell_crop)
        results[(x_left, y_top)] = label

    # SHOW RESULTS
    for cell_coords in results:
        x = cell_coords[0]
        y = cell_coords[1]
        label = results[cell_coords]
        cv2.rectangle(img, (x, y), (x + 575, y + 575), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3)
    cv2.imshow("Results", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    detect_and_classify("[filepath]")
