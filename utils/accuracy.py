import os
import json
import cv2
import detection

total_cells_count_truth = 0
total_cells_detected_count = 0
bad_labels = ['Band', 'Unknn', 'Not centered', 'Meta', 'None']


def collect_accuracy_per_folder(data_batch_path):
    # LOAD DATA
    images_dir_path = data_batch_path + '/images'
    jsons_dir_path = data_batch_path + '/jsons'
    images_dir = os.listdir(images_dir_path)
    image_count = len(images_dir)
    index = 1
    correct_count = 0
    over_count = 0
    under_count = 0
    report = "Filename \t Detected / Truth \n"
    for image_file in images_dir:
        print(str(index) + "/" + str(image_count))
        index += 1
        image_path = images_dir_path + '/' + image_file
        json_path = jsons_dir_path + '/' + image_file[0:-4] + '.json'

        # PERFORM DETECTION
        f = open(json_path)
        data = json.load(f)
        img = cv2.imread(image_path)
        cell_count_truth = int(data['Cell Numbers'])
        f.close()
        results = detection.detect(img)

        # INCREMENT COUNTERS ACCORDINGLY
        detection_cell_count = len(results)
        if detection_cell_count == cell_count_truth:
            correct_count += 1
        elif detection_cell_count > cell_count_truth:
            over_count += 1
        elif detection_cell_count < cell_count_truth:
            under_count += 1
        report += image_file + "\t" + str(detection_cell_count) + " / " + str(cell_count_truth) + "\n"

    # WRITE AND SAVE RESULTS
    report += "Correct count: " + str(correct_count) + "\n"
    report += "Over count: " + str(over_count) + "\n"
    report += "Under count: " + str(under_count) + "\n"
    filename = data_batch_path.split("/")[-1]
    with open("[Results directory]" + filename + ".txt", "w") as f:
        f.write(report)


def collect_accuracy_total(data_batch_path):
    global total_cells_count_truth
    global total_cells_detected_count
    print(data_batch_path)
    # LOAD DATA
    images_dir_path = data_batch_path + '/images'
    jsons_dir_path = data_batch_path + '/jsons'
    images_dir = os.listdir(images_dir_path)
    image_count = len(images_dir)
    index = 1
    for image_file in images_dir:
        print(str(index) + "/" + str(image_count))
        index += 1
        image_path = images_dir_path + '/' + image_file
        json_path = jsons_dir_path + '/' + image_file[0:-4] + '.json'

        f = open(json_path)
        data = json.load(f)
        f.close()
        cell_count_truth = int(data['Cell Numbers'])
        # CHECK IF IMAGE SHOULD BE DISREGARDED
        skip = False
        for i in range(0, cell_count_truth):
            cell_data = data['Cell_' + str(i)]
            l1 = cell_data['Label1']
            l2 = cell_data['Label2']
            if l1 != l2:
                skip = True
                break
            if l1 in bad_labels:
                skip = True
                break
        if skip:
            continue

        # PERFORM DETECTION AND INCREMENT COUNTERS
        total_cells_count_truth += cell_count_truth
        img = cv2.imread(image_path)
        results = detection.detect(img)
        detection_cell_count = len(results)
        if detection_cell_count <= cell_count_truth:
            total_cells_detected_count += detection_cell_count
            continue
        if (detection_cell_count - cell_count_truth)/ float(cell_count_truth) <= 0.5:
            total_cells_detected_count += cell_count_truth


if __name__ == "__main__":
    data_path = "[Data directory]"
    data_dir = os.listdir(data_path)
    for batch in data_dir:
        if batch == "Results":
            continue
        collect_accuracy_total(data_path + "/" + batch)
    print("--- FINAL RESULT ---")
    print("Cells Found: " + str(total_cells_detected_count))
    print("Ground Truth: " + str(total_cells_count_truth))
