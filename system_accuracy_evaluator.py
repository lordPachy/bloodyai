import os
import cv2
import json
import classification
import detection


labels_list = ['Neutrophil', 'Small Lymph', 'Large Lymph', 'Monocyte', 'Band', 'Eosinophil', 'Artifact', 'Unknn', 'Burst', 'Not centered', 'Meta' , 'None']

'''
<summary> System accuracy testing
Input:
    -external_dir_path: path containing all the folders with the images
    -path_to_parameters (in classification.prepare_ResNet18): path to the trained model parameters for ResNet-18
Output:
    -number of WBCs seen
    -number of WBCs correctly labeled
    -accuracy
    -number of valid images used for assessing accuracy
Description:
This script tests the entire system accuracies. It employs functions from (and described in) classification
and detection in order to perform the entire inference over multiple folder containing the images.
IMPORTANT: this program was designed to run on CUDA and it probably won't work without.
</summary>
'''


def detect_and_classify(model, filepath: str):
    # LOAD DATA
    img = cv2.imread(filepath)
    labels = []

    # PERFORM DETECTION AND CLASSIFICATION
    centroids = detection.detect(img)
    for coordinates in centroids:
        x_left = int(coordinates[0]) - 287
        y_top = int(coordinates[1]) - 287
        cell_crop = img[y_top:y_top + 575, x_left:x_left + 575, :]
        #print(cell_crop.shape)
        labels.append(classification.inference_with_ResNet18(model, classification.labels, cell_crop))

    return labels

def image_count_accuracy(model, image_dir_path, json_dir_path):
    images_dir = os.listdir(image_dir_path)
    image_count = len(images_dir)
    print("### Folder: " + image_dir_path)
    index = 0
    count = 0
    correct = 0
    images_considered = 0
    # Iterate over all image files
    for image_file in images_dir:
        if index == 30: break
        flag = True
        count_image = 0
        image_path = image_dir_path + '/' + image_file
        json_path = json_dir_path + '/' + image_file[0:-4] + '.json'

        # Open the respective json file
        f = open(json_path)
        data = json.load(f)
        num_cells = int(data['Cell Numbers'])
        #print(num_cells)
        print("Checking image: " + str(index) + "/" + str(image_count))
        index += 1
        # Iterate over all cells in the image
        label_for_cell = {'Neutrophil': 0, 'Small Lymph' : 0, 'Large Lymph': 0, 'Monocyte': 0, 'Eosinophil': 0, 'Artifact': 0, 'Burst': 0}
        for i in range(0, num_cells):
            # Get cropping coordinates
            cell_data = data['Cell_' + str(i)]
            l1 = cell_data['Label1']
            l2 = cell_data['Label2']
            if l1 == l2 and l1 != None:
                #Consider only valid images
                if l1 in ['Neutrophil', 'Small Lymph', 'Large Lymph', 'Monocyte', 'Eosinophil', 'Artifact', 'Burst']:
                    label_for_cell[l1] += 1
                    count_image += 1
                else:
                    count_image = 0
                    flag = False
                    break
        
        if flag:
            count += count_image
            images_considered +=1
            pred = detect_and_classify(model, image_path)
            if pred is not None:
                for word in pred:
                    if word in ['Neutrophil', 'Small Lymph', 'Large Lymph', 'Monocyte', 'Eosinophil', 'Artifact', 'Burst']:
                        if label_for_cell[word] > 0:
                            label_for_cell[word] -= 1
                            correct += 1 

    return count, correct, images_considered

if __name__ == '__main__':
    model = classification.prepare_ResNet18('classification/parameters/scheduler_wd.pt')
    external_dir_path = 'dataset/dl.raabindata.com/WBC/First_microscope'
    external_dir = os.listdir(external_dir_path)
    count_tot = 0
    correct_tot = 0
    index = 0
    total_images = 0
    # Go through all folders and perform the cropping and saving
    for data_batch_dir in external_dir:
        #if index == 10:
        #  break
        if data_batch_dir in ['None', 'Unknn', 'Results']:
            continue
        data_batch_path = external_dir_path + '/' + data_batch_dir
        images_dir_path = data_batch_path + '/images'
        jsons_dir_path = data_batch_path + '/jsons'
        results = image_count_accuracy(model, images_dir_path, jsons_dir_path)
        count_tot += results[0]
        correct_tot += results[1]
        total_images += results[2]
        index += 1
    
    print('Number of cells checked: ' + str(count_tot))
    print('Number of cells correctly labeled: ' + str(correct_tot))
    print('Accuracy: ' +  str(correct_tot/count_tot))
    print('Number of images considered: ' + str(total_images))
