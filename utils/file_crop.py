import os
import cv2
import json

labels_list = ['Neutrophil', 'Small Lymph', 'Large Lymph', 'Monocyte', 'Band', 'Eosinophil', 'Artifact', 'Unknn', 'Burst', 'Not centered', 'Meta' , 'None']
external_dir_path = '/home/pachy/Desktop/ACSAI/bloodyai/dataset/dl.raabindata.com/WBC/Second_microscope'

'''
<summary> Image Crop and Label
Input:
- image_dir_path:  A path to the directory with the uncropped images;
- json_dir_path:   A path to the directory with the respective .json files;
Output: void
Description:
This function goes through the images in image_dir_path and using the data from 
the respective .json file, crops the images and stores them accordingly in the
folder of the label it belongs to.
Filename format: /label/previousFileName_cellNumber.jpg
</summary>
'''
def image_crop(image_dir_path, json_dir_path):
    images_dir = os.listdir(image_dir_path)
    image_count = len(images_dir)
    print("### Folder: " + image_dir_path)
    index = 1
    # Iterate over all image files
    for image_file in images_dir:
        image_path = image_dir_path + '/' + image_file
        json_path = json_dir_path + '/' + image_file[0:-4] + '.json'

        # Open the respective json file
        f = open(json_path)
        data = json.load(f)
        img = cv2.imread(image_path)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        num_cells = int(data['Cell Numbers'])
        print("Cropping image: " + str(index) + "/" + str(image_count))
        index += 1
        # Iterate over all cells in the image
        for i in range(0, num_cells):
            # Get cropping coordinates
            cell_data = data['Cell_' + str(i)]
            x1 = int(cell_data['x1'])
            x2 = int(cell_data['x2'])
            y1 = int(cell_data['y1'])
            y2 = int(cell_data['y2'])
            # Crop
            new_img = img[y1:y2, x1:x2, :]
            # Save file
            l1 = cell_data['Label1']
            l2 = cell_data['Label2']

            if l1 == l2:
                save_dir_path = get_path_from_label(l1)
                save_filename = save_dir_path + image_file[0:-4] + '_cell' + str(i) + '.jpg'
                cv2.imwrite(save_filename, new_img)
            # Make two copies if labels don't correspond
            else:
                save_dir_path = get_path_from_label(l1)
                save_filename = save_dir_path + image_file[0:-4] + '_cell' + str(i) + '.jpg'
                cv2.imwrite(save_filename, new_img)
                save_dir_path = get_path_from_label(l2)
                save_filename = save_dir_path + image_file[0:-4] + '_cell' + str(i) + '.jpg'
                cv2.imwrite(save_filename, new_img)

def get_path_from_label(label):
    if label is None:
        return (external_dir_path + '/Results/None/')
    return (external_dir_path + '/Results/' + label + '/')

if __name__ == '__main__':
    external_dir = os.listdir(external_dir_path)
    results_dir_path = external_dir_path + '/Results'
    # Make necessary folders
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)
    for label in labels_list:
        label_dir_path = results_dir_path + '/' + label
        if not os.path.exists(label_dir_path):
            os.mkdir(label_dir_path)
    # Go through all folders and perform the cropping and saving
    for data_batch_dir in external_dir:
        data_batch_path = external_dir_path + '/' + data_batch_dir
        images_dir_path = data_batch_path + '/images'
        jsons_dir_path = data_batch_path + '/jsons'
        image_crop(images_dir_path, jsons_dir_path)

